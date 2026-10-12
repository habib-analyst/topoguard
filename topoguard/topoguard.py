"""TopoGuard: topology-adaptive Byzantine-robust aggregation for DFL.

The thesis: topology-agnostic defenses fail because the optimal
robustness/utility tradeoff is *topology-dependent*. On a dense graph a
node has rich statistical evidence about its neighborhood; on a sparse
ring (degree 2) the same fixed rule either rejects too much (stalling
learning) or too little (admitting poison). TopoGuard adapts *how it
aggregates* to the local topology, every round, at every node.

Mechanism (per node i, per round t). Let u_i be the node's own update and
{u_j : j in N(i)} the neighbors' updates, d_i = |N(i)|:

1. **Per-neighbor trust.** d_j = ||u_j - u_i||_2, med = median_j d_j.
   Typicality s_j = exp(-((d_j - med)/med)^2): honest neighbors cluster
   near the median distance (s ~ 1); far outliers score near 0. Trust is
   lost quickly and earned slowly (asymmetric EWMA):
   t_j <- t_j + ldown*(s_j - t_j) if s_j < t_j else t_j + lup*(s_j - t_j).
2. **Local topology descriptor.** phi_i = 1/sqrt(d_i): the evidence
   uncertainty of a small neighborhood.
3. **Adversarial exposure.** e_i = clip(lowfrac_i + c*phi_i, 0, 1), where
   lowfrac_i = fraction of neighbors with t_j < 0.5 (the node's estimate
   of the Byzantine share in its neighborhood).
4. **Rule selection.** L = #{tau_k <= e_i}, tau = (0.2, 0.45, 0.7).
5. **Topology-conditioned execution.** Self weight 1 + kappa*phi_i
   throughout (on sparse graphs your own update is the most trustworthy
   signal you have). Neighbors with t_j < 0.2 are hard-excluded from
   L0/L2: soft weights alone cannot fully exclude an unbounded-magnitude
   attacker (a 0.002 weight times an exponentially growing poison vector
   still diverges the model — found empirically, see paper).
   - L0: trust-weighted mean over trusted neighbors + self.
   - L1: coordinate trimmed mean (beta = 0.2).
   - L2: accept j iff d_j <= gamma*(1+kappa*phi_i)*med and t_j >= 0.2,
     plus self always; trust-weighted mean of the kept set.
   - L3: geometric median (Weiszfeld).

Why this is novel: RADAR (2026) randomizes *who you talk to* (MTD neighbor
selection) and uses a *time*-adaptive threshold; TrustGraph-DFL and DMTT
use graph structure for trust/neighbor choice. None selects or blends the
*aggregation rule itself* per round conditioned on the local topology.
See paper Sec. 2 for the full literature positioning.
"""

from __future__ import annotations

import torch

from .defenses import geo_median, trimmed_mean

EPS = 1e-12
TRUST_CUTOFF = 0.5   # below this a neighbor counts as suspected Byzantine
EXCLUDE_CUTOFF = 0.2  # below this a neighbor gets zero weight in L0/L2


class TopoGuard:
    """Stateful per-node TopoGuard aggregator.

    Flags (for pilot variants and ablations):
      select:    rule selection on/off (off -> fixed_level rule always).
      topology:  topology conditioning on/off (off -> phi := 0).
      trust:     trust scores on/off (off -> all trusts = 1).
      portfolio: "full" (L0..L3) or "small" (L0, L3 only).
      fixed_level: rule used when select=False.
    Hyperparameters: c, kappa, gamma, ldown/lup, thresholds tau.
    """

    def __init__(
        self,
        c: float = 0.5,
        kappa: float = 1.0,
        gamma: float = 2.0,
        ldown: float = 0.7,
        lup: float = 0.2,
        tau: tuple[float, float, float] = (0.2, 0.45, 0.7),
        select: bool = True,
        topology: bool = True,
        trust: bool = True,
        portfolio: str = "full",
        fixed_level: int = 2,
    ):
        self.c = c
        self.kappa = kappa
        self.gamma = gamma
        self.ldown = ldown
        self.lup = lup
        self.tau = tau
        self.select = select
        self.topology = topology
        self.trust_on = trust
        self.portfolio = portfolio
        self.fixed_level = fixed_level
        self.trust: dict[int, float] = {}

    def _level_map(self, level: int) -> int:
        if self.portfolio == "small":
            return 0 if level < 2 else 3
        return level

    def aggregate(
        self, own: torch.Tensor, neighbors: dict[int, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        u0 = own.reshape(-1).float()
        nids = list(neighbors.keys())
        d_i = len(nids)
        phi = 0.0 if not self.topology else 1.0 / max(d_i, 1) ** 0.5

        if d_i == 0:
            return u0, {"level": -1, "exposure": 0.0, "phi": phi,
                        "low_frac": 0.0, "n_accepted": 0}

        us = [neighbors[k].reshape(-1).float() for k in nids]
        s = torch.stack([u0] + us, dim=0)  # row 0 = self
        dists = (s[1:] - u0).norm(dim=1)  # neighbor distances to own update
        med = dists.median().clamp_min(EPS)

        # 1-2. trust update (asymmetric: lost quickly, earned slowly)
        z = (dists - med) / med
        sims = torch.exp(-(z**2))
        tvals = []
        for k, sim in zip(nids, sims):
            sim_f = float(sim.item())
            if not self.trust_on:
                t_new = 1.0
            else:
                t_old = self.trust.get(k, 1.0)
                lam = self.ldown if sim_f < t_old else self.lup
                t_new = t_old + lam * (sim_f - t_old)
            self.trust[k] = t_new
            tvals.append(t_new)
        t = torch.tensor(tvals, dtype=torch.float32)
        low_frac = float((t < TRUST_CUTOFF).float().mean().item())

        # 3. exposure
        e = low_frac + self.c * phi
        e = max(0.0, min(1.0, e))

        # 4. rule selection
        if self.select:
            level = sum(1 for th in self.tau if e >= th)
        else:
            level = self.fixed_level
        level = self._level_map(level)

        self_w = 1.0 + self.kappa * phi
        trusted = t >= EXCLUDE_CUTOFF
        n_trusted = int(trusted.sum().item())
        report = {
            "level": level,
            "exposure": e,
            "phi": phi,
            "low_frac": low_frac,
            "n_neighbors": d_i,
        }

        # Full isolation: if no neighbor is trusted, no neighborhood rule
        # can help (any aggregate including them is poisoned). Fall back to
        # the node's own update — it still learns from its local data.
        if n_trusted == 0:
            report["n_accepted"] = 0
            report["isolated"] = True
            return u0, report

        # 5. execute
        if level == 0:
            w = torch.tensor([self_w] + [tv if ok else 0.0
                                         for tv, ok in zip(tvals, trusted)],
                             dtype=torch.float32)
            agg = (w.unsqueeze(1) * s).sum(dim=0) / w.sum().clamp_min(EPS)
            report["n_accepted"] = int(trusted.sum().item())
            return agg, report
        if level == 1:
            agg = trimmed_mean([s[r] for r in range(s.shape[0])], u0, beta=0.2)
            report["n_accepted"] = d_i
            return agg, report
        if level == 2:
            thresh = self.gamma * (1 + self.kappa * phi) * med
            keep_rows = [0] + [
                r + 1 for r, (dd, ok) in enumerate(zip(dists, trusted))
                if dd <= thresh and ok
            ]
            kept = s[keep_rows]
            w = torch.tensor(
                [self_w] + [tvals[r] for r in range(d_i)
                            if (r + 1) in keep_rows],
                dtype=torch.float32,
            )
            agg = (w.unsqueeze(1) * kept).sum(dim=0) / w.sum().clamp_min(EPS)
            report["n_accepted"] = len(keep_rows) - 1
            report["threshold"] = float(thresh.item())
            return agg, report
        # level 3
        agg = geo_median([s[r] for r in range(s.shape[0])], u0)
        report["n_accepted"] = d_i
        return agg, report
