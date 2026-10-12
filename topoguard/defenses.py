"""Baseline aggregation rules (fixed, topology-agnostic).

Every rule maps a neighborhood's update vectors to one aggregate vector.
Signatures mirror the TopoGuard interface: ``rule(updates, ref, **kw)``
where ``updates`` includes the node's own update and ``ref`` is the node's
own update vector (used as the honest reference by distance-based rules).

- fedavg: plain mean (vulnerable baseline).
- krum: Blanchard et al. 2017, f = expected Byzantine in neighborhood.
- trimmed_mean: coordinate-wise, drop beta fraction per tail.
- geo_median: Weiszfeld geometric median.
- balance_fixed: BALANCE-style distance accept with a FIXED threshold
  (time-adaptive via the median only, like RADAR-Agg, but NOT
  topology-conditioned). This is the key comparison: balance_fixed vs
  TopoGuard's topology-conditioned acceptance isolates the value of
  conditioning on the graph.
"""

from __future__ import annotations

import torch


def _stack(updates: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([u.reshape(-1).float() for u in updates], dim=0)


def fedavg(updates: list[torch.Tensor], ref: torch.Tensor, **kw) -> torch.Tensor:
    return _stack(updates).mean(dim=0)


def trimmed_mean(
    updates: list[torch.Tensor], ref: torch.Tensor, beta: float = 0.2, **kw
) -> torch.Tensor:
    s = _stack(updates)
    n = s.shape[0]
    k = max(0, min(int(beta * n), n // 2 - 1))
    if k == 0:
        return s.mean(dim=0)
    ss, _ = torch.sort(s, dim=0)
    return ss[k : n - k].mean(dim=0)


def geo_median(
    updates: list[torch.Tensor],
    ref: torch.Tensor,
    iters: int = 50,
    eps: float = 1e-8,
    **kw,
) -> torch.Tensor:
    """Weiszfeld geometric median.

    Initialized at the coordinate-wise median (robust to extreme
    outliers, unlike the mean: a single -10x poison vector pulls the
    mean init far enough that few Weiszfeld steps cannot recover).
    """
    s = _stack(updates)
    m = s.median(dim=0).values
    for _ in range(iters):
        d = (s - m).norm(dim=1).clamp_min(eps)
        w = 1.0 / d
        m = (w.unsqueeze(1) * s).sum(dim=0) / w.sum()
    return m


def krum(
    updates: list[torch.Tensor], ref: torch.Tensor, f: int = 1, m: int = 1, **kw
) -> torch.Tensor:
    s = _stack(updates)
    n = s.shape[0]
    if n < 3:
        return s.mean(dim=0)
    keep = max(n - f - 2, 1)
    dists = torch.cdist(s, s, p=2).pow(2)
    nearest, _ = torch.sort(dists, dim=1)
    scores = nearest[:, 1 : keep + 1].sum(dim=1)
    order = torch.argsort(scores)
    return s[order[:m]].mean(dim=0)


def balance_fixed(
    updates: list[torch.Tensor],
    ref: torch.Tensor,
    gamma: float = 2.0,
    **kw,
) -> torch.Tensor:
    """BALANCE-style accept with FIXED (topology-blind) threshold.

    Accept neighbor j if ||u_j - ref|| <= gamma * median_k ||u_k - ref||.
    The threshold adapts to the round's dispersion (like RADAR-Agg) but is
    NOT conditioned on the local topology — the ablation target.
    """
    s = _stack(updates)
    d = (s - ref.reshape(-1)).norm(dim=1)
    med = d.median()
    thresh = gamma * med.clamp_min(1e-12)
    keep = d <= thresh
    # self (index 0) is always kept
    keep[0] = True
    if keep.sum() == 0:
        return ref.reshape(-1)
    return s[keep].mean(dim=0)


BASELINES = {
    "fedavg": fedavg,
    "krum": krum,
    "trimmedmean": trimmed_mean,
    "geomedian": geo_median,
    "balance_fixed": balance_fixed,
}
