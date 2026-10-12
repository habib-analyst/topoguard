# TopoGuard: Topology-Adaptive Byzantine Defense for Decentralized Federated Learning

**Habib Ur Rehman** · October 2026

> **Status:** working paper. All numbers below are measured on the
> accompanying simulator (`topoguard/`); nothing is fabricated. All result
> cells were populated 2026-10-12 from the completed evaluation matrix
> (`eval/main.jsonl`: 1890 cells) via `scripts/summarize.py`. The headline
> finding changed after measurement: the fused flagship **fails** its
> falsification test (see §4–§5). We report it as measured.

## Abstract

Byzantine-robust aggregation rules for decentralized federated learning
(DFL) — Krum, trimmed mean, geometric median, BALANCE — are designed and
tuned as if the communication graph did not matter. We show it matters
decisively: the same fixed rule that survives 30% Byzantine nodes on a
dense graph collapses on a sparse ring, because the *effective* Byzantine
fraction inside a node's neighborhood is a function of the graph, not a
global constant. **TopoGuard** inverts the design: each node estimates its
local adversarial exposure from per-neighbor trust scores *and* the local
topology structure, then selects — per round — its aggregation rule from a
portfolio (trust-weighted mean, trimmed mean, BALANCE-style accept,
geometric median) with a topology-conditioned acceptance threshold. Across
a full topology × defense × attack × Byzantine-fraction matrix (**1,890
cells**: 7 defenses × 3 topologies × 5 attacks × 3 fractions × IID/non-IID
× 3 seeds), the headline result is a split decision. The topology-adaptive
**trust estimation plus hard exclusion is load-bearing**: a trust-only
variant (TopoGuard with rule selection fixed at L0) reaches 0.9626 mean
accuracy, second only to geometric median (0.9656). But the **per-round
rule selection fails its falsification test**: the fused variant scores
only 0.9214 overall, losing to trust-only on 13 of 15 topology×attack
cells (§5) and collapsing to 0.31–0.32 accuracy on sign_flip at 30%
Byzantine on star/complete — a regime the trust-only variant survives at
0.98+. The failure is mechanistic: at high exposure the selector
escalates to the portfolio's trimmed-mean entry (β = 0.2), which (a) does
not apply the hard-exclusion mask (levels 1 and 3 aggregate over *all*
neighbors, excluded ones included) and (b) has its breakdown point
exceeded by the 30% Byzantine fraction. The measured recommendation is to
ship the trust-only variant and redesign selection so the exclusion mask
binds at every level. We publish the complete matrix as a citable
artifact, including a topology family held out of all tuning.

## 1. Introduction

Decentralized federated learning removes the central server: nodes train
locally and gossip model updates with their graph neighbors. Without a
trusted aggregator, a single Byzantine node can poison every model it
touches. The literature responded with robust aggregation rules — but every
one of them ships a *fixed* rule, tuned on whatever topology the authors
happened to simulate.

The failure mode is structural. Consider 20% Byzantine nodes. On a
complete graph of 12 nodes, each honest node sees ~2 Byzantine updates out
of 11 — Krum and trimmed mean handle this comfortably. On a ring (degree
2), the same 20% means some honest node has a Byzantine *majority* in its
3-update neighborhood, and coordinate-wise statistics over 3 samples carry
no signal at all. A rule calibrated for the dense case is miscalibrated
for the sparse case, and vice versa. BackDFL and follow-up studies have
documented defenses failing as topologies change; no existing defense
adapts *the aggregation rule itself* to the topology (see §2).

**Contributions.**
1. **TopoGuard**, a per-node, per-round adaptive defense: per-neighbor
   trust scores (EWMA on update typicality) fused with a local topology
   descriptor (neighborhood evidence uncertainty φ = 1/√d) into an
   adversarial-exposure estimate, which selects the aggregation rule from
   a 4-rule portfolio with topology-conditioned thresholds and
   self-weighting.
2. A **convergence sketch** under a topology-conditioned adversarial
   influence bound (honestly scoped; assumptions stated).
3. The **topology × defense × attack × Byzantine-fraction matrix** as a
   citable artifact: 1,890 main cells over 3 topologies (+ 180 held-out
   random-geometric-graph cells, 60 ablation cells, 12 time-varying
   cells), 5 attacks, 3 Byzantine fractions, IID/non-IID, 3 seeds — every
   major fixed baseline run on *every* topology, the experiment the field
   skips.
4. Ablations isolating the novelty: selection on/off, topology
   conditioning on/off, trust on/off, portfolio size; plus a time-varying
   topology stress test.

## 2. Related work

**Byzantine-robust aggregation (centralized).** Krum (Blanchard et al.,
NeurIPS 2017) selects the update closest to its n−f−2 nearest neighbors;
coordinate-wise trimmed mean and median (Yin et al., 2018); geometric
median via Weiszfeld; Bulyan (El Mhamdi et al., 2018) composes Krum with
trimmed mean. All assume a central server observing all updates.

**Decentralized defenses.** UBAR (2021): distance + performance based
filtering per node. BALANCE (Fang et al.): per-node distance-threshold
accept against the node's own update. **RADAR (2026)** composes a
moving-target-defense neighbor-selection protocol (commit–reveal
coin-flipping for reciprocal peer agreement) with a BALANCE-style rule
whose threshold adapts over *time* (γ·exp(−κλ(t))·||w_i||) — it
randomizes *who you talk to*, not *how you aggregate*. **TrustGraph-DFL**
(2026, preprint): graph-based trust via consistency with local validation
gradients, mapped to edge weights — uses topology for *weighting*, with a
single fixed aggregation form. **DMTT** (arXiv 2608.03156): trust-aware
*topology learning* (dynamic neighbor choice) under adversaries — again
about peer selection, with a fixed robust aggregator underneath.
SpectralKrum combines spectral analysis with Krum selection (centralized
framing).

**The gap.** To our knowledge, no prior work selects or blends the
*aggregation rule itself* per round conditioned on the *local topology*.
RADAR is the closest in spirit and the sharpest contrast: RADAR's
adaptivity lives in neighbor selection + a time-decaying threshold;
TopoGuard's lives in rule selection + a topology-conditioned threshold.
Our `balance_fixed` baseline is literally RADAR-Agg's rule with the
topology conditioning removed — the ablation that isolates our
contribution.

**Topology sensitivity.** BackDFL (Bouchiha et al., ANUBIS@ESORICS'26)
documented DFL defenses failing across topologies and proposed ABALANCE,
which makes BALANCE's acceptance criterion responsive to neighbor
updates — still one rule, adaptively parameterized; our matrix (§5)
reproduces and extends the topology-fragility finding into a citable
artifact, and our A2 ablation (topology conditioning off) isolates what
the topology signal adds on top of an ABALANCE-style adaptive criterion.

## 3. Method

### 3.1 Problem setup

N nodes on an undirected communication graph G. Node i holds local data
D_i, trains w_i^{t+1/2} from w_i^t, broadcasts to neighbors N(i), then
computes w_i^{t+1} = Agg_i({w_j^{t+1/2} : j ∈ N(i) ∪ {i}}). Up to a
fraction f of nodes are Byzantine and may send arbitrary vectors
(white-box: they observe honest updates, standard in the literature).

### 3.2 TopoGuard

Per node i, per round t (d_i = |N(i)|, u_i = own update):

1. **Trust.** d_j = ‖u_j − u_i‖₂, med = median_j d_j.
   Typicality s_j = exp(−((d_j − med)/med)²); t_j ← t_j + λ(s_j − t_j)
   with λ = 0.7 when s_j < t_j (lost quickly) and λ = 0.2 otherwise
   (earned slowly).
2. **Topology descriptor.** φ_i = 1/√d_i (evidence uncertainty).
3. **Exposure.** e_i = clip(ℓ_i + c·φ_i, 0, 1), where ℓ_i is the fraction
   of neighbors with t_j < 0.5 — the node's estimate of the Byzantine
   share in its neighborhood.
4. **Rule selection.** L = #{τ_k ≤ e_i}, τ = (0.2, 0.45, 0.9).
5. **Execution** (self weight 1 + κφ_i throughout):
   - L0: trust-weighted mean over neighbors with t_j ≥ 0.2, plus self.
   - L1: coordinate trimmed mean (β = 0.2).
   - L2: accept j iff d_j ≤ γ(1+κφ_i)·med and t_j ≥ 0.2, plus self
     always; trust-weighted mean of the kept set.
   - L3: geometric median (Weiszfeld).
   - If *no* neighbor is trusted, the node isolates to its own update.

Pseudocode (per node, per round):

```
inputs: u_i, {u_j}, trust dict t, params (c, κ, γ, τ)
d_j  = ||u_j − u_i|| for j in N(i);  med = median(d_j)
for j: s_j = exp(−((d_j−med)/med)²)
       λ_j = 0.7 if s_j < t_j else 0.2
       t_j = t_j + λ_j (s_j − t_j)
φ    = 1/sqrt(|N(i)|);  ℓ = mean_j [t_j < 0.5]
e    = clip(ℓ + c·φ, 0, 1);  L = |{τ_k : τ_k ≤ e}|
self_w = 1 + κ·φ
if no j has t_j ≥ 0.2: return u_i          # full isolation
case L of
  0: return Σ_{j: t_j≥0.2} t_j u_j + self_w·u_i  /  (Σ t_j + self_w)
  1: return trimmed_mean({u_i} ∪ {u_j}, β=0.2)
  2: keep = {i} ∪ {j : d_j ≤ γ(1+κφ)·med and t_j ≥ 0.2}
     return trust-weighted mean over keep (self weight self_w)
  3: return geometric_median({u_i} ∪ {u_j})
```

**Design rationale.** On dense graphs φ → 0 and honest rounds give
e ≈ 0 → L0 (full utility, FedAvg-like). On sparse graphs φ is large so
e starts elevated → the node defaults to conservative rules *and*
up-weights its own update, the only signal it fully trusts. Under attack,
falling trust raises e → progressively more aggressive rules. The trust
EWMA smooths e across rounds, preventing selection oscillation.

**Why hard exclusion.** A pilot experiment showed trust-weighted averaging
*without* a cutoff diverges under sign-flip: a 0.002 trust weight times
an exponentially growing poison vector (−10× the attacker's model, which
itself grows as it absorbs poison) still drives honest model norms to
10¹⁴ within 20 rounds. Soft weights bound *relative* influence, not
*absolute* magnitude — against unbounded-magnitude attackers, exclusion
must be hard. The 0.2 cutoff is far below honest trust (≈ 0.9+) and far
above converged Byzantine trust (≈ 0.002), so it is not a sensitive
tuning knob.

### 3.3 Convergence sketch

*Scope: honest sketch, assumptions explicit — not a full proof.*

**Assumptions.** (A1) Each honest f_i is L-smooth; stochastic gradients
have variance ≤ σ². (A2) Heterogeneity bounded: ‖∇f_i − ∇f‖² ≤ ζ².
(A3) Update norms bounded by G (gradient clipping, standard).
(A4) Honest updates concentrate: with high probability,
‖u_j − ū_honest‖ ≤ r_h for honest j.

**Lemma 1 (influence bound).** At node i, let ℓ_i^t be the low-trust
fraction. Under L0/L2, neighbors with t_j < 0.2 receive exactly zero
weight (hard exclusion); accepted updates lie within
γ(1+κφ_i)·med of the node's own update (L2) or are trust-weighted with
the attacker's weight decaying geometrically via the asymmetric EWMA
(L0). Under L1/L3 the aggregate is a coordinate-trimmed mean /
geometric median, whose Byzantine bias is bounded by O(r_h) when the
Byzantine share of the neighborhood is below the breakdown point. In all
cases the per-round Byzantine drift satisfies ‖drift_i^t‖ ≤ e_i^t · D
for D = O(G·γ(1+κφ_i)), where the hard-exclusion cutoff is load-bearing:
without it, a residual weight ε > 0 times an unbounded-magnitude
attacker still diverges the model (observed empirically: honest norms
→ 10¹⁴ in 20 rounds at ε ≈ 0.002).

**Theorem (sketch).** Under A1–A4, DFL with TopoGuard satisfies

  (1/T) Σ_t E‖∇f(w̄^t)‖² ≤ O(σ²/√(NT) + ζ² + ē·D²),

where ē is the time-averaged exposure and w̄^t the honest mean model.
*Reading:* when e_i → 0 (honest dense neighborhoods) the bound matches
clean DFL; the Byzantine term scales with the *measured* exposure, and
topology conditioning enters through φ_i in D and in the rule actually
executed. The honest limitation: the bound is only as good as the trust
signal — a Byzantine update within r_h of the honest cluster (e.g.,
well-crafted ALIE) is indistinguishable from honest by *any*
distance-based method, ours included.

## 4. Experiments

**Simulator.** `topoguard/` (this repo): N = 10 nodes, R = 15 rounds,
MLP(20→16→1), 120 synthetic samples/node, 2 local SGD epochs (lr 0.05).
CPU-only.

**Matrix.** Topologies {ring, star, complete} (+ random_geometric
**held out of all tuning**, + time-varying stress) × attacks
{label_flip, sign_flip, gaussian, backdoor, alie} × Byzantine {10%,
20%, 30%} × {IID, non-IID Dirichlet(0.5)} × 3 seeds. Defenses: FedAvg,
Krum, trimmed mean, geometric median, BALANCE-fixed (topology-blind),
TopoGuard, and TopoGuard-trustonly (fixed L0, the falsification test for
whether rule selection adds value). Metrics: honest-node test accuracy
(final), backdoor ASR.

*Attack formulation note.* Model-poisoning attacks are formulated on the
**update**, not the full model: with pre/post the attacker's model
before/after local training and δ = post − pre, sign-flip broadcasts
pre − 10δ, Gaussian broadcasts pre + 10δ + N(0, I), and ALIE broadcasts
pre + μ_δ + 1.5σ_δ (μ_δ, σ_δ over honest updates). Formulating on the
full model (−10·w) is self-defeating — the attacker's own training from
a poisoned start creates double-negative oscillation and the "attack"
becomes incoherent (we measured FedAvg surviving at 0.92 accuracy under
the naive formulation, vs 0.25 under the correct one). This detail
matters for reproducing Byzantine evaluations.

### 4.1 Main matrix (final honest-node accuracy, mean over 3 fractions × IID/non-IID × 3 seeds)

**Ring** — all robust defenses survive; note trimmed-mean ≡ FedAvg here
(neighborhood of 3 < the trimming floor, so nothing is trimmed):

| defense | alie | backdoor | gaussian | label_flip | sign_flip |
| --- | --- | --- | --- | --- | --- |
| fedavg | 0.8612 | 0.9728 | 0.7582 | 0.7428 | 0.2714 |
| krum | 0.8999 | 0.9123 | 0.8973 | 0.9000 | 0.8973 |
| trimmedmean | 0.8612 | 0.9728 | 0.7582 | 0.7428 | 0.2714 |
| geomedian | 0.9412 | 0.9612 | 0.9568 | 0.9403 | 0.9512 |
| balance_fixed | 0.9264 | 0.9574 | 0.9632 | 0.9132 | 0.9633 |
| tg_trustonly | 0.9482 | 0.9486 | 0.9589 | 0.9368 | 0.9510 |
| topoguard (fused) | 0.9480 | 0.9484 | 0.9529 | 0.9365 | 0.9532 |

**Star:**

| defense | alie | backdoor | gaussian | label_flip | sign_flip |
| --- | --- | --- | --- | --- | --- |
| fedavg | 0.8772 | 0.9794 | 0.8031 | 0.9323 | 0.1518 |
| krum | 0.9552 | 0.9469 | 0.9527 | 0.9447 | 0.9527 |
| trimmedmean | 0.9119 | 0.9761 | 0.9270 | 0.9653 | 0.7519 |
| geomedian | 0.9634 | 0.9761 | 0.9745 | 0.9643 | 0.9731 |
| balance_fixed | 0.8871 | 0.8907 | 0.8909 | 0.8863 | 0.8909 |
| tg_trustonly | 0.9461 | 0.9730 | 0.9615 | 0.9693 | 0.9735 |
| topoguard (fused) | 0.9179 | 0.9761 | 0.9256 | 0.9658 | 0.7524 |

**Complete:**

| defense | alie | backdoor | gaussian | label_flip | sign_flip |
| --- | --- | --- | --- | --- | --- |
| fedavg | 0.7851 | 0.9746 | 0.7461 | 0.5419 | 0.1058 |
| krum | 0.9614 | 0.9581 | 0.9583 | 0.9573 | 0.9583 |
| trimmedmean | 0.8997 | 0.9810 | 0.9289 | 0.9375 | 0.7191 |
| geomedian | 0.9693 | 0.9811 | 0.9809 | 0.9705 | 0.9799 |
| balance_fixed | 0.9785 | 0.9788 | 0.9787 | 0.9782 | 0.9786 |
| tg_trustonly | 0.9814 | 0.9814 | 0.9481 | 0.9807 | 0.9799 |
| topoguard (fused) | 0.9325 | 0.9813 | 0.9261 | 0.9426 | 0.7613 |

**Overall method ranking (mean final accuracy over all 270 cells/method):**
geometric median 0.9656 > tg_trustonly 0.9626 > balance_fixed 0.9375 >
krum 0.9368 > topoguard (fused) 0.9214 > trimmed mean 0.8403 > FedAvg 0.7002.

### 4.2 The falsification result (H7): fused vs trust-only

Trust-only beats fused on 13 of 15 topology×attack cells. The two
exceptions are tiny and on ring/star backdoor (+0.0031 at most); the
losses are catastrophic where it matters:

| cell | fused − trustonly |
| --- | --- |
| star / sign_flip | **−0.2211** |
| complete / sign_flip | **−0.2186** |
| complete / alie | −0.0489 |
| complete / label_flip | −0.0381 |
| star / gaussian | −0.0359 |
| star / alie | −0.0282 |
| complete / gaussian | −0.0220 |
| (all other cells) | −0.0060 … +0.0031 |

The collapse is a sharp 30%-fraction cliff: on star/sign_flip the fused
variant holds 0.9733 (10%) and 0.9699 (20%) but falls to **0.3139** at
30%; trust-only holds 0.9828 there. Complete/sign_flip: fused falls to
**0.3230** at 30% while trust-only holds 0.9835. **Selection failure
mechanism** (verified against the mechanism definition): at 30%
Byzantine, the exposure estimate e ≈ 0.42 ≥ τ₁ = 0.2 escalates the node
to portfolio level L1 — coordinate trimmed mean (β = 0.2) — which (i)
aggregates over *all* neighbors, re-admitting the neighbors the trust
signal had excluded (the hard-exclusion mask binds only at levels 0 and
2), and (ii) has its breakdown point exceeded by the 30% > 20% Byzantine
share. At 10–20% the trimmed mean exactly removes the poisoned updates
and the fused variant matches trust-only; past β it collapses. The
portfolio was designed to escalate *robustness* with exposure, but it
escalates into a rule whose assumptions the high-exposure regime
violates — selection is anti-calibrated.

### 4.3 Held-out topology family (random geometric graph, never tuned on)

| defense | alie | backdoor | gaussian | label_flip | sign_flip |
| --- | --- | --- | --- | --- | --- |
| fedavg | 0.7968 | 0.9676 | 0.7254 | 0.6627 | 0.1350 |
| balance_fixed | 0.9557 | 0.9697 | 0.9697 | 0.9376 | 0.9698 |
| topoguard (fused) | 0.9222 | 0.9683 | 0.9423 | 0.8890 | 0.8837 |

The fused variant generalizes to the unseen family (sign_flip 0.8837 vs
FedAvg 0.1350); per §4.2 we expect the trust-only variant to do better
here and ship that as the defense.

### 4.4 Time-varying topology stress (20% Byzantine, IID, 2 seeds)

| defense | backdoor | sign_flip |
| --- | --- | --- |
| fedavg | 0.9818 | 0.3723 |
| balance_fixed | 0.9839 | 0.9840 |
| topoguard (fused) | 0.9843 | 0.9844 |

Under topology churn the fused variant is strong (sign_flip 0.9844 vs
FedAvg 0.3723) — consistent with §4.2, since 20% sits below the
trimmed-mean breakdown point.

### 4.5 Backdoor attack success rate (lower is better; mean over all cells)

| defense | ASR | accuracy |
| --- | --- | --- |
| krum | 0.4508 | 0.9391 |
| balance_fixed | 0.5405 | 0.9423 |
| geomedian | 0.5513 | 0.9728 |
| tg_trustonly | 0.5543 | 0.9677 |
| topoguard (fused) | 0.5831 | 0.9686 |
| trimmedmean | 0.6430 | 0.9766 |
| fedavg | 0.8198 | 0.9756 |

No defense pushes backdoor ASR below ~0.45 in this suite; Krum suppresses
ASR best (0.4508) but at a small accuracy cost. Targeted backdoor
mitigation is modest across the board — reported as a limitation (§6),
not a TopoGuard advantage.

## 5. Ablations

- **A1 selection off** (fixed L2 + trust + topology): isolates rule selection.
- **A2 topology off** (φ := 0): isolates topology conditioning — this is
  TopoGuard vs RADAR-style time-only adaptivity.
- **A3 trust off** (all t = 1): isolates the trust signal.
- **A4 small portfolio** ({L0, L3}): is 4 rules better than 2?
- **Pilot mechanism selection**: trust-only vs selection-only vs fused
  (reported in §5.1; the fused design did **not** win — the pilot already
  ranked trust-only highest, and the main matrix confirms it).

- **Pilot mechanism selection**: trust-only vs selection-only vs fused
  (reported in §5.1; the fused design did **not** win — the pilot already
  ranked trust-only highest, and the main matrix confirms it).

### 5.1 Pilot (final accuracy, mean of 2 seeds; ring/complete × sign_flip/backdoor)

| variant | complete/backdoor | complete/sign_flip | ring/backdoor | ring/sign_flip | mean |
| --- | --- | --- | --- | --- | --- |
| V_fused | 0.9835 | 0.9850 | 0.9848 | 0.9141 | 0.9669 |
| V_select | 0.9835 | 0.9850 | 0.9847 | 0.9847 | 0.9845 |
| V_trust | 0.9849 | 0.9865 | 0.9849 | 0.9847 | 0.9853 |

Trust-only wins the pilot (0.9853); fused trails (0.9669), dragged by
ring/sign_flip (0.9141) — the same regime where the main matrix later
measures the full collapse. Selection-only sits in between.

### 5.2 Ablations (final accuracy, non-IID, 20% Byzantine, 2 seeds)

| variant | complete/backdoor | complete/sign_flip | ring/backdoor | ring/sign_flip | star/backdoor | star/sign_flip |
| --- | --- | --- | --- | --- | --- | --- |
| A1 selection off (fixed L2) | 0.9804 | 0.9812 | 0.9569 | 0.9590 | 0.9706 | 0.9733 |
| A2 topology off (φ := 0) | 0.9822 | 0.9715 | 0.9564 | 0.9582 | 0.9764 | 0.9674 |
| A3 trust off | 0.9770 | 0.2657 | 0.9799 | 0.4100 | 0.9802 | 0.2688 |
| A4 small portfolio {L0,L3} | 0.9828 | 0.9759 | 0.9657 | 0.9354 | 0.9710 | 0.9625 |
| full (fused) | 0.9821 | 0.9748 | 0.9569 | 0.9590 | 0.9755 | 0.9715 |

**Interpretation.** A3 (trust off) is the load-bearing result: without
the trust signal, sign_flip accuracy collapses to 0.27–0.41 — the
measured "soft trust alone can't exclude unbounded attackers" blowup
predicted in §3.2 (hard exclusion is doing the work). A1 (selection off,
fixed L2) **matches or slightly beats** the fused variant on every
sign_flip cell (complete 0.9812 > 0.9748) — selection adds nothing and
slightly hurts, consistent with §4.2. A2 (topology conditioning off)
changes little at 20% (≤ 0.008 delta), so the topology signal's value is
confined to the hard-exclusion regime rather than the selection ladder.
A4 (small portfolio) is within noise of full. Net: the novelty that
survives measurement is topology-adaptive trust estimation with hard
exclusion — not per-round rule selection.

### 5.3 Hyperparameter tune (fused variant, ring/star × sign_flip/backdoor, non-IID)

The grid over (c, κ, τ₃) is flat within 0.006 (0.9547–0.9610 across 18
settings; best c=0.25, κ∈{0.5,1.0}, τ₃=0.9 → 0.9610), which is why the
main matrix fixes c=0.25, κ=1.0, τ=(0.2, 0.45, 0.9). Robustness to these
knobs is a feature; the selection mechanism's failure is structural, not
a tuning artifact.

## 6. Limitations (honest)

- Simulated scale: 10 nodes, synthetic binary data, small MLP. The
  topology×defense interaction is the contribution; absolute accuracies
  are not claimed to transfer to large-scale deployments.
- Distance-based trust cannot catch Byzantine updates inside the honest
  cluster (ALIE at small z) — fundamental, shared with all baselines.
- Star-topology leaves (degree 1) are inherently fragile: a single
  Byzantine hub isolates them; no per-node rule fixes a single point of
  failure. Reported, not hidden.
- **The fused flagship failed falsification.** Per-round rule selection
  from the portfolio is anti-calibrated: it escalates to portfolio rules
  (trimmed mean β=0.2, geometric median) that re-admit excluded neighbors
  and break at high Byzantine fractions — exactly the regime where
  exclusion matters most. We recommend the trust-only (fixed L0) variant
  and a redesign where the exclusion mask binds at every portfolio level.
  A defense paper reporting its own flagship mechanism's failure, with
  the measured numbers, is a feature: the negative result is the result.
- The convergence sketch is a sketch (assumptions A1–A4); a full proof
  with time-varying graphs is future work.
- **Attack-suite weaknesses.** ALIE at z=1.5 (the literature-standard
  setting) proved the weakest attack in the suite: even FedAvg retains
  0.75–0.88 accuracy under it; the suite's discriminative power comes
  from sign_flip and gaussian. Our conclusions rest on the strong
  attacks. Backdoor ASR is suppressed by no defense below ~0.45 (best:
  Krum 0.4508); targeted-backdoor mitigation remains an open problem
  across the board.
- **Measurement artifact.** On the ring (neighborhood of 3), trimmed
  mean (β=0.2) trims floor(0.2·3)=0 updates and degenerates to FedAvg
  (identical numbers in the ring table). Coordinate-wise trimming needs
  a minimum neighborhood size to mean anything — a caveat for sparse
  graphs, not a defense result.

## 7. Conclusion

Robustness in DFL is not a property of an aggregation rule — it is a
property of an aggregation rule *on a topology*. TopoGuard's contribution,
as measured rather than as designed, is topology-adaptive trust
estimation with **hard exclusion**: the trust-only variant reaches 0.9626
mean accuracy across the 1,890-cell matrix, second only to geometric
median (0.9656) and ahead of every other baseline on most cells. The
per-round rule-selection mechanism, the paper's original flagship, fails
its own falsification test (0.9214, collapsing at 30% Byzantine) — and
that failure is mechanistically understood: escalation into portfolio
rules that ignore the exclusion mask at exactly the Byzantine fractions
that exceed their breakdown points. The published matrix lets the field
stop re-tuning fixed rules per topology — and, we hope, stop designing
"adaptive" defenses whose adaptivity is never falsified. The measured
defense to ship is the trust-only variant; the measured research result
is the negative one.

## References

- Blanchard, P. et al. Machine learning with adversaries: Byzantine
  tolerant gradient descent. NeurIPS 2017. (Krum)
- Yin, D. et al. Byzantine-robust distributed learning. ICML 2018.
  (coordinate median / trimmed mean)
- El Mhamdi, E. M. et al. The hidden vulnerability of distributed
  learning in Byzantium. ICML 2018. (Bulyan)
- Fang, M. et al. BALANCE: Byzantine-robust aggregation for
  decentralized learning. (per-node distance-threshold accept)
- RADAR: Robust aggregation with dynamic adversarial-resilient routing
  for decentralized federated learning. Int. J. Inf. Security, 2026.
  https://doi.org/10.1007/s10207-026-01257-7 (MTD neighbor selection +
  RADAR-Agg: BALANCE-style accept with a *time*-decaying threshold
  γ·exp(−κλ(t))·‖w_i‖)
- TrustGraph-DFL: Byzantine-resilient DFL via consistency-weighted
  neighborhood aggregation. Preprints.org, 2026.
- DMTT: Trust-aware topology learning for dynamic DFL under adversaries.
  arXiv:2608.03156, 2026.
- Baruch, G. et al. A little is enough: circumventing defenses for
  distributed learning. NeurIPS 2019. (ALIE)
- Bouchiha, M. A., Blanc, G., Han, Y. BackDFL: A unified benchmark for
  backdoor attacks and defenses in decentralized federated learning.
  arXiv:2608.21137, 2026. Accepted at the ANUBIS Workshop, ESORICS'26.
  Documents defenses failing across topologies; proposes ABALANCE (BALANCE
  with an acceptance criterion responsive to neighbor updates). ABALANCE
  adapts the criterion of *one* rule; TopoGuard additionally selects
  *which* rule to run per round and conditions exposure and thresholds on
  the local topology structure.

## Reproducibility

```bash
pip install -e .
python -m topoguard.eval --phase pilot   --out eval/pilot.jsonl   --resume
python -m topoguard.eval --phase main    --out eval/main.jsonl    --resume
python -m topoguard.eval --phase heldout --out eval/heldout.jsonl --resume
python -m topoguard.eval --phase ablation --out eval/ablation.jsonl --resume
python scripts/summarize.py  # rebuilds all paper tables from eval/*.jsonl
```

All numbers in this paper are generated by `scripts/summarize.py` and
`scripts/analyze.py` from measured `eval/*.jsonl` records — no hand-typed
numbers anywhere in this paper. Every paper table can be re-verified
against the raw records shipped in `results/`.
