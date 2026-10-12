# Pre-registered hypotheses (written before main matrix completed)

## H1: TopoGuard >= tg_trustonly > balance_fixed > krum/trimmed/geomedian > fedavg
On average across the matrix. The topology-adaptive mechanism should beat
fixed rules; selection should add a small bonus over fixed-L0.

## H2: Topology sensitivity
The TopoGuard-vs-best-baseline gap is LARGEST on sparse topologies (ring,
star) and at 30% Byzantine. On complete/10%, all robust methods tie (~0.98)
because the problem is easy.

## H3: Fixed baselines are topology-fragile
Krum/trimmed-mean/geomedian show HIGH variance across topologies (good on
complete, bad on ring/star). TopoGuard shows LOW variance (adapts).

## H4: Backdoor
TopoGuard achieves lower ASR than FedAvg (which should be ~0.9+) and
comparable-or-better than baselines. Backdoor is a data-poisoning attack;
trust may partially catch the trigger direction.

## H5: ALIE
All defenses ~0.98 (attack too weak in this setup). Honest negative result.

## H6: Star + Byzantine hub
All methods fail when the star hub is Byzantine (single point of failure).
This is an honest limitation, not a method flaw.

## H7: Selection value
topoguard - tg_trustonly > 0 on average, but small (< 0.02). Selection
matters most on star/ring at 20-30% where exposure escalates.

## Falsification criteria
- If H1 fails (TopoGuard < best baseline on average): the mechanism doesn't
  work; paper pivots to matrix-as-contribution + honest negative result.
- If H7 fails (selection adds nothing): paper reports it; contribution is
  the trust+topology mechanism and the matrix.
