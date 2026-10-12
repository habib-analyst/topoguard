"""Unit tests for TopoGuard (fast, no training loops)."""

import numpy as np
import torch

from topoguard import TopoGuard
from topoguard import topology as topo
from topoguard import defenses as D


def _updates(n, dim=50, seed=0, outlier_idx=None, outlier_scale=10.0):
    rng = np.random.default_rng(seed)
    us = [torch.as_tensor(rng.normal(0, 1, dim), dtype=torch.float32) for _ in range(n)]
    if outlier_idx is not None:
        us[outlier_idx] = torch.as_tensor(
            rng.normal(0, 1, dim) * outlier_scale + 20.0, dtype=torch.float32
        )
    return us


def test_topologies_connected():
    for name in ["ring", "star", "complete", "random_geometric"]:
        g = topo.build(name, 12, seed=0)
        assert len(g.nodes) == 12, name
        # random_geometric is repaired to be connected
        import networkx as nx

        assert nx.is_connected(g), name
    # ring degrees
    g = topo.ring(10)
    assert all(d == 2 for _, d in g.degree())
    # star hub
    g = topo.star(10)
    assert g.degree(0) == 9 and g.degree(5) == 1


def test_time_varying_callable():
    gat = topo.build("time_varying", 10, seed=1)
    g0, g1 = gat(0), gat(1)
    assert len(g0.nodes) == 10 and len(g1.nodes) == 10


def test_trust_downweights_outlier():
    tg = TopoGuard()
    own = torch.zeros(50)
    us = _updates(5, outlier_idx=3)
    nbrs = {i: u for i, u in enumerate(us)}
    _, rep = tg.aggregate(own, nbrs)
    # outlier neighbor (id 3) should have the lowest trust
    trusts = [tg.trust[i] for i in range(5)]
    assert trusts[3] == min(trusts), trusts
    assert rep["low_frac"] >= 0.0


def test_exposure_rises_under_attack():
    tg = TopoGuard()
    own = torch.zeros(50)
    us = _updates(5, outlier_idx=2, outlier_scale=30.0)
    nbrs = {i: u for i, u in enumerate(us)}
    for _ in range(8):  # let asymmetric EWMA settle
        _, rep = tg.aggregate(own, nbrs)
    assert rep["low_frac"] > 0.1, rep
    assert rep["level"] >= 1, rep


def test_clean_round_selects_gentle_level():
    tg = TopoGuard()
    own = torch.zeros(50)
    us = _updates(11)  # complete-ish neighborhood, all honest
    nbrs = {i: u for i, u in enumerate(us)}
    _, rep = tg.aggregate(own, nbrs)
    # dense + honest -> low_frac 0, e = 0.5/sqrt(11) = 0.15 < 0.2 -> L0
    assert rep["level"] == 0, rep


def test_sparse_topology_more_conservative():
    tg = TopoGuard()
    own = torch.zeros(50)
    us = _updates(2)  # ring-like: degree 2, all honest
    nbrs = {0: us[0], 1: us[1]}
    _, rep = tg.aggregate(own, nbrs)
    # phi = 1/sqrt(2) -> e = 0.354 -> L1
    assert rep["level"] == 1, rep
    assert abs(rep["phi"] - 1 / 2**0.5) < 1e-9


def test_hard_exclusion_blocks_unbounded_attacker():
    # regression test: soft weights alone diverge under sign-flip
    # (a 0.002 weight times an exponentially growing poison vector).
    tg = TopoGuard()
    own = torch.zeros(50)
    us = _updates(9)
    us[0] = torch.ones(50) * 1e6  # unbounded-magnitude attacker
    nbrs = {i: u for i, u in enumerate(us)}
    for _ in range(10):
        agg, rep = tg.aggregate(own, nbrs)
    assert tg.trust[0] < 0.2, tg.trust[0]
    # attacker must get zero weight: aggregate stays near honest scale
    assert agg.norm().item() < 10.0, agg.norm().item()


def test_exposure_counts_suspected_byzantine():
    tg = TopoGuard()
    own = torch.zeros(50)
    us = _updates(9, outlier_idx=0, outlier_scale=50.0)
    us.append(_updates(1, outlier_idx=0, outlier_scale=50.0)[0])
    nbrs = {i: u for i, u in enumerate(us)}  # 2 of 10 far outliers
    for _ in range(6):
        _, rep = tg.aggregate(own, nbrs)
    assert rep["low_frac"] >= 0.15, rep
    assert rep["level"] >= 1, rep


def test_level2_rejects_far_outlier():
    tg = TopoGuard(select=False, fixed_level=2)
    own = torch.zeros(50)
    us = _updates(6, outlier_idx=4, outlier_scale=50.0)
    nbrs = {i: u for i, u in enumerate(us)}
    agg, rep = tg.aggregate(own, nbrs)
    assert rep["n_accepted"] < 6, rep
    # aggregate should be close to the honest mean (origin), not the outlier
    assert agg.norm().item() < 5.0


def test_geo_median_robust():
    own = torch.zeros(40)
    us = _updates(7, dim=40, outlier_idx=0, outlier_scale=100.0)
    m = D.geo_median([own] + us, own)
    assert m.norm().item() < 3.0


def test_geo_median_extreme_outlier_few_points():
    # regression: mean-init Weiszfeld collapsed on 3 points with one
    # -10x poison vector (pilot ring/sign_flip seed 0: 0.8576).
    # Median init must keep the median near the honest points.
    a = torch.zeros(40)
    b = torch.randn(40) * 0.1
    c = torch.ones(40) * -1e4  # extreme poison
    m = D.geo_median([a, b, c], a)
    assert m.norm().item() < 5.0, m.norm().item()


def test_krum_picks_honest():
    own = torch.zeros(40)
    us = _updates(7, dim=40, outlier_idx=5, outlier_scale=100.0)
    k = D.krum([own] + us, own, f=1)
    assert k.norm().item() < 3.0


def test_balance_fixed_accepts_honest():
    own = torch.zeros(40)
    us = _updates(6, dim=40)
    b = D.balance_fixed([own] + us, own, gamma=2.0)
    assert b.norm().item() < 3.0


def test_no_neighbors_returns_own():
    tg = TopoGuard()
    own = torch.ones(10)
    agg, rep = tg.aggregate(own, {})
    assert torch.allclose(agg, own.float())
    assert rep["level"] == -1


def test_ablation_flags():
    own = torch.zeros(30)
    us = _updates(5, dim=30, outlier_idx=1)
    nbrs = {i: u for i, u in enumerate(us)}
    # trust off -> all trusts stay 1
    tg = TopoGuard(trust=False)
    _, rep = tg.aggregate(own, nbrs)
    assert rep["low_frac"] == 0.0
    # topology off -> phi 0
    tg2 = TopoGuard(topology=False)
    _, rep2 = tg2.aggregate(own, nbrs)
    assert rep2["phi"] == 0.0
    # small portfolio maps levels to {0, 3}
    tg3 = TopoGuard(portfolio="small")
    assert tg3._level_map(0) == 0 and tg3._level_map(1) == 0
    assert tg3._level_map(2) == 3 and tg3._level_map(3) == 3
