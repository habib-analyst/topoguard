"""Communication topologies for decentralized federated learning.

Each builder returns a networkx Graph over ``n`` nodes (0..n-1). Node 0 is
arbitrary unless noted. Topologies:

- ring: each node connects to k/2 neighbors on each side (k=2 default).
- star: node 0 is the hub.
- complete: all-to-all.
- random_geometric: nodes uniform in [0,1]^2, edges within radius r.
  **Held out of all mechanism tuning** (see paper) — the honest test of
  topology adaptivity.
- time_varying: base ring with per-round edge churn (each edge rewires
  with probability p per round). Returned as a callable ``graph_at(t)``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def ring(n: int, k: int = 2) -> nx.Graph:
    """Ring lattice: each node connects to k/2 neighbors per side."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    half = k // 2
    for i in range(n):
        for d in range(1, half + 1):
            g.add_edge(i, (i + d) % n)
            g.add_edge(i, (i - d) % n)
    return g


def star(n: int) -> nx.Graph:
    """Star: node 0 is the hub, all others are leaves (degree 1)."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(1, n):
        g.add_edge(0, i)
    return g


def complete(n: int) -> nx.Graph:
    """Complete graph: every node sees every other node."""
    return nx.complete_graph(n)


def random_geometric(
    n: int, radius: float = 0.45, seed: int = 0
) -> nx.Graph:
    """Random geometric graph: uniform points in [0,1]^2, edges < radius.

    NOTE: this family is held out of mechanism tuning (see paper Sec. 5).
    """
    rng = np.random.default_rng(seed)
    pos = {i: (rng.random(), rng.random()) for i in range(n)}
    g = nx.random_geometric_graph(n, radius, pos=pos)
    # Ensure connectivity by linking components in a chain.
    comps = list(nx.connected_components(g))
    for a, b in zip(comps, comps[1:]):
        g.add_edge(next(iter(a)), next(iter(b)))
    return g


def time_varying_ring(n: int, churn: float = 0.1, seed: int = 0):
    """Return ``graph_at(t)``: ring with per-round edge churn.

    Each round, every edge is rewired to a random non-edge with
    probability ``churn``. Models peer churn / mobile DFL.
    """
    rng = np.random.default_rng(seed)
    base = ring(n)

    def graph_at(t: int) -> nx.Graph:
        r = np.random.default_rng(seed * 100003 + t)
        g = base.copy()
        edges = list(g.edges())
        for u, v in edges:
            if r.random() < churn:
                g.remove_edge(u, v)
                # rewire u to a random non-neighbor
                candidates = [
                    w
                    for w in range(n)
                    if w != u and not g.has_edge(u, w)
                ]
                if candidates:
                    g.add_edge(u, r.choice(candidates))
        # keep it connected
        comps = list(nx.connected_components(g))
        for a, b in zip(comps, comps[1:]):
            g.add_edge(next(iter(a)), next(iter(b)))
        return g

    return graph_at


TOPOLOGIES = ("ring", "star", "complete", "random_geometric", "time_varying")


def build(name: str, n: int, seed: int = 0):
    """Build a topology by name. Returns Graph, or graph_at callable for time_varying."""
    if name == "ring":
        return ring(n)
    if name == "star":
        return star(n)
    if name == "complete":
        return complete(n)
    if name == "random_geometric":
        return random_geometric(n, seed=seed)
    if name == "time_varying":
        return time_varying_ring(n, seed=seed)
    raise ValueError(f"unknown topology {name!r}")
