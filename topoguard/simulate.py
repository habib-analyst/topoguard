"""Decentralized FL simulation loop.

Each round: every node trains locally from its current model, broadcasts
the post-training model to its neighbors, then every honest node aggregates
the received models (plus its own) with its defense rule. Byzantine nodes
follow the attack protocol. We track honest-node test accuracy and (for
backdoor) attack success rate.
"""

from __future__ import annotations

import numpy as np
import torch

from . import attacks as atk
from . import topology as topo
from .data import make_federated_data
from .models import MLP, accuracy, flatten_params, load_flat_params
from .topoguard import TopoGuard

MODEL_ATTACKS = ("sign_flip", "gaussian", "alie")
DATA_ATTACKS = ("label_flip", "backdoor")


def _make_defense(name: str, n_neighbors: int, byz_frac: float, seed: int, **kw):
    if name == "topoguard":
        return TopoGuard(**kw.get("topoguard_kw", {}))
    if name == "tg_trustonly":
        # Trust-only variant (pilot): fixed L0, topology-conditioned.
        # In the main matrix to test whether selection adds value.
        base = dict(kw.get("topoguard_kw", {}))
        base.update({"select": False, "fixed_level": 0,
                     "topology": True, "trust": True})
        return TopoGuard(**base)
    from . import defenses as D

    fn = D.BASELINES[name]

    def agg(own, neighbors):
        updates = [own] + [neighbors[k] for k in neighbors]
        f = max(1, int(round(byz_frac * len(updates))))
        return fn(updates, own, f=f), {"level": -1}

    return agg


def run_cell(cfg: dict) -> dict:
    """Run one experiment cell. Returns a JSON-serializable result dict."""
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n = cfg.get("n_nodes", 10)
    rounds = cfg.get("rounds", 20)
    attack = cfg["attack"]
    byz_frac = cfg["byz_frac"]
    defense = cfg["defense"]
    topo_name = cfg["topology"]
    noniid = cfg.get("noniid", False)

    # --- graph ---
    tv = topo_name == "time_varying"
    graph = topo.build(topo_name, n, seed=seed) if not tv else None
    graph_at = topo.build(topo_name, n, seed=seed) if tv else (lambda t: graph)

    # --- byzantine placement (fixed set, seeded) ---
    n_byz = max(1, int(round(n * byz_frac)))
    byz_ids = set(rng.choice(n, size=n_byz, replace=False).tolist())
    honest_ids = [i for i in range(n) if i not in byz_ids]

    # --- data ---
    clients, (x_test, y_test), (x_bd, y_bd) = make_federated_data(
        n_clients=n,
        samples_per_client=cfg.get("samples_per_client", 150),
        dirichlet_alpha=0.5 if noniid else None,
        seed=seed,
    )
    # data-poisoning happens once, up front, on malicious nodes' data
    local_data = {}
    for i in range(n):
        x, y = clients[i]
        if i in byz_ids and attack in DATA_ATTACKS:
            x, y = atk.corrupt_data(attack, x, y, np.random.default_rng(seed + i))
        local_data[i] = (x, y)

    # --- models (identical init) ---
    models = [MLP() for _ in range(n)]
    init_state = {k: v.clone() for k, v in models[0].state_dict().items()}
    for m in models[1:]:
        m.load_state_dict(init_state)

    # --- defenses (stateful per honest node) ---
    defenses = {}
    for i in honest_ids:
        g0 = graph_at(0)
        d0 = g0.degree(i)
        defenses[i] = _make_defense(
            defense, d0, byz_frac, seed + i, topoguard_kw=cfg.get("topoguard_kw", {})
        )

    lr = cfg.get("lr", 0.05)
    local_epochs = cfg.get("local_epochs", 2)
    eval_every = cfg.get("eval_every", 5)
    hist_acc, hist_asr = [], []

    for t in range(rounds):
        g = graph_at(t)
        # 1. local training -> post-training models (keep pre for attacks)
        pre, post = {}, {}
        for i in range(n):
            x, y = local_data[i]
            from .models import train_local

            pre[i] = flatten_params(models[i])
            train_local(
                models[i], x, y, epochs=local_epochs, lr=lr,
                rng=np.random.default_rng(seed * 7919 + t * 131 + i),
            )
            post[i] = flatten_params(models[i])

        # 2. model-poisoning attacks (white-box: see honest updates)
        honest_deltas = [post[i] - pre[i] for i in honest_ids]
        for i in byz_ids:
            if attack in MODEL_ATTACKS:
                post[i] = atk.corrupt_update(
                    attack, pre[i], post[i], honest_deltas,
                    np.random.default_rng(seed * 104729 + t * 17 + i),
                )

        # 3. aggregate at each honest node
        for i in honest_ids:
            nbrs = {j: post[j] for j in g.neighbors(i)}
            own = post[i]
            d = defenses[i]
            if isinstance(d, TopoGuard):
                agg_w, rep = d.aggregate(own, nbrs)
            else:
                agg_w, rep = d(own, nbrs)
            load_flat_params(models[i], agg_w)
        # byzantine nodes keep their (poisoned) post-training model
        for i in byz_ids:
            load_flat_params(models[i], post[i])

        # 4. evaluate honest nodes (every eval_every rounds + final)
        if (t + 1) % eval_every == 0 or t == rounds - 1:
            accs = [accuracy(models[i], x_test, y_test) for i in honest_ids]
            hist_acc.append(float(np.mean(accs)))
            if attack == "backdoor":
                asrs = [accuracy(models[i], x_bd, y_bd) for i in honest_ids]
                hist_asr.append(float(np.mean(asrs)))

    result = {
        "topology": topo_name,
        "attack": attack,
        "byz_frac": byz_frac,
        "defense": defense,
        "noniid": noniid,
        "seed": seed,
        "n_nodes": n,
        "rounds": rounds,
        "byz_ids": sorted(byz_ids),
        "final_acc": hist_acc[-1],
        "best_acc": max(hist_acc),
        "final_asr": hist_asr[-1] if hist_asr else None,
        "acc_history": [round(a, 4) for a in hist_acc],
        "topoguard_kw": cfg.get("topoguard_kw", {}),
    }
    return result
