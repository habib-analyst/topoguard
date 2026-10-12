"""Minimal TopoGuard example: 6 nodes on a ring, 1 Byzantine (sign-flip).

Run:  .venv/bin/python examples/quickstart.py
"""

import numpy as np
import torch

from topoguard import TopoGuard
from topoguard import topology as topo
from topoguard.data import make_federated_data
from topoguard.models import MLP, accuracy, flatten_params, load_flat_params, train_local

N, ROUNDS, BYZ = 6, 8, {4}

clients, (x_test, y_test), _ = make_federated_data(
    n_clients=N, samples_per_client=120, dirichlet_alpha=None, seed=7
)
models = [MLP() for _ in range(N)]
state = {k: v.clone() for k, v in models[0].state_dict().items()}
for m in models[1:]:
    m.load_state_dict(state)

graph = topo.ring(N)
aggs = {i: TopoGuard() for i in range(N) if i not in BYZ}
honest = [i for i in range(N) if i not in BYZ]

for t in range(ROUNDS):
    post = {}
    for i in range(N):
        x, y = clients[i]
        train_local(models[i], x, y, epochs=2, lr=0.05,
                    rng=np.random.default_rng(1000 + t * N + i))
        post[i] = flatten_params(models[i])
    # Byzantine node 4: sign-flip its update
    post[4] = -10.0 * post[4]
    for i in honest:
        nbrs = {j: post[j] for j in graph.neighbors(i)}
        new_w, rep = aggs[i].aggregate(post[i], nbrs)
        load_flat_params(models[i], new_w)
    acc = float(np.mean([accuracy(models[i], x_test, y_test) for i in honest]))
    print(f"round {t+1}: honest acc = {acc:.4f}")

print("done — honest nodes learned despite the sign-flip attacker.")
