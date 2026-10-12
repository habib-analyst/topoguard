"""Synthetic federated data for DFL experiments.

Binary classification on 20-D Gaussian blobs (fast, self-contained, no
downloads). Non-IID via Dirichlet label skew. Backdoor trigger: a fixed
outlier pattern on the last 3 features, relabeled to the target class.
"""

from __future__ import annotations

import numpy as np

N_FEATURES = 20
TRIGGER_FEATURES = (17, 18, 19)
TRIGGER_VALUE = 5.0


def _blobs(n: int, rng: np.random.Generator, sep: float = 2.0):
    x = rng.normal(0, 1, (n, N_FEATURES))
    y = rng.integers(0, 2, n)
    # shift class means apart along the first 5 dims
    x[y == 0, :5] -= sep / 2
    x[y == 1, :5] += sep / 2
    return x, y


def apply_trigger(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[:, TRIGGER_FEATURES] = TRIGGER_VALUE
    return x


def make_federated_data(
    n_clients: int,
    samples_per_client: int,
    dirichlet_alpha: float | None,
    test_size: int = 2000,
    backdoor_target: int = 0,
    seed: int = 0,
):
    """Return (clients, (x_test, y_test), (x_bd, y_bd)).

    clients: list of (x, y) per client. If dirichlet_alpha is None the data
    is IID; otherwise labels are skewed per client with that concentration.
    (x_bd, y_bd): triggered test set for backdoor ASR measurement.
    """
    rng = np.random.default_rng(seed)
    total = n_clients * samples_per_client
    x_all, y_all = _blobs(total, rng)

    clients: list[tuple[np.ndarray, np.ndarray]] = []
    if dirichlet_alpha is None:
        idx = rng.permutation(total).reshape(n_clients, samples_per_client)
        for k in range(n_clients):
            clients.append((x_all[idx[k]], y_all[idx[k]]))
    else:
        # Dirichlet label skew: per-class Dirichlet over clients
        per_client: list[list[int]] = [[] for _ in range(n_clients)]
        for c in (0, 1):
            c_idx = np.where(y_all == c)[0]
            rng.shuffle(c_idx)
            props = rng.dirichlet([dirichlet_alpha] * n_clients)
            cuts = (np.cumsum(props) * len(c_idx)).astype(int)
            start = 0
            for k in range(n_clients):
                end = cuts[k] if k < n_clients - 1 else len(c_idx)
                per_client[k].extend(c_idx[start:end].tolist())
                start = end
        for k in range(n_clients):
            ii = np.array(per_client[k])
            # guard against empty clients
            if len(ii) == 0:
                ii = rng.choice(total, size=samples_per_client, replace=False)
            clients.append((x_all[ii], y_all[ii]))

    x_test, y_test = _blobs(test_size, np.random.default_rng(seed + 999))
    x_bd = apply_trigger(x_test.copy())
    y_bd = np.full(test_size, backdoor_target)
    return clients, (x_test, y_test), (x_bd, y_bd)


def poison_labels(y: np.ndarray) -> np.ndarray:
    """Label-flip attack: 0 <-> 1."""
    return (1 - y).astype(y.dtype)


def poison_backdoor(
    x: np.ndarray,
    y: np.ndarray,
    poison_fraction: float = 0.3,
    target: int = 0,
    rng: np.random.Generator | None = None,
):
    """Plant the trigger on a fraction of samples, relabel to target."""
    rng = np.random.default_rng() if rng is None else rng
    n_poison = int(len(x) * poison_fraction)
    idx = rng.choice(len(x), size=n_poison, replace=False)
    x = x.copy()
    x[idx] = apply_trigger(x[idx])
    y = y.copy()
    y[idx] = target
    return x, y
