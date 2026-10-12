"""Small MLP (CPU) and flat-parameter helpers.

Model math is intentionally tiny: the contribution of this project is the
aggregation algorithm, not the model. Update vectors are 1-D tensors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer perceptron, binary logit output."""

    def __init__(self, n_features: int = 20, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).float()


def load_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset : offset + n].reshape(p.shape))
            offset += n


def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        preds = (model(torch.as_tensor(x, dtype=torch.float32)) >= 0).long().numpy()
    return float((preds == y).mean())


def train_local(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int = 32,
    rng: np.random.Generator | None = None,
) -> None:
    """In-place SGD on the client's data."""
    rng = np.random.default_rng() if rng is None else rng
    model.train()
    # foreach=False: torch's foreach dispatch has ~10x overhead on this CPU
    # build for tiny models (profiled 2026-10-11).
    opt = torch.optim.SGD(model.parameters(), lr=lr, foreach=False)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    n = len(x)
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(n))
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
