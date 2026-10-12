"""Byzantine attacks for DFL simulation.

Data-poisoning attacks corrupt the malicious node's local training data
(label_flip, backdoor). Model-poisoning attacks corrupt the update the node
broadcasts (sign_flip, gaussian, alie). In simulation the attacker may
observe the honest updates (standard white-box evaluation); the defense
never sees the attacker set.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import poison_backdoor, poison_labels

ATTACKS = ("none", "label_flip", "sign_flip", "gaussian", "backdoor", "alie")


def corrupt_data(
    attack: str,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    poison_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a data-poisoning attack to a malicious node's local data."""
    if attack == "label_flip":
        return x, poison_labels(y)
    if attack == "backdoor":
        return poison_backdoor(x, y, poison_fraction=poison_fraction, rng=rng)
    return x, y


def corrupt_update(
    attack: str,
    pre: torch.Tensor,
    post: torch.Tensor,
    honest_deltas: list[torch.Tensor],
    rng: np.random.Generator,
    scale: float = 10.0,
    noise_std: float = 1.0,
    alie_z: float = 1.5,
) -> torch.Tensor:
    """Apply a model-poisoning attack, formulated on the UPDATE.

    ``pre``/``post``: the attacker's model before/after local training;
    delta = post - pre is the honest update. ``honest_deltas``: honest
    nodes' updates this round (white-box attacker knowledge, standard).
    Returns the malicious MODEL to broadcast.

    Formulating on the update (not the full model) is load-bearing: flipping
    the full model (-10x w) is self-defeating, because the attacker's own
    training from a poisoned start creates double-negative oscillation and
    the "attack" becomes incoherent (measured: FedAvg survives at 0.92).
    Flipping the update is the standard, devastating formulation.
    """
    delta = post - pre
    if attack == "sign_flip":
        return pre - scale * delta
    if attack == "gaussian":
        # Pure high-variance noise around pre: drowns the honest signal.
        # (Amplifying the honest delta, pre + 10*delta, is NOT adversarial
        # -- it overshoots in the correct direction and FedAvg survives.)
        # Std 10 >> honest update scale (O(1)): the noise dominates.
        noise = torch.randn(delta.shape, dtype=delta.dtype) * 10.0
        return pre + noise
    if attack == "alie":
        # "A Little Is Enough": shift the honest mean update by z*std.
        h = torch.stack(honest_deltas).float()
        mu = h.mean(dim=0)
        sigma = h.std(dim=0).clamp_min(1e-8)
        return pre + mu + alie_z * sigma
    return post
