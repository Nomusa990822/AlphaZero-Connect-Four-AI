from __future__ import annotations

import numpy as np


def add_dirichlet_noise(
    priors: dict[int, float],
    alpha: float = 0.3,
    epsilon: float = 0.25,
    seed: int | None = None,
) -> dict[int, float]:
    """
    Mix Dirichlet noise into root priors for exploration.

    Args:
        priors: move -> prior probability
        alpha: Dirichlet concentration
        epsilon: interpolation factor
        seed: optional RNG seed

    Returns:
        New move -> noisy prior dict
    """
    if not priors:
        return {}

    rng = np.random.default_rng(seed)

    moves = list(priors.keys())
    base = np.array([priors[m] for m in moves], dtype=np.float32)

    noise = rng.dirichlet([alpha] * len(moves)).astype(np.float32)
    mixed = (1.0 - epsilon) * base + epsilon * noise

    total = mixed.sum()
    if total > 0:
        mixed = mixed / total

    return {move: float(prob) for move, prob in zip(moves, mixed)}
