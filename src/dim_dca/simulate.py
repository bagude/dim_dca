from __future__ import annotations

import numpy as np

from .models import RATE_FUNCS
from .types import Array


def simulate(
    model: str,
    t: Array,
    params: dict[str, float],
    noise: str = "gaussian",
    sigma: float = 0.02,
    seed: int = 123,
) -> Array:
    rng = np.random.default_rng(seed)
    clean = RATE_FUNCS[model](t, params)
    if noise == "gaussian":
        eps = rng.normal(0.0, sigma, size=t.shape)
        return np.maximum(clean + eps, 0.0)
    if noise == "lognormal":
        return clean * rng.lognormal(mean=0.0, sigma=sigma, size=t.shape)
    if noise == "heteroskedastic":
        eps = rng.normal(0.0, sigma * np.maximum(clean, 1e-8), size=t.shape)
        return np.maximum(clean + eps, 0.0)
    raise ValueError(f"Unknown noise model: {noise}")
