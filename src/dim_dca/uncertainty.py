from __future__ import annotations

import numpy as np

from .fit import FitOptions, fit_model
from .models import MODEL_SPECS
from .types import Array, FitResult


def fisher_information(jacobian: Array, sigma2: float = 1.0) -> Array:
    return (jacobian.T @ jacobian) / sigma2


def bootstrap_params(
    model: str,
    t: Array,
    q: Array,
    base_params: dict[str, float],
    n_boot: int = 100,
    seed: int = 123,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    fits: list[dict[str, float]] = []
    n = len(t)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tt, qq = t[idx], q[idx]
        order = np.argsort(tt)
        res = fit_model(model, tt[order], qq[order], base_params, FitOptions())
        if res.success:
            fits.append(res.params)
    return fits


def param_ci(samples: list[dict[str, float]], alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    if not samples:
        return {}
    keys = samples[0].keys()
    out: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = np.array([s[k] for s in samples])
        out[k] = (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
    return out


def random_walk_mcmc(
    model: str,
    t: Array,
    q: Array,
    start: dict[str, float],
    sigma: float,
    n_samples: int = 2000,
    step_scale: float = 0.05,
    seed: int = 123,
) -> Array:
    from .models import RATE_FUNCS

    rng = np.random.default_rng(seed)
    spec = MODEL_SPECS[model]
    order = spec.param_order
    lb = np.array(spec.bounds[0])
    ub = np.array(spec.bounds[1])
    fn = RATE_FUNCS[model]

    theta = np.array([start[k] for k in order], dtype=float)

    def logp(th: Array) -> float:
        if np.any(th <= lb) or np.any(th >= ub):
            return -np.inf
        pred = fn(t, {k: float(v) for k, v in zip(order, th, strict=True)})
        return float(-0.5 * np.sum(((q - pred) / sigma) ** 2))

    lp = logp(theta)
    chain = np.zeros((n_samples, len(theta)))
    for i in range(n_samples):
        prop = theta + rng.normal(0.0, step_scale, size=len(theta))
        lpp = logp(prop)
        if np.log(rng.random()) < (lpp - lp):
            theta, lp = prop, lpp
        chain[i] = theta
    return chain
