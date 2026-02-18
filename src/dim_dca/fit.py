from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

from .models import MODEL_SPECS, RATE_FUNCS
from .objectives import huber_loss, ls_loss
from .types import Array, FitResult


@dataclass
class FitOptions:
    objective: str = "ls"
    global_search: bool = False
    robust_delta: float = 1.0
    max_nfev: int = 20_000


def _pack(params: dict[str, float], order: tuple[str, ...]) -> Array:
    return np.array([params[k] for k in order], dtype=float)


def _unpack(theta: Array, order: tuple[str, ...]) -> dict[str, float]:
    return {k: float(v) for k, v in zip(order, theta, strict=True)}


def fit_model(model: str, t: Array, q: Array, initial: dict[str, float], options: FitOptions | None = None) -> FitResult:
    options = options or FitOptions()
    spec = MODEL_SPECS[model]
    rate_fn = RATE_FUNCS[model]

    theta0 = _pack(initial, spec.param_order)
    lb, ub = np.array(spec.bounds[0]), np.array(spec.bounds[1])

    def pred(theta: Array) -> Array:
        return rate_fn(t, _unpack(theta, spec.param_order))

    def res(theta: Array) -> Array:
        return q - pred(theta)

    if options.global_search:
        bounds = list(zip(lb, ub, strict=True))

        def objective(theta: Array) -> float:
            qp = pred(theta)
            if options.objective == "huber":
                return huber_loss(q, qp, options.robust_delta)
            return ls_loss(q, qp)

        de = differential_evolution(objective, bounds=bounds, polish=False, seed=123)
        theta0 = de.x

    result = least_squares(res, theta0, bounds=(lb, ub), max_nfev=options.max_nfev)
    theta = result.x
    qp = pred(theta)

    if options.objective == "huber":
        loss = huber_loss(q, qp, options.robust_delta)
    else:
        loss = ls_loss(q, qp)

    n = len(q)
    k = len(theta)
    rss = np.sum((q - qp) ** 2)
    sigma2 = max(rss / max(n - k, 1), 1e-12)
    aic = n * np.log(max(rss / n, 1e-12)) + 2 * k
    bic = n * np.log(max(rss / n, 1e-12)) + k * np.log(n)

    cov = None
    if result.jac.size > 0:
        jtj = result.jac.T @ result.jac
        try:
            cov = sigma2 * np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            cov = None

    return FitResult(
        model=model,
        params=_unpack(theta, spec.param_order),
        success=bool(result.success),
        objective=options.objective,
        loss=float(loss),
        aic=float(aic),
        bic=float(bic),
        covariance=cov,
        message=result.message,
        n_obs=n,
        n_params=k,
    )


def fit_bayesian_map(model: str, t: Array, q: Array, initial: dict[str, float], sigma: float = 1.0) -> FitResult:
    spec = MODEL_SPECS[model]
    rate_fn = RATE_FUNCS[model]
    theta0 = _pack(initial, spec.param_order)
    lb, ub = np.array(spec.bounds[0]), np.array(spec.bounds[1])

    def neg_log_post(theta: Array) -> float:
        if np.any(theta < lb) or np.any(theta > ub):
            return np.inf
        pred = rate_fn(t, _unpack(theta, spec.param_order))
        nll = 0.5 * np.sum(((q - pred) / sigma) ** 2)
        return float(nll)

    r = minimize(neg_log_post, theta0, method="L-BFGS-B", bounds=list(zip(lb, ub, strict=True)))
    theta = r.x
    n = len(q)
    k = len(theta)
    return FitResult(
        model=model,
        params=_unpack(theta, spec.param_order),
        success=bool(r.success),
        objective="bayesian_map",
        loss=float(r.fun),
        aic=float(2 * k + 2 * r.fun),
        bic=float(k * np.log(n) + 2 * r.fun),
        covariance=None,
        message=r.message,
        n_obs=n,
        n_params=k,
    )
