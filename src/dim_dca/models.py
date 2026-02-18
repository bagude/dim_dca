from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Array

_EPS = 1e-12


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_order: tuple[str, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]]


def arps_hyperbolic_rate(t: Array, p: dict[str, float]) -> Array:
    qi, di, b = p["qi"], p["di"], p["b"]
    x = 1.0 + b * di * t
    return qi / np.power(np.maximum(x, _EPS), 1.0 / np.maximum(b, _EPS))


def arps_hyperbolic_cum(t: Array, p: dict[str, float]) -> Array:
    qi, di, b = p["qi"], p["di"], p["b"]
    if abs(b - 1.0) < 1e-7:
        return (qi / di) * np.log1p(di * t)
    one_minus_b = 1.0 - b
    return (qi / ((1.0 - b) * di)) * (1.0 - np.power(1.0 + b * di * t, -one_minus_b / b))


def arps_exponential_rate(t: Array, p: dict[str, float]) -> Array:
    qi, di = p["qi"], p["di"]
    return qi * np.exp(-di * t)


def arps_exponential_cum(t: Array, p: dict[str, float]) -> Array:
    qi, di = p["qi"], p["di"]
    return (qi / di) * (1.0 - np.exp(-di * t))


def arps_harmonic_rate(t: Array, p: dict[str, float]) -> Array:
    qi, di = p["qi"], p["di"]
    return qi / np.maximum(1.0 + di * t, _EPS)


def arps_harmonic_cum(t: Array, p: dict[str, float]) -> Array:
    qi, di = p["qi"], p["di"]
    return (qi / di) * np.log1p(di * t)


def stretched_exponential_rate(t: Array, p: dict[str, float]) -> Array:
    qi, tau, n = p["qi"], p["tau"], p["n"]
    x = np.maximum(t / np.maximum(tau, _EPS), 0.0)
    return qi * np.exp(-np.power(x, n))


def stretched_exponential_cum(t: Array, p: dict[str, float]) -> Array:
    qi, tau, n = p["qi"], p["tau"], p["n"]
    # numerical quadrature via trapz on dense local grid for stability
    tt = np.linspace(0.0, float(np.max(t)), 400)
    qt = stretched_exponential_rate(tt, p)
    cum = np.cumsum((qt[1:] + qt[:-1]) * np.diff(tt) * 0.5)
    cum = np.insert(cum, 0, 0.0)
    return np.interp(t, tt, cum)


def duong_rate(t: Array, p: dict[str, float]) -> Array:
    q1, a, m = p["q1"], p["a"], p["m"]
    tp1 = np.maximum(t + 1.0, _EPS)
    return q1 * np.power(tp1, -m) * np.exp((a / (1.0 - m)) * (np.power(tp1, 1.0 - m) - 1.0))


def duong_cum(t: Array, p: dict[str, float]) -> Array:
    tt = np.linspace(0.0, float(np.max(t)), 500)
    qt = duong_rate(tt, p)
    cum = np.cumsum((qt[1:] + qt[:-1]) * np.diff(tt) * 0.5)
    cum = np.insert(cum, 0, 0.0)
    return np.interp(t, tt, cum)


def gompertz_rate(t: Array, p: dict[str, float]) -> Array:
    qmax, alpha, beta = p["qmax"], p["alpha"], p["beta"]
    cum = qmax * np.exp(-alpha * np.exp(-beta * t))
    return np.gradient(cum, t, edge_order=2)


def gompertz_cum(t: Array, p: dict[str, float]) -> Array:
    qmax, alpha, beta = p["qmax"], p["alpha"], p["beta"]
    return qmax * np.exp(-alpha * np.exp(-beta * t))


def logistic_rate(t: Array, p: dict[str, float]) -> Array:
    qmax, k, t0 = p["qmax"], p["k"], p["t0"]
    e = np.exp(-k * (t - t0))
    return (qmax * k * e) / np.power(1.0 + e, 2)


def logistic_cum(t: Array, p: dict[str, float]) -> Array:
    qmax, k, t0 = p["qmax"], p["k"], p["t0"]
    return qmax / (1.0 + np.exp(-k * (t - t0)))


def dimensionless_time(t: Array, di: float) -> Array:
    return di * t


def dimensionless_rate(q: Array, qi: float) -> Array:
    return q / np.maximum(qi, _EPS)


MODEL_SPECS: dict[str, ModelSpec] = {
    "arps_exp": ModelSpec("arps_exp", ("qi", "di"), ((1e-8, 1e-8), (1e6, 10.0))),
    "arps_harm": ModelSpec("arps_harm", ("qi", "di"), ((1e-8, 1e-8), (1e6, 10.0))),
    "arps_hyp": ModelSpec("arps_hyp", ("qi", "di", "b"), ((1e-8, 1e-8, 1e-6), (1e6, 10.0, 1.999))),
    "stretched_exp": ModelSpec("stretched_exp", ("qi", "tau", "n"), ((1e-8, 1e-8, 0.05), (1e6, 1e4, 2.0))),
    "duong": ModelSpec("duong", ("q1", "a", "m"), ((1e-8, -5.0, 0.01), (1e6, 5.0, 0.999))),
    "gompertz": ModelSpec("gompertz", ("qmax", "alpha", "beta"), ((1e-8, 1e-6, 1e-6), (1e9, 30.0, 5.0))),
    "logistic": ModelSpec("logistic", ("qmax", "k", "t0"), ((1e-8, 1e-6, -1e3), (1e9, 5.0, 1e3))),
}

RATE_FUNCS = {
    "arps_exp": arps_exponential_rate,
    "arps_harm": arps_harmonic_rate,
    "arps_hyp": arps_hyperbolic_rate,
    "stretched_exp": stretched_exponential_rate,
    "duong": duong_rate,
    "gompertz": gompertz_rate,
    "logistic": logistic_rate,
}

CUM_FUNCS = {
    "arps_exp": arps_exponential_cum,
    "arps_harm": arps_harmonic_cum,
    "arps_hyp": arps_hyperbolic_cum,
    "stretched_exp": stretched_exponential_cum,
    "duong": duong_cum,
    "gompertz": gompertz_cum,
    "logistic": logistic_cum,
}
