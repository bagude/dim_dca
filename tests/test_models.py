from __future__ import annotations

import numpy as np

from dim_dca.models import (
    arps_exponential_rate,
    arps_harmonic_rate,
    arps_hyperbolic_rate,
    dimensionless_rate,
    dimensionless_time,
)


def test_arps_monotone_decreasing() -> None:
    t = np.linspace(0, 100, 400)
    q = arps_hyperbolic_rate(t, {"qi": 1000.0, "di": 0.1, "b": 0.8})
    assert np.all(np.diff(q) <= 1e-10)


def test_hyperbolic_limit_to_exponential() -> None:
    t = np.linspace(0, 20, 300)
    qh = arps_hyperbolic_rate(t, {"qi": 1000.0, "di": 0.1, "b": 1e-4})
    qe = arps_exponential_rate(t, {"qi": 1000.0, "di": 0.1})
    rel = np.mean(np.abs(qh - qe) / np.maximum(qe, 1e-12))
    assert rel < 0.02


def test_hyperbolic_limit_to_harmonic() -> None:
    t = np.linspace(0, 20, 300)
    qh = arps_hyperbolic_rate(t, {"qi": 1000.0, "di": 0.1, "b": 0.9999})
    q1 = arps_harmonic_rate(t, {"qi": 1000.0, "di": 0.1})
    rel = np.mean(np.abs(qh - q1) / np.maximum(q1, 1e-12))
    assert rel < 0.02


def test_dimensionless_invariance_scaling() -> None:
    t = np.linspace(0, 10, 50)
    q = arps_exponential_rate(t, {"qi": 900.0, "di": 0.2})
    tau = dimensionless_time(t, 0.2)
    qd = dimensionless_rate(q, 900.0)

    t2 = t * 2.0
    q2 = arps_exponential_rate(t2, {"qi": 1800.0, "di": 0.1})
    tau2 = dimensionless_time(t2, 0.1)
    qd2 = dimensionless_rate(q2, 1800.0)

    assert np.allclose(tau, tau2)
    assert np.allclose(qd, qd2)
