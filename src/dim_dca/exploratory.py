from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .diagnostics import loglog_curvature
from .types import Array


@dataclass
class HypothesisResult:
    name: str
    hypothesis: str
    metric: float
    passed: bool
    falsification_rule: str


def test_scaling_invariant(t: Array, q: Array, di: float, qi: float) -> HypothesisResult:
    """H1: Dimensionless transform collapses trajectories across scales."""
    tau = di * t
    qd = q / qi
    corr = float(np.corrcoef(np.log1p(tau), np.log(np.maximum(qd, 1e-12)))[0, 1])
    return HypothesisResult(
        name="H1_scaling",
        hypothesis="Dimensionless coordinates preserve decline shape under multiplicative scaling.",
        metric=abs(corr),
        passed=abs(corr) > 0.85,
        falsification_rule="Fail if |corr(log(1+tau), log(qd))| <= 0.85",
    )


def test_curvature_changepoints(t: Array, q: Array) -> HypothesisResult:
    """H2: Regime shifts appear as peaks in log-log curvature."""
    curv = np.abs(loglog_curvature(t, q))
    peaks, _ = signal.find_peaks(curv, height=np.quantile(curv, 0.8))
    n_peaks = int(len(peaks))
    return HypothesisResult(
        name="H2_regime",
        hypothesis="Changepoints correspond to strong curvature peaks in log-log space.",
        metric=float(n_peaks),
        passed=n_peaks >= 1,
        falsification_rule="Fail if no peaks above 80th percentile curvature.",
    )


def symbolic_surrogate_exponents(t: Array, q: Array) -> HypothesisResult:
    """H3: Power-law basis can approximate decline in log-space."""
    x = np.log(np.maximum(t + 1.0, 1e-12))
    y = np.log(np.maximum(q, 1e-12))
    a = np.vstack([np.ones_like(x), x, x**2]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    yhat = a @ coef
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return HypothesisResult(
        name="H3_symbolic",
        hypothesis="Quadratic polynomial in log-time is a constrained symbolic surrogate.",
        metric=float(r2),
        passed=float(r2) > 0.9,
        falsification_rule="Fail if surrogate R^2 <= 0.9 in log space.",
    )


def run_exploratory_suite(t: Array, q: Array, di: float, qi: float) -> list[HypothesisResult]:
    return [
        test_scaling_invariant(t, q, di=di, qi=qi),
        test_curvature_changepoints(t, q),
        symbolic_surrogate_exponents(t, q),
    ]
