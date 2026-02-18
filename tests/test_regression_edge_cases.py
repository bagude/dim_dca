from __future__ import annotations

import numpy as np

from dim_dca.fit import FitOptions, fit_model


def test_fit_with_irregular_times_and_zeros() -> None:
    t = np.array([0.0, 0.5, 1.7, 3.2, 3.3, 8.5, 9.1, 12.4])
    q = np.array([1000.0, 950.0, 880.0, 760.0, 0.0, 550.0, 540.0, 400.0])
    mask = q > 0
    fit = fit_model(
        "arps_exp",
        t[mask],
        q[mask],
        {"qi": 900.0, "di": 0.1},
        FitOptions(objective="huber"),
    )
    assert fit.success


def test_missing_values_filtered() -> None:
    t = np.linspace(0, 10, 11)
    q = np.array([1000, 900, np.nan, 760, 700, np.nan, 580, 540, 490, 450, 420], dtype=float)
    m = np.isfinite(q)
    fit = fit_model("arps_harm", t[m], q[m], {"qi": 900.0, "di": 0.05}, FitOptions())
    assert fit.success
