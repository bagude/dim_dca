from __future__ import annotations

import numpy as np

from dim_dca.compare import compare_models
from dim_dca.fit import FitOptions, fit_model
from dim_dca.simulate import simulate


def test_golden_recovery_hyperbolic() -> None:
    t = np.linspace(0, 36, 180)
    true = {"qi": 1200.0, "di": 0.08, "b": 0.7}
    q = simulate("arps_hyp", t, true, noise="gaussian", sigma=2.0, seed=42)
    fit = fit_model("arps_hyp", t, q, {"qi": 1000.0, "di": 0.1, "b": 0.5}, FitOptions())
    assert fit.success
    assert abs(fit.params["qi"] - true["qi"]) / true["qi"] < 0.1
    assert abs(fit.params["di"] - true["di"]) / true["di"] < 0.2
    assert abs(fit.params["b"] - true["b"]) / true["b"] < 0.25


def test_compare_models_prefers_true_family() -> None:
    t = np.linspace(0, 24, 120)
    q = simulate("arps_exp", t, {"qi": 1000.0, "di": 0.2}, noise="gaussian", sigma=1.0, seed=7)
    initials = {
        "arps_exp": {"qi": 900.0, "di": 0.1},
        "arps_harm": {"qi": 900.0, "di": 0.1},
        "arps_hyp": {"qi": 900.0, "di": 0.1, "b": 0.9},
    }
    rows = compare_models(list(initials.keys()), t, q, initials)
    assert rows[0]["model"] == "arps_exp"
