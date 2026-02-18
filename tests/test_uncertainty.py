from __future__ import annotations

import numpy as np

from dim_dca.fit import FitOptions, fit_model
from dim_dca.simulate import simulate
from dim_dca.uncertainty import bootstrap_params, param_ci, random_walk_mcmc


def test_bootstrap_ci_contains_true_qi() -> None:
    t = np.linspace(0, 20, 100)
    true = {"qi": 800.0, "di": 0.15}
    q = simulate("arps_exp", t, true, noise="gaussian", sigma=3.0, seed=2)
    fit = fit_model("arps_exp", t, q, {"qi": 700.0, "di": 0.1}, FitOptions())
    samples = bootstrap_params("arps_exp", t, q, fit.params, n_boot=40, seed=1)
    ci = param_ci(samples)
    assert ci["qi"][0] <= true["qi"] <= ci["qi"][1]


def test_mcmc_shape_and_finiteness() -> None:
    t = np.linspace(0, 8, 40)
    q = simulate("arps_exp", t, {"qi": 500.0, "di": 0.2}, seed=12)
    chain = random_walk_mcmc(
        "arps_exp", t, q, {"qi": 450.0, "di": 0.18}, sigma=10.0, n_samples=300, step_scale=0.01
    )
    assert chain.shape == (300, 2)
    assert np.isfinite(chain).all()
