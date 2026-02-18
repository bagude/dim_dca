from __future__ import annotations

import numpy as np

from dim_dca.exploratory import run_exploratory_suite
from dim_dca.simulate import simulate


def test_exploratory_suite_runs() -> None:
    t = np.linspace(0.1, 30, 120)
    q = simulate("arps_hyp", t, {"qi": 900.0, "di": 0.09, "b": 0.8}, noise="gaussian", sigma=2.0)
    res = run_exploratory_suite(t, q, di=0.09, qi=900.0)
    assert len(res) == 3
    assert all(isinstance(r.metric, float) for r in res)
