from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from dim_dca.models import arps_exponential_rate


@settings(max_examples=40)
@given(
    qi=st.floats(min_value=1.0, max_value=5000.0),
    di=st.floats(min_value=1e-4, max_value=1.0),
)
def test_exponential_nonnegative_and_decreasing(qi: float, di: float) -> None:
    t = np.linspace(0, 100, 300)
    q = arps_exponential_rate(t, {"qi": qi, "di": di})
    assert np.all(q >= 0.0)
    assert np.all(np.diff(q) <= 1e-10)
