from __future__ import annotations

import numpy as np

from .fit import FitOptions, fit_model
from .models import RATE_FUNCS
from .types import Array


def blocked_time_series_splits(n: int, n_splits: int = 4, min_train: float = 0.5) -> list[tuple[np.ndarray, np.ndarray]]:
    start = int(n * min_train)
    test_size = max((n - start) // n_splits, 1)
    splits = []
    for i in range(n_splits):
        tr_end = start + i * test_size
        te_end = min(tr_end + test_size, n)
        if te_end <= tr_end:
            continue
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
    return splits


def cv_rmse(model: str, t: Array, q: Array, initial: dict[str, float], n_splits: int = 4) -> float:
    errors = []
    for tr, te in blocked_time_series_splits(len(t), n_splits=n_splits):
        fit = fit_model(model, t[tr], q[tr], initial, FitOptions())
        qp = RATE_FUNCS[model](t[te], fit.params)
        errors.append(float(np.sqrt(np.mean((q[te] - qp) ** 2))))
    return float(np.mean(errors))
