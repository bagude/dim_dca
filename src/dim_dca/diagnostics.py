from __future__ import annotations

import numpy as np

from .types import Array


def residual_diagnostics(y_true: Array, y_pred: Array) -> dict[str, float]:
    r = y_true - y_pred
    mse = float(np.mean(r**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(r)))
    mape = float(np.mean(np.abs(r / np.maximum(y_true, 1e-12))))
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


def loglog_curvature(t: Array, q: Array) -> Array:
    lt = np.log(np.maximum(t, 1e-8))
    lq = np.log(np.maximum(q, 1e-12))
    d1 = np.gradient(lq, lt)
    d2 = np.gradient(d1, lt)
    return d2
