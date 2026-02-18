from __future__ import annotations

import numpy as np

from .types import Array


def residuals(y_true: Array, y_pred: Array, weights: Array | None = None) -> Array:
    r = y_true - y_pred
    if weights is None:
        return r
    return np.sqrt(np.maximum(weights, 1e-12)) * r


def ls_loss(y_true: Array, y_pred: Array, weights: Array | None = None) -> float:
    r = residuals(y_true, y_pred, weights)
    return 0.5 * float(np.dot(r, r))


def huber_loss(y_true: Array, y_pred: Array, delta: float = 1.0) -> float:
    r = y_true - y_pred
    abs_r = np.abs(r)
    quad = np.minimum(abs_r, delta)
    lin = abs_r - quad
    return float(np.sum(0.5 * quad**2 + delta * lin))


def gaussian_nll(y_true: Array, y_pred: Array, sigma: float) -> float:
    r = y_true - y_pred
    n = y_true.size
    return float(0.5 * np.sum((r / sigma) ** 2) + n * np.log(sigma) + 0.5 * n * np.log(2.0 * np.pi))
