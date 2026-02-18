from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass
class FitResult:
    model: str
    params: dict[str, float]
    success: bool
    objective: str
    loss: float
    aic: float
    bic: float
    covariance: Array | None
    message: str
    n_obs: int
    n_params: int


ModelCallable = Callable[[Array, dict[str, float]], Array]
