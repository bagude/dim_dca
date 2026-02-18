from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .simulate import simulate
from .types import Array


def synthetic_time_grid(n: int = 180, t_max: float = 36.0) -> Array:
    return np.linspace(0.0, t_max, n)


def generate_synthetic_dataset(seed: int = 123) -> tuple[Array, Array, dict[str, float]]:
    t = synthetic_time_grid()
    true = {"qi": 1250.0, "di": 0.08, "b": 0.75}
    q = simulate("arps_hyp", t, true, noise="heteroskedastic", sigma=0.03, seed=seed)
    return t, q, true


def save_dataset_csv(path: str | Path, t: Array, q: Array) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "rate"])
        for ti, qi in zip(t, q, strict=True):
            w.writerow([float(ti), float(qi)])
