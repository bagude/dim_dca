from __future__ import annotations

from dataclasses import asdict

from .fit import FitOptions, fit_model
from .validation import cv_rmse
from .types import Array


def compare_models(models: list[str], t: Array, q: Array, initials: dict[str, dict[str, float]]) -> list[dict]:
    rows = []
    for model in models:
        fit = fit_model(model, t, q, initials[model], FitOptions())
        rmse_cv = cv_rmse(model, t, q, initials[model], n_splits=4)
        row = asdict(fit)
        row["cv_rmse"] = rmse_cv
        rows.append(row)
    rows.sort(key=lambda r: (r["bic"], r["cv_rmse"]))
    return rows
