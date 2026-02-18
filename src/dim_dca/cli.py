from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .compare import compare_models
from .data import generate_synthetic_dataset, save_dataset_csv
from .exploratory import run_exploratory_suite
from .fit import FitOptions, fit_model
from .models import RATE_FUNCS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end DCA pipeline")
    parser.add_argument("--out", default="artifacts", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t, q, _ = generate_synthetic_dataset()
    save_dataset_csv(out / "synthetic.csv", t, q)

    initials = {
        "arps_exp": {"qi": float(np.max(q)), "di": 0.1},
        "arps_harm": {"qi": float(np.max(q)), "di": 0.1},
        "arps_hyp": {"qi": float(np.max(q)), "di": 0.1, "b": 0.7},
        "stretched_exp": {"qi": float(np.max(q)), "tau": 10.0, "n": 0.8},
        "duong": {"q1": float(np.max(q)), "a": -0.2, "m": 0.5},
        "gompertz": {"qmax": np.trapz(q, t), "alpha": 3.0, "beta": 0.1},
        "logistic": {"qmax": np.trapz(q, t), "k": 0.1, "t0": 10.0},
    }

    rows = compare_models(list(initials.keys()), t, q, initials)
    with (out / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    best = rows[0]["model"]
    fit = fit_model(best, t, q, initials[best], FitOptions(global_search=True))
    qhat = RATE_FUNCS[best](t, fit.params)

    expl = run_exploratory_suite(t, q, di=fit.params.get("di", 0.1), qi=fit.params.get("qi", float(np.max(q))))
    with (out / "exploratory.json").open("w", encoding="utf-8") as f:
        json.dump([e.__dict__ for e in expl], f, indent=2)

    plt.figure(figsize=(8, 4))
    plt.scatter(t, q, s=10, label="Observed")
    plt.plot(t, qhat, color="red", lw=2, label=f"Best fit: {best}")
    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fit.png", dpi=140)


if __name__ == "__main__":
    main()
