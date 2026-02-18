# dim-dca

Research-grade framework for Decline Curve Analysis (DCA) with classical models and exploratory mathematical hypotheses.

## Repository tree

```text
src/dim_dca/
  __init__.py
  cli.py
  compare.py
  data.py
  diagnostics.py
  exploratory.py
  fit.py
  models.py
  objectives.py
  simulate.py
  uncertainty.py
  validation.py
  types.py
tests/
docs/math_notes.md
notebooks/
```

## API

- `fit_model(model, t, q, initial, options)`
- `simulate(model, t, params, noise=...)`
- `residual_diagnostics(y_true, y_pred)`
- `compare_models(models, t, q, initials)`

## Run pipeline

```bash
make install
make run
```

Artifacts are generated in `artifacts/`:
- synthetic dataset CSV
- model comparison JSON
- exploratory hypothesis report
- fit plot

## Scientific scope

Implemented model families:
- Arps exponential, harmonic, hyperbolic
- Stretched exponential
- Duong
- Logistic/Gompertz

Estimation + validation:
- Local nonlinear least squares (`least_squares`)
- Optional global initialization (`differential_evolution`)
- Blocked time-series cross-validation
- AIC/BIC and CV RMSE comparison
- Uncertainty via Hessian covariance, bootstrap, optional MCMC

See `docs/math_notes.md` for derivations and assumptions.
