# Mathematical notes for dim-dca

## 1. Arps family
Let rate be `q(t)` and initial rate `q_i`.

### Exponential
Assumption: constant nominal decline `D(t)=D_i`.

\[
\frac{dq}{dt} = -D_i q \Rightarrow q(t)=q_i e^{-D_i t}.
\]

Cumulative production:
\[
N_p(t)=\int_0^t q(\tau)\,d\tau = \frac{q_i}{D_i}(1-e^{-D_i t}).
\]

### Harmonic
Assumption: decline exponent `b=1` in Arps hyperbolic form.

\[
q(t)=\frac{q_i}{1+D_i t}, \quad N_p(t)=\frac{q_i}{D_i}\ln(1+D_i t).
\]

### Hyperbolic
\[
q(t)=\frac{q_i}{(1+bD_i t)^{1/b}},\quad b>0.
\]

Cumulative for `b\neq1`:
\[
N_p(t)=\frac{q_i}{(1-b)D_i}\left[1-(1+bD_i t)^{-(1-b)/b}\right].
\]

Limiting cases:
- `b\to 0` recovers exponential via `\log(1+bD_it)/b\to D_it`.
- `b\to 1` recovers harmonic by continuity.

## 2. Dimensionless form and invariance
Define `\tau=D_i t`, `\tilde q=q/q_i`. Hyperbolic becomes
\[
\tilde q(\tau)= (1+b\tau)^{-1/b}.
\]
This removes dimensions and exhibits scale invariance w.r.t. multiplicative changes in `q_i` and reciprocal time scaling by `D_i`.

## 3. Identifiability and degeneracy
- Short windows (`t\ll 1/D_i`) create parameter correlation between `q_i` and `D_i`.
- In hyperbolic, `b` becomes weakly identified when data do not include late-time behavior.
- `b\to 0` causes numerical ill-conditioning if naively evaluated; stabilized via exponential limit.

## 4. Objective functions
- LS: `\min_\theta \sum_i (q_i-\hat q_i)^2`.
- WLS: weighted by inverse variance if heteroskedasticity is known.
- Robust Huber: quadratic near zero, linear in tails.
- Gaussian likelihood equivalent to LS under iid Gaussian noise.

## 5. Uncertainty
- Local asymptotic covariance: `\Sigma\approx\sigma^2 (J^T J)^{-1}`.
- Bootstrap: refit on resampled series for non-linear/non-Gaussian uncertainty.
- Bayesian option: MAP and random-walk MCMC posterior samples.

## 6. Exploratory hypotheses (explicitly exploratory)
1. **Scaling-invariant collapse:** dimensionless transform improves cross-well comparability.
2. **Curvature regime detector:** change-points correspond to log-log curvature peaks.
3. **Symbolic surrogate:** constrained polynomial in log-time can approximate decline and suggest alternative forms.

Each hypothesis includes falsification criteria in code and tests.
