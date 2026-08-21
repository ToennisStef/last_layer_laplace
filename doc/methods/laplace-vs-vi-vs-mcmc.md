# M2/M3 — LLLA vs. VI vs. MCMC on two moons

**Source: measured** (2026-08-19 → 2026-08-20). Same model, same fixed feature map,
same prior `Normal(0, 44.7)` on the last-layer weights `w ∈ R^20`.
Driver: [two_moons_comparison.py](../../two_moons_comparison.py).

| Inference | Guide / kernel | Observed confidence away from the data |
|---|---|---|
| LLLA (paper's code, exact Hessian) | — | lowest (broad uncertainty band) |
| LLLA (Pyro `AutoLaplaceApproximation`) | — | matches the paper's implementation ✔ |
| VI | `AutoNormal` | clearly **more confident** than LLLA |
| VI | `AutoMultivariateNormal` | clearly **more confident** than LLLA |
| MCMC | NUTS, 5 chains × 500 (1000 warmup) | also **more confident** than LLLA, close to VI |

Judged visually from confidence maps (`conf_mcmc.png`, `p_mcmc.png`,
`w_pairplot.png`) — no calibration metric yet ([M4](../lessons-methodology.md)).

## Reasoning chain

1. VI ≠ LLLA → first hypothesis: *the VI approximation is bad* (mean-field
   underestimates variance, a textbook failure mode — which would predict VI being
   **over**confident, consistent with what was seen).
2. `AutoMultivariateNormal` (full-covariance) did not close the gap → the mean-field
   argument alone does not explain it.
3. MCMC as ground truth → MCMC lands with VI, not with LLLA.

## Current hypothesis

If NUTS is the reference, then **LLLA is the outlier and its extra uncertainty is
bloat**, not superior calibration. `Σ = H⁻¹` at the MAP is a local quadratic fit; on
a logistic last layer with a very broad prior the curvature along weakly-determined
directions is tiny, so `H⁻¹` has huge eigenvalues that the true posterior — bounded
by the likelihood's tails and the prior — does not have.

Status: **inferred, not established.** Before accepting it, rule out:

- prior scale not actually identical across the three ([Q3](../open-questions.md#q3));
- MCMC not converged / stuck near `w_map` ([I6](../gotchas/mcmc-setup.md)) — check
  `r_hat`, ESS, and the pair plot for a posterior that is merely narrow because the
  chains never left the mode;
- feature map not identically frozen ([I4](../gotchas/batchnorm-frozen-features.md));
- probit approximation vs. MC-averaged sigmoid: LLLA is scored through
  `sigmoid(m/sqrt(1+πv/8))` while VI/MCMC average sampled sigmoids. These are *not*
  the same estimator — compare LLLA by sampling from its Gaussian too.

## Where this leads

Two candidate conclusions, both untested:

- **last-layer is too little Bayes** for this comparison — go to a full BNN with
  NUTS ([Q2](../open-questions.md#q2));
- **the ReLU-overconfidence framing does not transfer** to a 2-D toy with a bounded
  test grid ([M5](reproduction-scope.md)).
