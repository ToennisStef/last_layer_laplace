# M2/M3 — LLLA vs. VI vs. MCMC on two moons

**Source: measured.** Rewritten 2026-08-21 after
[two_moons_comparison.py](../../two_moons_comparison.py) replaced the notebook
run. Same model, same frozen feature map, same prior `Normal(0, 44.72)` on the
last-layer weights `w` in R^20. Artifacts in `artifacts/`.

> **This note previously concluded the opposite.** The earlier version had MCMC
> siding with VI and inferred that Laplace was "bloated". That run seeded NUTS at
> `w_map`; with `init_to_sample()` and converged chains the ordering changes.
> Kept visible on purpose — the retraction is the finding. See
> [learning-log](../learning-log.md).

## Result (run 2: 5 chains x 500 draws, `r_hat` <= 1.0036, `n_eff` 1270-2122, 0 divergences)

| Inference | `\|\|E[w]\|\|` | mean sd | `\|loc\|/sd` | mean conf | mean epistemic (nats) |
|---|---|---|---|---|---|
| Laplace (`AutoLaplaceApproximation`) | 17.1 | 27.7 | 0.13 | **0.616** | **0.637** |
| VI mean-field (`AutoNormal`) | 57.0 | 1.8 | 4.47 | 0.919 | 0.175 |
| VI full-rank (`AutoMultivariateNormal`) | 56.0 | 7.8 | 0.90 | 0.910 | 0.202 |
| **NUTS (reference)** | 125.2 | 34.7 | 0.70 | **0.845** | **0.328** |

Max possible epistemic is `ln 2 = 0.693`. Prior scale 44.72 for comparison.

**NUTS sits between Laplace and VI.** Laplace is the *least* confident, VI the
most. Laplace and NUTS reproduce to three decimals across two independent runs
(4 vs 5 chains); the VI numbers move (mean-field sd 3.03 -> 1.78 once the VI
optimizer got LR decay) but their *ordering* does not.

All four are scored with the identical sampled predictive
([decision 0009](../decisions/0009-uniform-sampled-predictive.md)), so the
differences are posterior differences, not estimator differences.

## Why Laplace ends up least confident

Two effects compound, and neither is "the approximation is bad":

1. **Prior-dominated spread.** `Sigma = H^-1` at the mode. The network separates
   the data, so `sigma(1-sigma) -> 0` and the likelihood's Hessian `Phi^T S Phi`
   collapses:

   ```
   Hessian eigenvalues: min 5.049e-04   max 1.401e+00
   prior alone contributes 1/var0 = 5.000e-04 to every eigenvalue
   -> 8 of 20 directions are within 2x of pure prior curvature
   ```

   In the flattest direction the likelihood supplies ~1% of the curvature.
   Laplace's mean sd of 27.7 is **62% of the prior scale** — it largely hands the
   prior back.

2. **The mode is deep inside the typical set** — `\|\|w_map\|\| = 17.1` against a
   sampled shell at `\|\|w\|\| ~ 198`. Full account in
   [M6](mode-vs-typical-set.md).

Confidence tracks `m(x)/sqrt(1 + pi*v/8)` — the **ratio**. Laplace has a small
centre and a prior-sized spread (`\|loc\|/sd = 0.13`), so averaged sigmoids collapse
toward 0.5. NUTS has a *larger* spread yet is more confident, because its centre is
7x further out. **Ranking methods by covariance magnitude alone will mislead you.**

## Why full-rank VI does not recover Laplace

It lands on mean-field instead. The mean-field independence assumption is
therefore not what separates VI from Laplace. The two optimise different things:

- Laplace takes local curvature at the mode and minimises no divergence at all.
- VI minimises `KL(q||p)` over the whole family; a `q` as wide as the prior would
  make `E_q[log p(y|w)]` terrible, so the ELBO shrinks it — and shrinks it further
  the longer it converges (LR decay tightened both guides without moving
  confidence).

The textbook expectation that a full-covariance Gaussian recovers Laplace holds
only when the posterior is Gaussian *in shape near the mode and dominated by it*.
Here it is a shell.

## Current reading

- **Laplace is underconfident here**, but "bloated covariance" is the wrong
  diagnosis: the covariance is roughly the prior, and the centre is unrepresentative.
- **VI is overconfident**, and converging it further makes that worse, not better.
- **NUTS is the only one describing the actual posterior**, and it is well converged
  by every available diagnostic.
- None of this says Laplace is a bad method in general. It says that on a
  **separable** problem with a **weakly-informative** prior the Laplace assumptions
  fail in a specific, predictable way.

## Still open

- No calibration metric has been computed — everything here is confidence and
  entropy on a grid, not ECE/NLL/OOD ([Q3](../open-questions.md#q3), [M4](../lessons-methodology.md)).
- The prediction that a tighter prior closes the gap is untested ([Q8](../open-questions.md#q8)).
- Whether any of this transfers beyond last-layer Bayes ([Q2](../open-questions.md#q2)).
