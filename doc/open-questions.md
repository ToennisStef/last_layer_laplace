# Open questions & untried ideas

Everything here is **not yet tried**. Nothing on this page is evidence.
Move an item into [lessons-implementation](lessons-implementation.md) or
[lessons-methodology](lessons-methodology.md) once it has been run.

## Q1 — Prior-scale hyperparameter tuning (Ritter et al. 2018)
*Status: paper not read, not implemented.*
Kristiadi et al. cite **"A Scalable Laplace Approximation for Neural Networks"**
(Ritter, Botev, Barber, ICLR 2018) for tuning the prior precision instead of
guessing it — the marginal-likelihood / validation-NLL route. Unknown whether it
actually fixes the calibration, and it may only re-fit an already-misspecified
Gaussian.
→ Read the paper, implement the tuning, re-run the comparison at the tuned scale.
Directly targets [M1](methods/prior-scale-calibration.md) and would test
[M3](methods/laplace-vs-vi-vs-mcmc.md#current-hypothesis).
Ref: <https://openreview.net/forum?id=Skdvd2xAZ> ·
`laplace-torch` implements `optimize_prior_precision`: <https://aleximmer.github.io/Laplace/>

## Q2 — Full BNN instead of last-layer only
*Status: not attempted.*
Run NUTS over **all** weights (not just `w`) and compare against full Laplace. Tests
whether the LLLA/MCMC gap is a property of the Laplace approximation or of the
last-layer restriction. Expensive; `h = 20`, 2 hidden layers is probably still
feasible on two moons.
→ Would settle [M3](methods/laplace-vs-vi-vs-mcmc.md#current-hypothesis).

## Q3 — Actual calibration metrics
*Status: not computed. Currently everything is judged by eye.*
Add held-out NLL, ECE / reliability diagrams, and an OOD confidence measure
(AUROC vs. a far-field or second-dataset control). Without these,
[M2](methods/laplace-vs-vi-vs-mcmc.md) and [M4](lessons-methodology.md) stay
qualitative and "over/underconfident" is an impression, not a result.

## Q4 — Same predictive estimator for all methods
*Status: **DONE** (2026-08-21) — see [decision 0009](decisions/0009-uniform-sampled-predictive.md).*
`two_moons_comparison.py` now scores all four methods by sampling weights and
averaging sigmoids, so the maps differ only through the posterior. The paper's
`Model.predict` keeps the probit form for reproducing the original figure.

## Q5 — MCMC convergence audit
*Status: **DONE** (2026-08-21).*
NUTS is seeded with `init_to_sample()` (not the MAP), 5 chains x 500 draws:
`r_hat` <= 1.0036, `n_eff` 1270-2122 of 2500, **0 divergences**, and the numbers
reproduce across an independent 4-chain run. The chains are converged, so
[M2](methods/laplace-vs-vi-vs-mcmc.md) can be trusted — and the *old* M3 conclusion,
which came from `w_map`-seeded chains, is retracted. Diagnostics are saved to
`artifacts/mcmc_diagnostics.json` on every run.

## Q6 — `laplace-torch` as a cross-check
*Status: not used.*
The repo README points to <https://github.com/AlexImmer/Laplace>. Running its LLLA on
the same trained net would separate "Laplace is bloated" from "our Laplace has a bug".

## Q7 — Is the ReLU framing the right one at all?
*Status: open thought.*
The asymptotic overconfidence result may simply not be what a bounded 2-D grid tests
— see [M5](methods/reproduction-scope.md). Possible alternative framings: GP-limit
comparison, deep ensembles as a second reference, or an explicit far-field
(`‖x‖ = 10, 100, 1000`) confidence sweep, which is what the theorem is actually about.

## Q8 — Does a tighter prior close the Laplace/NUTS gap?
*Status: predicted, not run.*
[M6](methods/mode-vs-typical-set.md) says the typical set sits at radius
`prior_scale * sqrt(d)` = 44.72 * 4.47 ~ 200 while the mode is at 17. Shrinking
`prior_scale` should pull the shell toward the mode and shrink the Laplace-vs-NUTS
disagreement; it should also raise the likelihood's share of the Hessian.
→ Sweep `Config.var0` over a few decades and plot mean confidence per method.
Cheap, and it is a real prediction that could fail — which makes it the most
informative next run. Connects [M1](methods/prior-scale-calibration.md),
[M6](methods/mode-vs-typical-set.md), and [Q1](#q1).
