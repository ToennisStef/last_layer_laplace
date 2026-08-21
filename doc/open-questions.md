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

## Q4 — Same predictive estimator for all three methods
*Status: not done.*
LLLA is currently scored with the probit approximation, VI/MCMC with sampled
sigmoids. Re-score LLLA by drawing from its Gaussian posterior so all three use the
identical predictive. Cheap; should be done **before** trusting
[M2](methods/laplace-vs-vi-vs-mcmc.md).

## Q5 — MCMC convergence audit
*Status: `mcmc.summary()` printed, not systematically checked.*
Record `r_hat`, ESS, divergences; check the pair plot for chains that never left
`w_map`. If MCMC is not converged, the whole ordering in
[M2](methods/laplace-vs-vi-vs-mcmc.md) is unsafe. See [I6](gotchas/mcmc-setup.md).

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
