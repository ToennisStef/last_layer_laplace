# Methodological lessons

What the methods actually deliver, as opposed to how to code them
([lessons-implementation.md](lessons-implementation.md)).
`Source:` — **measured**, **read**, **inferred**, **assumed**.

| # | Lesson | Status | Detail |
|---|---|---|---|
| M1 | Prior scale is *the* calibration knob for LLLA: too small → overconfident, too large → underconfident. Nothing about the inference machinery moves the confidence map as much | measured | [prior-scale-calibration](methods/prior-scale-calibration.md) |
| M2 | On two moons, **NUTS sits between Laplace and VI**: Laplace least confident (0.616), NUTS 0.845, VI ~0.91. Reproduces across runs; NUTS well converged (`r_hat` <= 1.004, 0 divergences) | measured | [laplace-vs-vi-vs-mcmc](methods/laplace-vs-vi-vs-mcmc.md) |
| M3 | ~~LLLA's uncertainty is bloat~~ **RETRACTED 2026-08-21.** That came from chains seeded at `w_map`. Laplace is underconfident, but because its covariance is ~the prior *and* its centre is unrepresentative — not because the covariance is inflated | measured | [laplace-vs-vi-vs-mcmc](methods/laplace-vs-vi-vs-mcmc.md#why-laplace-ends-up-least-confident) |
| M6 | **The mode is not where the mass is.** `\|\|w_map\|\| = 17` vs a sampled shell at `\|\|w\|\| ~ 198 ~ prior s*sqrt(d)`. Textbook high-dimensional geometry (median draw 9.72 nats above the mode vs a Gaussian's 10.0), not a pathology | measured | [mode-vs-typical-set](methods/mode-vs-typical-set.md) |
| M4 | "Well-calibrated" was judged by eye from confidence maps. No proper metric (ECE, NLL, OOD AUROC) has been computed — so M2/M3 are qualitative | honest gap | [open-questions](open-questions.md#q3) |
| M5 | Reproducing the paper's *figures* is not the same as reproducing its *claim*. The ReLU-overconfidence result is about behaviour far from the data; a matching picture on two moons is weak evidence | inferred | [reproduction-scope](methods/reproduction-scope.md) |

Untried directions that would settle M3 are in
[open-questions.md](open-questions.md).
