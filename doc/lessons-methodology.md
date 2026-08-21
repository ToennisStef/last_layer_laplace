# Methodological lessons

What the methods actually deliver, as opposed to how to code them
([lessons-implementation.md](lessons-implementation.md)).
`Source:` — **measured**, **read**, **inferred**, **assumed**.

| # | Lesson | Status | Detail |
|---|---|---|---|
| M1 | Prior scale is *the* calibration knob for LLLA: too small → overconfident, too large → underconfident. Nothing about the inference machinery moves the confidence map as much | measured | [prior-scale-calibration](methods/prior-scale-calibration.md) |
| M2 | On two moons, LLLA, VI (`AutoNormal` / `AutoMultivariateNormal`) and NUTS **disagree**: VI and MCMC are both markedly more confident than LLLA at the same prior | measured | [laplace-vs-vi-vs-mcmc](methods/laplace-vs-vi-vs-mcmc.md) |
| M3 | Because MCMC (the reference) agrees with VI rather than with LLLA, the working hypothesis flipped: LLLA's extra uncertainty is likely **bloat**, not better calibration | inferred, open | [laplace-vs-vi-vs-mcmc](methods/laplace-vs-vi-vs-mcmc.md#current-hypothesis) |
| M4 | "Well-calibrated" was judged by eye from confidence maps. No proper metric (ECE, NLL, OOD AUROC) has been computed — so M2/M3 are qualitative | honest gap | [open-questions](open-questions.md#q3) |
| M5 | Reproducing the paper's *figures* is not the same as reproducing its *claim*. The ReLU-overconfidence result is about behaviour far from the data; a matching picture on two moons is weak evidence | inferred | [reproduction-scope](methods/reproduction-scope.md) |

Untried directions that would settle M3 are in
[open-questions.md](open-questions.md).
