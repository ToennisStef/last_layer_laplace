# M1 — Prior scale sets LLLA calibration

**Source: measured** (2026-08-18/19, two moons, `h = 20`, last layer only).

With `w ~ Normal(0, s)` on the last layer:

| prior scale `s` | effect on the confidence map |
|---|---|
| too small | posterior covariance ≈ prior-dominated and tight → **overconfident** far from data |
| too large | `v = diag(φ Σ φᵀ)` blows up, probit shrinks logits → **underconfident** everywhere |

Current setting: `var0 = 1/5e-4 = 2000`, `s = sqrt(var0) ≈ 44.7`, matched to the
MAP baseline's weight decay `λ = 5e-4` (`s = sqrt(1/λ)`, [I1](../gotchas/prior-scale-units.md)).
That match is a *consistency* argument, not a calibration argument — the weight
decay was itself inherited from the paper's code, not tuned for calibration.

Consequences:

- Any comparison of LLLA against VI or MCMC is only meaningful **at the same prior**
  ([M2](laplace-vs-vi-vs-mcmc.md)); the prior is shared, so a prior-driven
  disagreement would show up in all three.
- The knob is a hyperparameter that ought to be *fit*, not guessed. Ritter et al.
  (2018) tune it by marginal likelihood / validation — **not yet tried**,
  [Q1](../open-questions.md#q1).

Do not fan-in scale the prior — [I1](../gotchas/prior-scale-units.md).
