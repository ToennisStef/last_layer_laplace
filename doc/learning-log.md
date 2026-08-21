# Learning log

The lab log: chronological record of *how* each lesson was arrived at. Newest last.
Each entry: what was done → what was seen → what it produced → **Transferable?**

`Transferable?` asks: *would this still be true in a different codebase?*
`no` means it belongs here and nowhere else. `yes` means it is owed a note outside
the repo, restated in own words — never copy-pasted — per
[decision 0008](decisions/0008-notes-live-in-repo.md).

---

## 2026-08-18 — LLLA reimplementation
**Goal:** reproduce *Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU
Networks* (Kristiadi et al., ICML 2020), last-layer Laplace variant, in Pyro.
Files: `bnn_laplace.ipynb` (reference), `two_moons_ll_Laplace.ipynb`.

- Ported `Model` (MAP + exact Hessian via `hessian.py`) and a Pyro `BayesianMLP`
  with `AutoLaplaceApproximation`.
- **Result: reproduction succeeded** — the Pyro LLLA confidence map matches the
  paper's implementation.
- Along the way: unit confusion between variance / std / precision
  → [I1](gotchas/prior-scale-units.md); autoguide init put every weight at 0
  → [I2](gotchas/pyro-autoguide-init.md); two-phase Laplace guide
  → [I3](gotchas/pyro-laplace-guide-workflow.md).
- Choices made here: [0001](decisions/0001-last-layer-only.md) last-layer only,
  [0002](decisions/0002-pyro-alongside-paper-code.md) dual implementation,
  [0004](decisions/0004-fan-in-guide-init.md) guide init,
  [0007](decisions/0007-probit-predictive.md) probit predictive.
- **Transferable?** yes — "reproduce against a second independent implementation
  before believing a method result" is not specific to this code.

## 2026-08-19 — prior scale is the calibration knob
**Trigger:** confidence maps looked wrong before the units were fixed.

- Sweeping the prior scale showed the whole spectrum: small → overconfident,
  large → underconfident, with the qualitative "band of uncertainty" picture only
  appearing in a narrow range.
- **Lesson [M1](methods/prior-scale-calibration.md)**: calibration is set by the
  prior, not by the inference method.
- Read in the paper that Ritter et al. (2018) tune this scale automatically.
  **Not read, not implemented** → parked as [Q1](open-questions.md#q1).
- Choice made here: [0003](decisions/0003-prior-scale-from-weight-decay.md).
- **Transferable?** yes — prior scale as the dominant calibration knob in BNNs is a
  claim about the method, not about this repo.

## 2026-08-19/20 — VI does not reproduce LLLA
Files: `two_moons_ll_AutoNormal.ipynb`.

- Trained SVI with `AutoNormal`, then `AutoMultivariateNormal`, same prior.
- **Observed:** both are clearly *more confident* than LLLA. First hypothesis:
  "the VI approximation is bad" (mean-field variance underestimation).
- Full-covariance MVN did not close the gap → hypothesis weakened.
- Freezing the feature map correctly became a prerequisite for a fair comparison
  → [I4](gotchas/batchnorm-frozen-features.md) and [0005](decisions/0005-freeze-feature-map.md).
- **Transferable?** partly — "mean-field underestimates variance" is a general claim
  (and it did *not* explain what was seen here); the BatchNorm freezing is repo-local.

## 2026-08-20 — MCMC baseline, and the hypothesis flips
Files: `two_moons_ll_MCMC.ipynb`, `two_moons_ll_Laplace_MCMC.ipynb`,
`two_moons_comparison.py`, `conf_mcmc.png`, `p_mcmc.png`, `w_pairplot.png`.

- Ran NUTS (5 chains × 500 samples, 1000 warmup) on the last-layer weights, seeded
  at `w_map` → [I6](gotchas/mcmc-setup.md).
- **Observed:** MCMC is also much more confident than LLLA — it sides with VI.
- **This inverted the interpretation.** If MCMC is the reference, VI was never the
  problem: **LLLA looks bloated**, i.e. underconfident, not better calibrated
  → [M2/M3](methods/laplace-vs-vi-vs-mcmc.md).
- Doubt raised in the same session: maybe last-layer-only Bayes is too little
  ([Q2](open-questions.md#q2)), or the ReLU-overconfidence framing does not transfer
  to this toy setting at all ([M5](methods/reproduction-scope.md), [Q7](open-questions.md#q7)).
- Choice made here: [0006](decisions/0006-nuts-as-ground-truth.md).
- **Transferable?** yes, once verified — "a Laplace posterior at the MAP can be
  far wider than the true posterior along weakly-determined directions" is a
  general claim. **Do not promote it until [Q5](open-questions.md#q5) is done.**

## Open at end of 2026-08-20
The comparison is not yet trustworthy enough to conclude anything about LLLA:
same-estimator scoring ([Q4](open-questions.md#q4)), MCMC convergence audit
([Q5](open-questions.md#q5)) and real calibration metrics ([Q3](open-questions.md#q3))
all still missing.
