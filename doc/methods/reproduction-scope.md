# M5 — What a matching figure does and does not prove

**Source: inferred** (2026-08-20).

The claim of Kristiadi et al. (2020) is asymptotic: a ReLU network's softmax
confidence → 1 as `‖x‖ → ∞`, and *any* Gaussian posterior on the last layer
provably bounds that confidence away from 1. The two-moons figure is an
illustration of the bound, not evidence about calibration.

Therefore:

- Reproducing the figure (done ✔) confirms the **implementation**, not that LLLA
  gives *well-calibrated* uncertainty at finite `x`.
- The test grid here is `[-5, 5]²` — close to the data by the standards of the
  asymptotic argument. Confidence differences on that grid are outside what the
  theorem speaks to.
- "LLLA is underconfident" ([M3](laplace-vs-vi-vs-mcmc.md#current-hypothesis)) and
  "the bound holds" are compatible statements: a loose-but-valid bound is exactly
  what bloated uncertainty looks like.

Practical consequence: to test *calibration* rather than the bound, use a metric on
held-out and OOD data ([Q3](../open-questions.md#q3)), not the picture.
