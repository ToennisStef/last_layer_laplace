# 0006 — NUTS treated as the reference posterior

**Status:** Accepted, contested · **Date:** 2026-08-20

## Context
LLLA and VI disagreed ([M2](../methods/laplace-vs-vi-vs-mcmc.md)). With two
approximations and no reference, there was no way to say which was wrong. The
20-dimensional last-layer posterior ([0001](0001-last-layer-only.md)) is small enough
for asymptotically exact sampling.

## Decision
Run NUTS — 5 chains × 500 samples, 1000 warmup, seeded at `w_map` via
`init_to_value` — and treat its posterior as ground truth for the comparison.

## Alternatives
- Trust the ELBO to rank VI variants — measures fit to *its own* bound, not to the truth.
- Analytic posterior — unavailable; Bernoulli likelihood is non-conjugate.

## Consequences
- Reverses the interpretation of the whole comparison: LLLA, not VI, becomes the
  outlier ([M3](../methods/laplace-vs-vi-vs-mcmc.md#current-hypothesis)).
- The conclusion inherits MCMC's failure modes. Seeding at `w_map` in particular
  biases toward "the posterior is narrow" if the chains never leave the mode.

## Revisit if
[Q5](../open-questions.md#q5) shows poor `r_hat` / low ESS / chains stuck at the
mode — then the reference is unusable and M3 collapses.
