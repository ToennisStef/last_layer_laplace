# 0009 — Score every method with the same sampled predictive

**Status:** Accepted · **Date:** 2026-08-21 · **Supersedes:** [0007](0007-probit-predictive.md) *for the comparison script*

## Context
[0007](0007-probit-predictive.md) scored Laplace with MacKay's probit
approximation while VI and MCMC averaged sampled sigmoids. That left the
confidence gap in [M2](../methods/laplace-vs-vi-vs-mcmc.md) partly attributable
to the *estimator* rather than the posterior — logged as [Q4](../open-questions.md#q4).

## Decision
`predict_probs()` in `two_moons_comparison.py` draws weights from the posterior,
pushes them through the network, and takes `sigmoid`, for **all four** methods:
Laplace, `AutoNormal`, `AutoMultivariateNormal`, NUTS. Posterior spread is
likewise measured uniformly, as the std of sampled `w` (see I8).

## Alternatives
- Probit everywhere — not available for NUTS, which has no Gaussian to plug in.
- Keep the mixed scoring — cheaper, but the comparison would stay confounded.

## Consequences
- Differences between the four maps are now attributable to the posterior alone.
- The paper's `Model.predict` still uses probit, so it remains the reference for
  reproducing the paper's *figure* ([0002](0002-pyro-alongside-paper-code.md));
  0007 is not wrong, it is simply not what the comparison uses.
- Costs `pred_samples x n_grid` forward passes per method, hence the chunking in
  `predict_probs`.

## Revisit if
Sampling noise at `pred_samples=1000` turns out to be visible in the maps — then
raise the draw count rather than reverting to a closed form.
