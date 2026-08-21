# 0007 — LLLA scored with the probit approximation

**Status:** Accepted, to be revisited · **Date:** 2026-08-18

## Context
Turning a Gaussian over logits into a class probability requires
`E[sigmoid(f)]`, which has no closed form.

## Decision
Use MacKay's probit approximation, as the paper's code does:

```python
v = torch.diag(phi @ sigma @ phi.T)
p = torch.sigmoid(m / np.sqrt(1 + pi / 8 * v))
```

## Alternatives
- Sample from the Gaussian and average sigmoids — what VI and MCMC do here.
- Exact quadrature in 1-D — cheap, but unnecessary if sampling is used anyway.

## Consequences
- Matches the reference implementation, which is what made
  [0002](0002-pyro-alongside-paper-code.md)'s agreement check possible.
- **LLLA is therefore scored with a different estimator than VI and MCMC.** Part of
  the confidence gap in [M2](../methods/laplace-vs-vi-vs-mcmc.md) could be the
  estimator, not the posterior — the probit form deliberately shrinks logits toward 0.

## Revisit if
Nothing — this should just be done: [Q4](../open-questions.md#q4), re-score LLLA by
sampling so all three methods share one predictive.
