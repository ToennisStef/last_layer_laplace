# 0001 — Last-layer Bayes only, deterministic feature map

**Status:** Accepted, under review · **Date:** 2026-08-18

## Context
Reproducing Kristiadi et al. (2020). The paper offers both a full Laplace
approximation and a last-layer one (LLLA), and argues the last-layer variant is
enough to fix ReLU overconfidence — "being Bayesian, even just a bit".

## Decision
Make only `w` (last layer, `h = 20`, no bias) a `pyro.sample` site. The feature map
`phi` (2 × [Linear → BatchNorm → ReLU]) stays a plain deterministic `nn.Module`,
trained by SGD/MAP.

## Alternatives
- Full BNN over all weights — the paper's other variant; far more expensive and not
  what the headline claim is about.
- Deep kernel / GP last layer — different framework, not a reproduction.

## Consequences
- The posterior is 20-dimensional → exact Hessian, full-covariance VI, and NUTS are
  all affordable, so three inference methods can be compared honestly.
- `phi` must be genuinely frozen for that comparison to mean anything → [0005](0005-freeze-feature-map.md).
- Everything observed is conditional on one point-estimate feature map. Uncertainty
  from the earlier layers is not modelled *at all* — a candidate explanation for the
  LLLA/MCMC gap in [M2](../methods/laplace-vs-vi-vs-mcmc.md).

## Revisit if
The LLLA-vs-MCMC disagreement survives [Q4](../open-questions.md#q4) and
[Q5](../open-questions.md#q5) → go full BNN, [Q2](../open-questions.md#q2).
