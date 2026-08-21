# 0002 — Keep the paper's exact-Hessian code as a reference implementation

**Status:** Accepted · **Date:** 2026-08-18

## Context
Two ways to get the Laplace posterior: the paper's explicit route (`hessian.py`,
`exact_hessian` of the negative log posterior, then `torch.inverse`) or Pyro's
`AutoLaplaceApproximation`. A single implementation gives no way to tell a porting
bug from a method property.

## Decision
Implement both and keep both in [two_moons_comparison.py](../../two_moons_comparison.py):
`Model` (paper's route) and `BayesianMLP` (Pyro). Treat agreement between them as
the reproduction criterion.

## Alternatives
- Pyro only — faster, but then any surprise is unattributable.
- `laplace-torch` — the library the README recommends; deliberately deferred to
  [Q6](../open-questions.md#q6) so the reproduction stays first-principles.

## Consequences
- The reproduction claim in [learning-log](../learning-log.md) rests on two
  independent implementations agreeing, which is why
  [M3](../methods/laplace-vs-vi-vs-mcmc.md#current-hypothesis) can point at the
  *method* rather than at a bug.
- Two code paths to keep in sync; the prior scale in particular must be identical in
  both — see [0003](0003-prior-scale-from-weight-decay.md), [I1](../gotchas/prior-scale-units.md).

## Revisit if
The two implementations ever disagree — that is a bug hunt, not a finding.
