# 0005 — Freeze the feature map with `eval()` *and* `requires_grad_(False)`

**Status:** Accepted · **Date:** 2026-08-20

## Context
[0001](0001-last-layer-only.md) assumes `phi` is fixed. BatchNorm breaks that in two
independent ways: trainable affine params, and running statistics that keep updating
on every forward pass — including over the dense test grid.

## Decision
Before the Laplace covariance step and before MCMC:

```python
bnn.feature_map.eval()
for p in bnn.feature_map.parameters():
    p.requires_grad_(False)
```

## Alternatives
- `torch.no_grad()` at the call site — stops gradients, does not stop running-stat updates.
- Drop BatchNorm from the architecture — cleanest, but diverges from the paper's net.

## Consequences
- All three inference methods see the identical `phi`, which is what makes
  [M2](../methods/laplace-vs-vi-vs-mcmc.md) a comparison of *inference* rather than
  of features.
- Ordering becomes load-bearing: freeze after SVI phase 1, before
  `laplace_approximation()` ([I3](../gotchas/pyro-laplace-guide-workflow.md)).

## Revisit if
BatchNorm is replaced (LayerNorm / none) — then this decision is moot, not wrong.
