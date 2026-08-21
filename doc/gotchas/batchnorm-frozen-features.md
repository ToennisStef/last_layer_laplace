# I4 — Freezing the feature map (BatchNorm)

**Source: measured** (2026-08-19/20).

LLLA treats the network as `logits = phi(x) @ w`, with `phi` **deterministic and
fixed**. With `BatchNorm` in the feature map, "fixed" needs two separate actions:

```python
bnn.feature_map.eval()                       # use running stats, stop updating them
for p in bnn.feature_map.parameters():
    p.requires_grad_(False)                  # keep gradients/Hessian confined to w
```

- `.eval()` alone still leaves the affine params trainable.
- `requires_grad_(False)` alone still lets the running mean/var update on every
  forward pass — the test-time features then depend on how many grid points were
  pushed through, which quietly changes the confidence map.
- BatchNorm also makes `phi(x)` depend on the *batch*: evaluating a dense test grid
  in `train()` mode normalises against the grid, not the training data.

Applies to both the Laplace covariance step ([I3](pyro-laplace-guide-workflow.md))
and the MCMC run ([I6](mcmc-setup.md)) — otherwise the three methods are not
comparing the same `phi`.

- API: <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>
