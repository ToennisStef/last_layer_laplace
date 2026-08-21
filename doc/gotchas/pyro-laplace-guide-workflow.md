# I3 — `AutoLaplaceApproximation` is two-phase

**Source: read** (Pyro API docs) **+ measured**.

```python
guide = AutoLaplaceApproximation(model, init_loc_fn=...)
svi = SVI(model, guide, optim, Trace_ELBO())      # phase 1: delta guide -> MAP
for _ in range(n_steps):
    svi.step(x, y)
guide.laplace_approximation(x, y)                 # phase 2: Hessian -> MvNormal
```

During phase 1 the guide behaves like `AutoDelta`: sampling from it gives the MAP
point and **zero** predictive spread. `laplace_approximation(*args, **kwargs)`
must be called with the *same model arguments* and returns an
`AutoMultivariateNormal`-like guide holding the Gaussian posterior.

Practical notes:

- Grab `w_map` from `dict(guide.named_pyro_params())["loc"]` before phase 2 if you
  want to reuse it (e.g. as an MCMC init, [I6](mcmc-setup.md)).
- Freeze the feature map before phase 2 ([I4](batchnorm-frozen-features.md)),
  otherwise the Hessian is taken w.r.t. features that are still drifting.

- API: <https://docs.pyro.ai/en/stable/infer.autoguide.html#autolaplaceapproximation>
