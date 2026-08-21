# I6 — NUTS on last-layer weights

**Source: measured** (2026-08-19/20).

```python
kernel = NUTS(bnn.model, init_strategy=init_to_value(values={"w": w_map}))
mcmc = MCMC(kernel, num_samples=500, warmup_steps=1000, num_chains=5)
mcmc.run(x_train, y_train)
print(mcmc.summary())          # check r_hat ≈ 1.0 and n_eff before comparing
```

Notes:

- Seeding at `w_map` is cheap and cuts warmup, but it also means a badly mixing
  chain can look fine by staying near the MAP — `r_hat`/ESS is the check, not the
  picture.
- Multiple chains are the only way to see multimodality; the last-layer posterior
  is not obviously unimodal even though the model is linear-in-`w` *given* `phi`.
  (With a Bernoulli likelihood and a fixed `phi`, the posterior in `w` **is**
  log-concave → unimodal — **inferred**, worth confirming against the samples.)
- The feature map must already be frozen ([I4](batchnorm-frozen-features.md)) or
  every chain sees a different `phi`.
- `Predictive(..., return_sites=("logits",))` returns shape
  `[num_samples * num_chains, n_test]`; reshape to the plotting grid **per sample**,
  not on the flattened tensor.

MCMC's role here was as a *ground-truth baseline* for the comparison — see
[M2](../methods/laplace-vs-vi-vs-mcmc.md).

- API: <https://docs.pyro.ai/en/stable/mcmc.html>
