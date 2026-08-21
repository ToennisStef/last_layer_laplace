# I9 — Reloading a pyro param store restores some guides, not all

**Source: measured** (2026-08-21, verified against `artifacts/` from run 2).

Two separate problems with `pyro.get_param_store().load(path)`.

## 1. It raises under torch >= 2.6

`ParamStore.save` pickles constraint objects; `torch.load` now defaults to
`weights_only=True` and refuses them:

```
UnpicklingError: Unsupported global: GLOBAL pyro.distributions.constraints._SoftplusPositive
```

Allow-listing classes one at a time is whack-a-mole — the file references *both*
`torch.distributions.constraints.*` and `pyro.distributions.constraints.*` variants
(`_Real`, `_SoftplusPositive`, `_UnitLowerCholesky`, ...). Use
`load_param_store()` in `two_moons_comparison.py`, which trusts the file.

## 2. Worse: it silently restores only part of the state

After rebuilding the model, calling each guide once to materialise its parameters,
and loading the store:

| guide | stored `\|\|loc\|\|` | after reload | |
|---|---|---|---|
| `AutoNormal` | 33.13 | 33.1 | correct |
| `AutoMultivariateNormal` | 33.18 | **0.8** | **silently wrong** |

No exception, no warning — the full-rank guide keeps its *initial* `loc` and
produces a posterior centred near zero. Downstream that looks like a real result
(a directionless posterior) rather than a loading failure. It was caught only
because the reloaded numbers contradicted the confidence recorded at run time.

Also note the guides must be materialised by calling them with **real observations**
(`guide(x, y)`), not `guide(x, None)` — with `y=None` the `obs` site becomes a
latent discrete site and autoguide construction fails with
"Continuous inference cannot handle discrete sample site 'obs'".

## What to do instead

Save what you actually need, not the machinery that produced it:

- **`w_samples.pt`** — posterior draws of `w` per method, written by
  `save_artifacts()`. No reconstruction, no name matching, no version coupling.
- `laplace_posterior.pt` — loc + covariance directly.
- For `AutoMultivariateNormal` specifically, the posterior can be rebuilt by hand:
  `scale_tril_posterior = scale[:, None] * scale_tril` (the stored `scale_tril` is
  unit-lower-cholesky), then `MultivariateNormal(loc, scale_tril=...)`. Verified to
  reproduce the run-time numbers.

General rule: **an artifact you cannot verify against a number recorded at write
time is not a working artifact.** Round-trip checks belong in the run, not in the
docstring — see [0010](../decisions/0010-artifacts-and-figures-layout.md).
