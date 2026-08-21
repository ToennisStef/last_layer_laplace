# 0010 — Runs persist to `artifacts/`, figures to `figures/`

**Status:** Accepted · **Date:** 2026-08-21

## Context
Each comparison run costs a MAP fit, two VI fits and 4x1500 NUTS steps. Re-running
it to change a colormap is waste, and a figure whose inputs are gone cannot be
checked later. The data itself is *random* (`two_moons` draws fresh points), so a
figure is not reproducible from the script alone.

## Decision
Two folders, written by `save_artifacts()` and read back by `load_artifacts()`:

- `artifacts/` — `config.json` (full `Config` + torch/pyro/numpy versions +
  timestamp), `data.pt` (**training data and grid**), `feature_map.pt`,
  `param_store.pt`, `laplace_posterior.pt`, `mcmc_samples.pt` (pooled *and* by
  chain), `mcmc_diagnostics.json` (`r_hat`, `n_eff`), `predictions.pt`, `losses.pt`.
- `figures/` — `confidence_comparison.svg`, `epistemic_comparison.svg`,
  `mcmc_pairplot.svg`.

Reproducibility rests on `Config.seed` through `pyro.set_rng_seed`, plus saving
the realised data so a figure can be regenerated even if seeding behaviour
changes across versions.

## Alternatives
- Pickle the guide objects wholesale — brittle across pyro/torch versions, and it
  pickles a bound method of the model.
- One folder for everything — figures are small, text-ish and reviewable; the
  `.pt` files are large binaries. Different lifecycles, different folders.

## Consequences
- Re-plotting needs no inference: `load_artifacts()["predictions"]` holds the
  per-method fields.
- Guides come back by *reconstruction* (rebuild `BayesianMLP`, load
  `feature_map.pt`, `init_guides()`, `param_store.load(...)`), not by unpickling —
  documented in `load_artifacts`'s docstring. Renaming the guide attributes on
  `BayesianMLP` would break the param-name match.
- `artifacts/` grows by ~a few MB per run and is **not** gitignored yet — decide
  before committing a run.

## Revisit if
Runs start being swept over configurations — then each run needs its own
timestamped subfolder rather than overwriting `artifacts/`.
