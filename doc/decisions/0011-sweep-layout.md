# 0011 — Sweeps get one timestamped folder, and import the comparison rather than fork it

**Status:** Accepted · **Date:** 2026-08-24

## Context
[Q8](../open-questions.md#q8) needs the same four-way comparison run at several
prior scales. [0010](0010-artifacts-and-figures-layout.md) writes to a fixed
`artifacts/` + `figures/` pair, so a second run overwrites the first — and its own
"Revisit if" names exactly this case: *runs start being swept over configurations*.
The ad-hoc precedent in the repo (`artifacts_run1_4chains/`, `figures_run1_4chains/`)
does not scale past two runs and encodes nothing about what varied.

## Decision
`prior_scale_sweep.py` **imports** `two_moons_comparison` and calls its `main(cfg)`
once per prior scale with `dataclasses.replace(base, var0=..., artifact_dir=...,
figure_dir=...)`. `two_moons_comparison.py` is not modified.

One timestamped folder per sweep, nothing ever overwritten:

```
sweeps/<UTC stamp>_prior_scale[_<tag>]/
    sweep_config.json    base Config, var0 grid, flags, library versions
    sweep_status.json    per-run outcome: ok / skipped / the exception text
    sweep_summary.json   all collected scalars
    sweep_summary.csv    one row per (run, method)
    sweep_runs.csv       one row per run: diagnostics, gaps vs NUTS
    figures/             cross-run figures
    runs/run<NN>_var0_<v>_ps_<s>/
        run.log          that run's full stdout
        artifacts/       unchanged 0010 layout
        figures/         that run's own three figures
```

Run folders carry both the variance and the scale, plus an index that preserves
sweep order. An existing run folder is refused unless `--resume` (skip finished
runs) or `--force`. Collection reads only from disk, so `--plot-only <dir>`
re-plots without inference.

## Alternatives
- **Copy the script and edit it** — what was asked for first, and rejected: the
  copy drifts, and then the sweep no longer measures the pipeline under study.
- **Return metrics from `main()`** — would mean editing `two_moons_comparison.py`.
  Reading the artifacts back is free and makes `--plot-only` fall out for nothing.
- **A subprocess per run** — better isolation, but `main(cfg)` already calls
  `pyro.clear_param_store()`, and each run is wrapped so one failure cannot end
  the sweep. Revisit if runs start contaminating each other.
- **Flat `sweeps/<param>_<value>/`, no sweep folder** — loses which runs formed one
  sweep, and a second sweep with the same grid collides.

## Consequences
- `sweeps/**/*.pt` is gitignored (tens of MB per sweep); the tables, figures and
  `run.log` files are tracked. A sweep is therefore reviewable from git, but the
  tensors live only on disk.
- Adding another swept parameter means one more `replace()` key and one more field
  in the folder name — the layout does not change.
- `--couple-weight-decay` is **off** by default: `var0` alone moves, and the SGD
  weight decay stays at its base value. This *narrows* but does not remove the
  confound — stage 1 maximises the ELBO of `bnn.model`, which contains the prior,
  so every prior scale still trains a different feature map. `train_accuracy` is
  recorded per run so a worse fit cannot be misread as a wider posterior; see
  [I-prior-enters-map](../gotchas/prior-enters-map-fit.md).
- Both modes also expose that an L2 `weight_decay` on `w` duplicates the Gaussian
  prior already in the model — a second copy, not a consistency guarantee.

## Revisit if
A sweep grows past a few dozen runs (then the per-run `artifacts/` dominates the
disk and the sweep needs to save predictions only), or runs need to go in parallel
(then subprocess isolation becomes necessary, not optional).
