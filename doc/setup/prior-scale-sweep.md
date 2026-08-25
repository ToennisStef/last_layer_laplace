# The prior-scale sweep — one base experiment per prior scale

**Describes:** [`prior_scale_sweep.py`](../../prior_scale_sweep.py). Apparatus, not a
finding — the result is [M1 § measured sweep](../methods/prior-scale-calibration.md#measured-sweep)
and [Q8](../open-questions.md#q8). Symbols in [Symbols](#symbols).

The sweep does not define an experiment of its own. It **imports**
[`two_moons_comparison`](two-moons-comparison.md) and calls its `main(cfg)` once per
prior variance, with `dataclasses.replace` supplying the one changed field. That is
deliberate: a forked copy would drift, and then the sweep would no longer measure the
pipeline under study.

---

## 1. What varies, and what does not

```
  for each var0 in the grid:
      cfg = replace(base_config, var0=var0,
                    artifact_dir=<run folder>/artifacts,
                    figure_dir=<run folder>/figures)
      two_moons_comparison.main(cfg)          # the whole 3-stage experiment
```

| quantity | across the sweep |
|---|---|
| `var0` — the prior variance | **swept** — this is the independent variable |
| output directories | **swept** — one folder per run, so nothing is overwritten |
| `map_weight_decay` | pinned by default; `--couple-weight-decay` sets it to $1/\mathrm{var}_0$ |
| the training data | **invariant** (fixed seed, drawn before any fitting) — checked by `data_fingerprint` |
| the feature map, $w_{\text{MAP}}$ | **not invariant**, even at pinned weight decay — the prior is in stage 1's ELBO ([I10](../gotchas/prior-enters-map-fit.md)) |

That last row is the trap this script is instrumented against. Because stage 1
maximises the ELBO of a model that *contains* the prior, every prior scale trains a
different network, so `train_accuracy` is recomputed per run and reported next to the
calibration numbers. No setting isolates the prior's effect on the posterior from its
effect on the fit.

## 2. The default grid

$\mathrm{var}_0$ is the prior **variance**; the standard deviation
$s = \sqrt{\mathrm{var}_0}$ is what `dist.Normal` takes
([I1](../gotchas/prior-scale-units.md)). Eight values spanning four decades of scale:

| $\mathrm{var}_0$ | 1e-2 | 1e-1 | 1 | 10 | 100 | 1e3 | **2e3** | 1e4 |
|---|---|---|---|---|---|---|---|---|
| $s = \sqrt{\mathrm{var}_0}$ | 0.1 | 0.316 | 1 | 3.16 | 10 | 31.6 | **44.7** | 100 |

The bold column is the repo default, inherited from the weight decay
([0003](../decisions/0003-prior-scale-from-weight-decay.md)). Override with
`--var0 1 100 10000`.

> The range is only safe because the weight decay is **decoupled** by default. Under
> `--couple-weight-decay`, $\mathrm{var}_0 = 10^{-2}$ would mean a weight decay of
> 100, which kills the network outright rather than narrowing the prior. Keep the grid
> at $\mathrm{var}_0 \ge 1$ in that mode.

## 3. Where the grid is split into regions

Every field is summarised over three masks of the $10\,000$-point evaluation grid,
using $r = \lVert x \rVert$:

```
   r = 0        1.5                4.0            7.07
   ├───────────┤                   ├───────────────┤
   │  "near"   │                   │     "far"     │
   │ the moons │                   │  far field    │
   ├───────────────────────────────────────────────┤
   │                    "all"                      │
```

- **near** ($r \le 1.5$) — where the two arcs live.
- **far** ($r \ge 4.0$) — the far field, which is what the paper's asymptotic claim is
  about ([M5](../methods/reproduction-scope.md), [Q7](../open-questions.md#q7)).
- **all** — the whole grid; the corner is at $5\sqrt{2} = 7.07$.

## 4. What is collected

Nothing is recomputed by inference: `collect_run` reads each finished run's artifacts
and reduces them to scalars, which is why `--plot-only` can re-plot a whole sweep with
no fitting.

**Per (run, method)** → `sweep_summary.csv`:

| field | meaning |
|---|---|
| `confidence`, `epistemic`, `aleatoric`, `total` | the [base experiment's](two-moons-comparison.md#8-reported-quantities) fields, each averaged over `all` / `near` / `far` |
| `w_std_mean` | posterior std per coordinate, averaged over the 20 coordinates |
| `w_norm_median` | median $\lVert w \rVert$ of the posterior draws — tests M6's shell radius |

**Per run** → `sweep_runs.csv`:

| field | meaning |
|---|---|
| `var0`, `prior_scale`, `map_weight_decay`, `seed` | what was set |
| `train_accuracy` | fit quality at $w_{\text{MAP}}$, recomputed from `feature_map.pt` — the guard against reading a broken fit as a calibration effect |
| `w_map_norm` | $\lVert w_{\text{MAP}} \rVert$ — the Laplace centre |
| `typical_prior_radius` | $s\sqrt{d}$, M6's predicted shell radius |
| `data_fingerprint` | $\sum x_{\text{train}}$; differing values mean the runs are **unpaired** and the report says so loudly |
| `r_hat_max`, `n_eff_min` | worst-case MCMC diagnostics for that run |
| `map_loss_final`, `elbo_*_final` | optimiser end states |
| `conf_gap_<method>` | mean confidence minus NUTS mean confidence (signed) |
| `conf_gap_far_<method>` | the same, restricted to the far field |
| `p_mean_l1_<method>` | $\text{mean}\,\lvert \bar p - \bar p_{\text{NUTS}} \rvert$ over the grid — **the Q8 quantity** |

No new metric is defined here: everything is either a `Config` field, a base-experiment
field averaged over a region, or a difference between two of them.

## 5. Layout

One timestamped folder per invocation, nothing ever overwritten
([0011](../decisions/0011-sweep-layout.md)):

```
sweeps/<UTC stamp>_prior_scale[_<tag>]/
    sweep_config.json     base Config, the var0 grid, flags, library versions
    sweep_status.json     per-run outcome: ok / skipped / the exception text
    sweep_summary.json    everything collected
    sweep_summary.csv     one row per (run, method)
    sweep_runs.csv        one row per run
    figures/              six cross-run figures
    runs/run00_var0_1.0e-02_ps_1.0e-01/
        run.log           that run's full stdout
        artifacts/        the unchanged doc/decisions/0010 layout
        figures/          that run's own three figures
```

Run folders carry the variance **and** the scale, plus an index preserving grid order.
A run folder that already holds results is refused unless `--resume` (skip finished
runs) or `--force`.

## 6. The six cross-run figures

| file | shows |
|---|---|
| `confidence_vs_prior_scale.svg` | mean confidence vs $s$, one line per method, one panel per region |
| `epistemic_vs_prior_scale.svg` | the same for $I[y; w \mid x]$ |
| `posterior_width_vs_prior_scale.svg` | `w_std_mean` and median $\lVert w \rVert$ against $s$, with the reference lines $y = s$ (prior std) and $y = s\sqrt{d}$ (prior typical radius), plus $\lVert w_{\text{MAP}} \rVert$ |
| `gap_vs_nuts_vs_prior_scale.svg` | the Q8 verdict: signed confidence gap and $\lvert \bar p - \bar p_{\text{NUTS}} \rvert$ |
| `mcmc_diagnostics_vs_prior_scale.svg` | $\hat r$, $n_{\text{eff}}$, and **train accuracy** — the confound check sits in the diagnostics panel on purpose |
| `confidence_grid.svg` | rows = prior scale, columns = method; the poster-candidate figure |

All are drawn in the BAM categorical palette, black reserved for NUTS because it is
the reference.

## 7. Running it

```bash
uv run python -u prior_scale_sweep.py                        # default 8-point grid
uv run python -u prior_scale_sweep.py --var0 1 100 10000     # explicit grid
uv run python -u prior_scale_sweep.py --couple-weight-decay  # wd = 1/var0
uv run python -u prior_scale_sweep.py --quick --tag smoke    # plumbing check
uv run python -u prior_scale_sweep.py --resume sweeps/<dir>  # continue
uv run python prior_scale_sweep.py --plot-only sweeps/<dir>  # re-plot, no inference
```

**Cost.** One run is a full base experiment — 5000 MAP steps, $2 \times 10\,000$ VI
steps, and $5 \times 1500$ NUTS iterations — so the default grid is eight of those.
Measured on this machine: ~25 min per run at moderate $s$, but **2.5 hours** at
$s = 100$, because the typical set sits at radius $s\sqrt{d} \approx 447$ and NUTS
needs long trajectories to traverse it. Loose priors are disproportionately expensive.

`--resume` is the recovery path: finished runs are skipped by the presence of their
`predictions.pt`, so an interrupted sweep continues where it stopped and then
re-collects everything.

## Symbols

| symbol | meaning |
|---|---|
| $\mathrm{var}_0$, $s = \sqrt{\mathrm{var}_0}$ | prior variance; prior standard deviation |
| $\lambda$ | `map_weight_decay` — a prior *precision* |
| $d = 20$ | latent dimension (the last layer) |
| $w$, $w_{\text{MAP}}$ | last-layer weights; their MAP value |
| $r = \lVert x \rVert$ | distance of a grid point from the origin |
| $\bar p$, $\bar p_{\text{NUTS}}$ | posterior-mean predicted probability; the same under NUTS |
| $I[y; w \mid x]$ | mutual information — the epistemic term |
| $\hat r$, $n_{\text{eff}}$ | MCMC diagnostics: split-$\hat R$, effective sample size |
