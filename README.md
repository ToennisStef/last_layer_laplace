# Important!

LLLA, among other Laplace-approximation flavors are available in the easy-to-use `laplace-torch` library: <https://github.com/AlexImmer/Laplace>.

Please use that library instead of this code repository!


---

# Last-layer Laplace approximation (LLLA)

Code examples of the last-layer Laplace approximation method used in "Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks" paper (ICML 2020). See `bnn_laplace.ipynb` and `bnn_laplace_multiclass.ipynb`. Codes for reproducing the paper's results are in the `paper` directory.


## Running the Pyro re-implementation (uv)

The Pyro work lives in three scripts at the repo root. Dependencies come from
`pyproject.toml`, so `uv run` resolves and installs them on first use — no manual
venv activation. Working notes and decisions are in [doc/](doc/).

```bash
# the four-way comparison at one prior scale (Laplace / VI x2 / NUTS)
uv run python two_moons_comparison.py

# prior-scale sensitivity: one full comparison per var0, into sweeps/<stamp>_prior_scale/
uv run python prior_scale_sweep.py                       # default grid, 8 runs
uv run python prior_scale_sweep.py --var0 1 100 10000    # explicit grid
uv run python prior_scale_sweep.py --plot-only sweeps/<dir>   # re-plot, no inference

# ablation over how much of the net is Bayesian, into ablations/<stamp>_bayes_depth/
uv run python full_bnn_ablation.py --quick --tag smoke   # plumbing check, minutes
uv run python full_bnn_ablation.py --arms ll             # cheapest real arm, d = 21
uv run python full_bnn_ablation.py --arms last2 --skip-full-rank
uv run python full_bnn_ablation.py --arms ll last2 full  # the whole ladder
uv run python full_bnn_ablation.py --plot-only ablations/<dir>
```

`--help` on any of them lists the rest. Notes that matter before launching a long
run:

* **Cost.** The ablation's `full` arm has 501 latent dimensions against 21 for `ll`;
  NUTS and the full-rank guide both scale badly in that number. Run one arm at a
  time, and prefer `--skip-full-rank` for the deep arms (its covariance is
  `d(d+1)/2` parameters — about 125k at `d = 501`). Do not run two of these scripts
  at once: multi-chain NUTS already uses every core, and they will halve each
  other's speed.
* **Nothing is overwritten.** Both the sweep and the ablation write one timestamped
  folder per invocation, with per-run/per-arm subfolders named after what varied.
  Re-running into an existing folder is refused unless `--resume` (which skips
  finished units) or `--force`. `sweeps/**/*.pt` and `ablations/**/*.pt` are
  gitignored; tables, figures and `run.log` are tracked.
* **Windows.** Multi-chain NUTS spawns processes, so anything that calls these
  scripts must sit behind `if __name__ == "__main__"` — all three already do. Import
  `main()` into a notebook and chains silently collapse to one.
* Log progress live with `python -u`; otherwise stdout is block-buffered and each
  run's `run.log` only appears in chunks.

## Requirements

* PyTorch
* Scikit-learn
* Matplotlib
* Seaborn
* Numpy
* [BackPack](https://github.com/f-dangel/backpack)


## Credits

* <https://github.com/AlexMeinke/certified-certain-uncertainty>
* <https://github.com/f-dangel/backpack>
* <https://github.com/wjmaddox/swa_gaussian>
