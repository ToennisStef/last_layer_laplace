# Implementation gotchas

Library-level traps hit while porting last-layer Laplace (LLLA) to Pyro.
`Source:` — how it was learned: **measured** (own experiment), **read** (docs/paper),
**inferred** (reasoning, unverified), **assumed** (belief, flag for checking).

| # | Gotcha | Status | Detail |
|---|---|---|---|
| I1 | `dist.Normal(loc, scale)` wants a **std**, not a variance; weight decay λ is a **precision** → `scale = sqrt(1/λ)` | measured | [prior-scale-units](gotchas/prior-scale-units.md) |
| I2 | Pyro autoguide defaults (`init_to_median`/`init_to_feasible`) put every NN weight at 0 → no symmetry breaking. Override with `init_to_value` + fan-in draw | read + measured | [pyro-autoguide-init](gotchas/pyro-autoguide-init.md) |
| I3 | `AutoLaplaceApproximation` is two-phase: SVI finds the MAP (delta guide), then `.laplace_approximation(*args)` builds the covariance. Forgetting phase 2 silently leaves a point estimate | read | [pyro-laplace-guide-workflow](gotchas/pyro-laplace-guide-workflow.md) |
| I4 | With a frozen feature map, `BatchNorm` must be in `.eval()` **and** the params `requires_grad_(False)` before the Laplace/MCMC step — otherwise the "fixed" features move | measured | [batchnorm-frozen-features](gotchas/batchnorm-frozen-features.md) |
| I5 | LLLA needs the Hessian of the **negative log posterior** (NLL + neg-log-prior) and a `reduction="sum"` likelihood — a mean-reduced BCE rescales the Hessian by 1/N | measured | [hessian-scaling](gotchas/hessian-scaling.md) |
| I6 | NUTS on last-layer weights: seed with `init_strategy=init_to_value({"w": w_map})`, use several chains, and check `r_hat`/ESS from `mcmc.summary()` before believing the comparison | measured | [mcmc-setup](gotchas/mcmc-setup.md) |
| I7 | Multi-chain `MCMC` on Windows spawns processes — the **entry script** needs `if __name__ == "__main__"`, or chains collapse to one | measured | [windows-multiprocessing-mcmc](gotchas/windows-multiprocessing-mcmc.md) |
| I8 | `AutoNormal` has no `get_posterior()` (it is not an `AutoContinuous`); comparing posterior spread across guide families means sampling `w` through `Predictive` for all of them | measured | — |
| I9 | `param_store.load()` raises under torch>=2.6 **and** silently leaves `AutoMultivariateNormal.loc` unrestored — save posterior draws instead of guide machinery | measured | [param-store-reload](gotchas/param-store-reload.md) |

Related: methodological consequences of I1 are in
[lessons-methodology.md](lessons-methodology.md) (M1).
