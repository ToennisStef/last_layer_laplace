# I — The prior changes the MAP fit, so a prior sweep is never a clean A/B

**Source: measured** (2026-08-24, `prior_scale_sweep.py --quick`, two prior scales
at a fixed `map_weight_decay = 5e-4`).

Sweeping `Config.var0` with everything else pinned does **not** hold the network
fixed. Same seed, same data, same weight decay — and still:

| prior scale `s` | train accuracy at `w_map` |
|---|---|
| 0.1 | 0.850 |
| 44.7 | 0.960 |

## Why
Stage 1 is `SVI(model=bnn.model, guide=AutoLaplaceApproximation, ...)`, which acts
as `AutoDelta`, so its ELBO **is** `-log p(w, D)` — likelihood *and* prior. The
prior scale is therefore part of the stage-1 objective, and it is optimised jointly
with the feature map. A tighter prior pulls `w` in, the feature map compensates, and
the fit that the posterior is later built on is a different fit.

Consequence for a prior sweep: a change in mean confidence cannot be attributed to
the posterior alone. Record the train accuracy per run and read the two together —
where accuracy drops, part of the confidence change is a worse fit, not a wider
posterior. `prior_scale_sweep.py` reports `train_accuracy` for exactly this reason
(recomputed from `feature_map.pt` + `w_map`, which `save_artifacts` does not store).

## Second point, same line of code — first noticed 2026-08-21

The double-counting below was already spotted while reading the paper's code (see
the [learning log](../learning-log.md) entry for 2026-08-21, "our Pyro port is a
factor of 2 off for the same reason"). What is new here is the *first* point above:
the prior also reaches the MAP through the ELBO, so it moves the feature map even
when the weight decay is held fixed.

`optim=SGD({..., "weight_decay": cfg.map_weight_decay})` applies an L2 penalty to
**every** trainable parameter, $w$ included. For $w$, that penalty is a *second*
Gaussian prior on top of the one already in `bnn.model`, so the precisions add:

$$\underbrace{\tau_{\text{eff}}}_{\text{what stage 1 really applies to } w}
\;=\;
\underbrace{\frac{1}{\mathrm{var}_0}}_{\text{prior inside the model}}
\;+\;
\underbrace{\lambda}_{\text{optimiser weight decay}}$$

- $\tau_{\text{eff}}$ — effective prior precision on $w$ during stage 1.
- $\mathrm{var}_0$ — the prior variance declared in the model.
- $\lambda$ — `map_weight_decay`.

Setting $\mathrm{var}_0 = 1/\lambda$
([0003](../decisions/0003-prior-scale-from-weight-decay.md)) makes the two copies
*agree*; it does not make there be one. At the matched setting
$\tau_{\text{eff}} = 2\lambda$ — **exactly double**.

**This is a loose-prior problem only.** What matters is the weight decay's *share* of
the total, $\lambda / (1/\mathrm{var}_0 + \lambda) = \lambda\,\mathrm{var}_0 / (1 + \lambda\,\mathrm{var}_0)$:

| $s$ | 0.1 | 1 | 10 | 31.6 | 44.7 | 100 |
|---|---|---|---|---|---|---|
| weight decay's share of $\tau_{\text{eff}}$ | 0.000% | 0.05% | 4.8% | 33% | 50% | 83% |

So the double count is negligible for a tight prior and dominant for a loose one. Do
**not** reach for it to explain a tight-prior pathology — at $s = 0.1$ the declared
prior has precision 100 against the weight decay's $5\times10^{-4}$, and any
over-regularisation there is the model's own prior
([M1](../methods/prior-scale-calibration.md#measured-sweep)).

Two further consequences:

- **the MAP location is capped, and only the MAP location.** Since
  $\mathrm{var}_0^{\text{eff}} = 1/\tau_{\text{eff}} \le 1/\lambda$, the mode stops
  moving outward once $1/\mathrm{var}_0 \ll \lambda$: measured,
  $\lVert w_{\text{MAP}} \rVert$ plateaus at 14.29 → 16.94 → 17.14 → 17.30 for
  $s = 10 \dots 100$. But stage 1 is the *only* place $\lambda$ appears — the
  Laplace Hessian, the VI objective and the NUTS target all use the model prior
  alone, so the posterior **width** is not capped at all and grows $\propto s$
  (NUTS sd$/s$ = 0.774, 0.776, 0.776, 0.778 over the same range). Getting this
  wrong cost a falsified prediction, recorded in
  [M1](../methods/prior-scale-calibration.md#measured-sweep);
- only the feature map is regularised by the weight decay alone, since the model
  places no prior on it.

**Transferable?** yes — any Pyro model fitted by SVI with an `AutoDelta`-like guide
*and* an optimiser-level `weight_decay` double-counts its own prior.
