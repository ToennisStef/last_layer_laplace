# The base experiment — four posteriors over one frozen feature map

**Describes:** [`two_moons_comparison.py`](../../two_moons_comparison.py). Counts
verified against the code 2026-08-25 (regeneration snippet at the end). Apparatus,
not a finding — findings are in [M2](../methods/laplace-vs-vi-vs-mcmc.md).

This is the experiment every other script builds on:
[`prior_scale_sweep.py`](prior-scale-sweep.md) runs it once per prior scale, and
[`full_bnn_ablation.py`](bayesian-depth-arms.md) reuses its data, predictive
estimator and `Config`. Symbols are listed in [Symbols](#symbols).

---

## 1. The question the setup is built for

Fit **four different posteriors over exactly the same last-layer problem**, so that
any difference between their confidence maps is attributable to the inference method
alone:

| | method | Pyro object |
|---|---|---|
| 1 | Laplace approximation at the MAP | `AutoLaplaceApproximation` |
| 2 | VI, mean-field Gaussian | `AutoNormal` |
| 3 | VI, full-rank Gaussian | `AutoMultivariateNormal` |
| 4 | **NUTS** — the reference | `MCMC(NUTS(...))` |

The two things that make the comparison fair are that all four share one **frozen
feature map** ([0005](../decisions/0005-freeze-feature-map.md)) and one **predictive
estimator** ([0009](../decisions/0009-uniform-sampled-predictive.md)).

## 2. The data

`two_moons(n, sigma)` is **not** `sklearn.make_moons`. It draws an angle
$\theta \sim \mathcal{U}(0, 2\pi)$, splits the circle in half to make the label, and
offsets the two halves:

$$y = \mathbb{1}[\theta > \pi], \qquad
x = \begin{pmatrix} \cos\theta + y - \tfrac{1}{2} \\[2pt] \sin\theta + \tfrac{y}{2} - \tfrac{1}{4} \end{pmatrix} + \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

- $\theta$ — angle on the unit circle; $y \in \{0, 1\}$ — the class label.
- $\varepsilon$ — isotropic Gaussian noise, $\sigma = 0.1$ (`Config.noise`).
- Class 0 is the upper semicircle shifted by $(-\tfrac12, -\tfrac14)$; class 1 is the
  lower semicircle shifted by $(+\tfrac12, +\tfrac14)$ — two interleaving arcs.

$N = 200$ training points (`Config.n_train`). The data is **separable**, which is not
incidental: it is what collapses the likelihood's Hessian and lets the prior dominate
the posterior ([M2](../methods/laplace-vs-vi-vs-mcmc.md)).

**Evaluation grid:** $100 \times 100 = 10\,000$ points on $[-5, 5]^2$, so the corner
sits at $\lVert x \rVert = 5\sqrt{2} = 7.07$. The grid is the entire evaluation —
there is no held-out set, which is why [Q3](../open-questions.md#q3) is open.

## 3. Two model classes, for two different jobs

The file defines **two** networks, and confusing them is easy:

| class | purpose | inference | predictive |
|---|---|---|---|
| `Model` | reproduce the paper's figure | hand-built exact Hessian via `hessian.exact_hessian` | MacKay probit, `Model.predict` |
| `BayesianMLP` | the four-way comparison | Pyro guides + NUTS | sampled, `predict_probs` |

`Model` exists to check the port against the paper
([0002](../decisions/0002-pyro-alongside-paper-code.md)) and keeps the probit form
([0007](../decisions/0007-probit-predictive.md)). `BayesianMLP` is what `main()`
actually runs.

## 4. Architecture and parameter count

```
  x ∈ R²
    │
    ├─ Linear (2 → 20) ──► BatchNorm ──► ReLU  ┐
    │                                          ├─ feature map φ(x), frozen after stage 1
    ├─ Linear (20 → 20) ─► BatchNorm ──► ReLU  ┘
    │
    └─ Linear (20 → 1), no bias ──────────────────►  logit  f(x) = φ(x)ᵀw
```

| component | shape | parameters |
|---|---|---|
| `Linear(2, 20)` | $(20,2)$ + $(20,)$ | 60 |
| `BatchNorm1d(20)` | $\gamma, \beta$ | 40 |
| `Linear(20, 20)` | $(20,20)$ + $(20,)$ | 420 |
| `BatchNorm1d(20)` | $\gamma, \beta$ | 40 |
| **feature map $\phi$, total** | | **560** |
| `clf = Linear(20, 1, bias=False)` | $(1,20)$ | **20** |
| **whole network** | | **580** |

Plus **82 BatchNorm buffers** (two layers' `running_mean`, `running_var`, and their
`num_batches_tracked` counters). Buffers are not parameters — they are estimated by a
running average during stage 1, then frozen.

**The Bayesian part is only the 20 weights of `clf`.** That is the whole content of
"last-layer Laplace" ([0001](../decisions/0001-last-layer-only.md)): the posterior
lives in $\mathbb{R}^{20}$, and the other 560 parameters are a point estimate.

> **Note the missing bias.** `clf` has `bias=False`, matching the paper, so the latent
> dimension is **20**. The ablation's `ll` arm gives its output layer a bias and so
> has **21** — the two are near-identical setups but not the same number
> ([arms](bayesian-depth-arms.md)).

## 5. The three stages of `main()`

```
  stage 1   MAP            SGD(lr=1e-3, momentum=0.9, weight_decay=5e-4), 5000 steps
            │              optimises  w  AND  the feature map jointly
            │              guide = AutoLaplaceApproximation, which acts as AutoDelta
            │              so the ELBO is exactly  -log p(w, D)
            ▼
          FREEZE           feature_map.eval()  +  requires_grad_(False)
            │              both are needed (I3), and train accuracy is printed here
            ▼
  stage 2   three Gaussians   Laplace: .laplace_approximation(x, y) — exact Hessian
            │                 VI x2:   ClippedAdam(lr=1e-2, decayed), 10000 steps each
            ▼
  stage 3   NUTS           5 chains x 500 draws, 1000 warmup, init_to_sample()
            │
            ▼
          predictive       all four scored by predict_probs: sample w, average sigmoids
```

Three choices in there are load-bearing and each has a record:

- **SGD for stage 1, ClippedAdam for stage 2.** SGD+momentum reaches a better MAP
  (it matches the paper's loop) but diverges for `AutoMultivariateNormal`, whose
  `scale_tril` goes through an `exp` transform.
- **`init_to_sample()` for NUTS**, not the MAP. Seeding at $w_{\text{MAP}}$ makes the
  chains report a narrow posterior because they never leave the mode — that error
  produced a conclusion that had to be retracted
  ([Q5](../open-questions.md#q5), [0006](../decisions/0006-nuts-as-ground-truth.md)).
- **Guides initialised from a fan-in draw**, not Pyro's default `init_to_median`,
  which would start every weight at 0 ([0004](../decisions/0004-fan-in-guide-init.md)).

## 6. What `Config` controls

Everything that changes the numbers lives in one frozen dataclass, serialised next to
the artifacts, so `config.json` plus the seed reproduces a run.

| field | default | what it sets |
|---|---|---|
| `seed` | 7777 | all randomness, via `pyro.set_rng_seed` |
| `n_train`, `noise` | 200, 0.1 | the dataset |
| `grid_lim`, `grid_n` | 5.0, 100 | the evaluation grid |
| `h`, `k` | 20, 1 | hidden width, output width |
| `var0` | $1/5\times10^{-4} = 2000$ | prior **variance**; $s = \sqrt{\mathrm{var}_0} \approx 44.7$ is the std ([I1](../gotchas/prior-scale-units.md)) |
| `map_steps`, `map_lr`, `map_momentum`, `map_weight_decay` | 5000, 1e-3, 0.9, 5e-4 | stage 1 |
| `vi_steps`, `vi_lr`, `vi_particles`, `vi_lr_final_frac` | 10000, 1e-2, 8, 0.1 | stage 2 |
| `mcmc_samples`, `mcmc_warmup`, `mcmc_chains` | 500, 1000, 5 | stage 3 |
| `pred_samples`, `pred_chunk` | 1000, 2500 | the predictive; chunking caps peak memory |
| `artifact_dir`, `figure_dir` | `artifacts`, `figures` | where output goes |

`var0` and `map_weight_decay` are **two copies of the same prior** on $w$ — see
[I10](../gotchas/prior-enters-map-fit.md) before changing either.

## 7. What it writes

`artifacts/` and `figures/`, per [0010](../decisions/0010-artifacts-and-figures-layout.md):

| file | contents |
|---|---|
| `config.json` | the `Config`, library versions, timestamp |
| `data.pt` | training data **and** the grid — the data is random, so a figure is not reproducible from the script alone |
| `feature_map.pt` | frozen feature-map `state_dict` |
| `param_store.pt` | all guide parameters (do not restore guides from it — [I9](../gotchas/param-store-reload.md)) |
| `laplace_posterior.pt` | loc / covariance / stddev / `w_map` |
| `mcmc_samples.pt` | posterior draws, pooled and per chain |
| `mcmc_diagnostics.json` | $\hat r$ and $n_{\text{eff}}$ per coordinate |
| `w_samples.pt` | posterior draws of $w$ per method — the reconstruction-free route |
| `predictions.pt` | per-method predictive fields, to re-plot without refitting |
| `losses.pt` | SVI loss traces |

Figures: `confidence_comparison.svg`, `epistemic_comparison.svg`,
`mcmc_pairplot.svg`.

**This layout is overwritten on every run** — one `artifacts/`, one `figures/`. The
sweep and the ablation exist partly to fix that, giving each run its own timestamped
folder ([0011](../decisions/0011-sweep-layout.md)).

## 8. Reported quantities

Per method: mean posterior spread (`std mean`), mean confidence, mean epistemic
uncertainty. Per grid point, `decompose_uncertainty` returns

$$\underbrace{H[\bar p]}_{\text{total}}, \qquad
\underbrace{\mathbb{E}_q\big[H[p]\big]}_{\text{aleatoric}}, \qquad
\underbrace{H[\bar p] - \mathbb{E}_q\big[H[p]\big]}_{\text{epistemic} \;=\; I[y; w \mid x]}$$

with $H$ the binary entropy in nats and $\bar p$ the posterior-mean probability. The
binary maximum is $\ln 2 = 0.693$. Confidence is $\max(\bar p, 1 - \bar p)$, a
property of $\bar p$ alone — it says nothing about posterior spread, which is exactly
why both are reported.

## 9. Running it

```bash
uv run python -u two_moons_comparison.py
```

No CLI: it runs one configuration, `Config()`. To change anything, either edit
`Config` or call `main(replace(Config(), var0=...))` from another script — which is
precisely what [`prior_scale_sweep.py`](prior-scale-sweep.md) does.

**Windows:** `mcmc_chains > 1` spawns processes, so any caller must sit behind
`if __name__ == "__main__"` or the chains silently collapse to one
([I7](../gotchas/windows-multiprocessing-mcmc.md)).

## Regenerating these counts

```python
from two_moons_comparison import Model, Config
cfg = Config()
m = Model(n=2, h=cfg.h, k=cfg.k)
print(sum(p.numel() for p in m.feature_map.parameters()),  # 560
      sum(p.numel() for p in m.clf.parameters()),           # 20
      sum(b.numel() for b in m.buffers()))                  # 82
```

## Symbols

| symbol | meaning |
|---|---|
| $N$, $x$, $y$ | training-set size; an input in $\mathbb{R}^2$; its label in $\{0,1\}$ |
| $\theta$, $\sigma$ | the angle generating a data point; the noise standard deviation |
| $\phi(x)$ | the frozen feature map, $\mathbb{R}^2 \to \mathbb{R}^{20}$ |
| $w$, $w_{\text{MAP}}$ | last-layer weights (the only Bayesian parameters); their MAP value |
| $f(x) = \phi(x)^\top w$ | the logit |
| $\mathrm{var}_0$, $s$ | prior variance; prior standard deviation $\sqrt{\mathrm{var}_0}$ |
| $\bar p$ | posterior-mean predicted probability at a grid point |
| $H$, nat | binary entropy; unit of log-probability in base $e$ |
| $I[y; w \mid x]$ | mutual information — the epistemic term |
| $\hat r$, $n_{\text{eff}}$ | MCMC diagnostics: split-$\hat R$, effective sample size |
