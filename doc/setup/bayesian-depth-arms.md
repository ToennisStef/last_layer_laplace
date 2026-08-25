# Bayesian-depth arms — what an "arm" is, and its parameter count

**Describes:** [`full_bnn_ablation.py`](../../full_bnn_ablation.py). Counts verified against the code 2026-08-25
(regeneration snippet at the end). This is a description of the apparatus, not a
finding — findings live in [methods/](../methods/) and [gotchas/](../gotchas/).

An **arm** is one rung of the ablation ladder: the *same* network, differing only in
**how many layers get a posterior instead of a point estimate**. The word is borrowed
from experiment design — one arm of a trial is one treatment group — and here the
treatment is *Bayesian depth*. [Q2](../open-questions.md#q2) is what the ladder
exists to answer.

---

## 1. The architecture, identical in every arm

```
  x ∈ R²
    │
    ├─ Linear₀ (2 → 20) ──► BatchNorm₀ ──► ReLU
    │
    ├─ Linear₁ (20 → 20) ─► BatchNorm₁ ──► ReLU
    │
    └─ Linear₂ (20 → 1) ──────────────────────────►  logit  f(x)
```

Symbols used below:

- $n = 2$ — input dimension; $h = 20$ — hidden width; $k = 1$ — output dimension.
- $w_i$, $b_i$ — weight matrix and bias vector of `Linear`$_i$, $i \in \{0, 1, 2\}$.
- $\theta$ — the vector of all **Bayesian** parameters in a given arm; $P = \dim\theta$
  is the *latent dimension*, which is what NUTS has to explore and what a dense
  curvature matrix is $P \times P$ in.
- **point-estimated** — fitted once in stage 1 and then frozen
  (`eval()` + `requires_grad_(False)`); a single number, no posterior.
- **Bayesian** — a `pyro.sample` site with a prior $\mathcal{N}(0, s^2)$ and a
  posterior.

## 2. The three arms

`bayes_depth` counts **trailing** `Linear` layers that are Bayesian:

```
                     Linear₀       Linear₁       Linear₂
                     (2→20)        (20→20)       (20→1)      latent dim P
  ──────────────────────────────────────────────────────────────────────
  ll        depth 1  point         point         BAYES              21
  last2     depth 2  point         BAYES         BAYES             441
  full      depth 3  BAYES         BAYES         BAYES             501
  ──────────────────────────────────────────────────────────────────────
  BatchNorm₀,₁       point         point           —       never Bayesian
```

- **`ll`** — the paper's last-layer setup: `Linear`$_0$ and `Linear`$_1$ act as a
  frozen feature map $\phi(x)$, and only the output layer is Bayesian.
- **`last2`** — the output layer plus the second hidden layer.
- **`full`** — every `Linear` layer; the full BNN.

## 3. Full parameter breakdown

Per-layer counts, with $w_i$ of shape $(\text{out}, \text{in})$:

| layer | shape | $w_i$ | $b_i$ | subtotal |
|---|---|---|---|---|
| `Linear`$_0$ | $(20, 2)$ | 40 | 20 | **60** |
| `Linear`$_1$ | $(20, 20)$ | 400 | 20 | **420** |
| `Linear`$_2$ | $(1, 20)$ | 20 | 1 | **21** |
| `BatchNorm`$_0$ + `BatchNorm`$_1$ | $\gamma, \beta$ per layer | 40 | 40 | **80** |

Total trainable parameters in the network: $501 + 80 = 581$, the same in every arm.
What changes is only how they are split:

| arm | Bayesian sites | latent $P$ | point-estimated | check |
|---|---|---|---|---|
| `ll` | $w_2, b_2$ | $20 + 1 = $ **21** | 480 Linear + 80 BN = **560** | $21 + 560 = 581$ ✔ |
| `last2` | $w_1, b_1, w_2, b_2$ | $400 + 20 + 20 + 1 = $ **441** | 60 Linear + 80 BN = **140** | $441 + 140 = 581$ ✔ |
| `full` | $w_0, b_0, w_1, b_1, w_2, b_2$ | $40 + 20 + 400 + 20 + 20 + 1 = $ **501** | 0 Linear + 80 BN = **80** | $501 + 80 = 581$ ✔ |

Plus **82 BatchNorm buffers** in every arm — the two layers' `running_mean` and
`running_var` (20 each) and their `num_batches_tracked` counters. Buffers are not
parameters: they are estimated by a running average during stage 1 and then frozen,
which is part of why a Bayesian BatchNorm is not a well-posed object (§5).

Note the jump: `ll` → `last2` multiplies the latent dimension by 21, while
`last2` → `full` adds only 14%. The 400-entry hidden-to-hidden weight matrix
dominates, so **`last2` is nearly as expensive as `full`** — worth knowing before
planning runs.

## 4. Why *trailing* layers

`bayes_depth` counts from the output backwards, so the point-estimated layers always
form a **prefix**. Under `Predictive(parallel=True)` and vectorised particles, sampled
weights carry leading sample dimensions; keeping them in the tail means only the tail
has to broadcast over those dimensions. A Bayesian layer sandwiched between
point-estimated ones would force every later layer to thread sample dimensions
through as well.

Consequence: "middle layer only" is not expressible as an arm. If that is ever
wanted, `bayes_depth` has to become a *set* of layer indices rather than a count.

## 5. BatchNorm is never Bayesian, in any arm

It is point-estimated in stage 1 and then frozen. Three reasons, in decreasing order
of severity:

1. **In train mode it breaks the likelihood.** Batch statistics couple observations,
   so `pyro.plate("data", ...)` would stop describing a product over data points, and
   the sampled posterior would belong to a different model than the one used at
   prediction time (which uses running statistics).
2. **$\gamma$ is non-identifiable** against the next layer's weights: ReLU is
   positively homogeneous, so scaling $\gamma$ up and the next weights down leaves the
   function unchanged. A prior on both creates a ridge for NUTS to explore at no
   function-space gain.
3. **The running statistics are buffers, not parameters** — there is no likelihood
   term that would make them Bayesian.

Frozen in eval mode, BatchNorm is an affine map with *fixed* constants, so all three
problems disappear: the likelihood factorises again and $\gamma$ is not a latent
variable. It is applied functionally after freezing
($h \mapsto (h - \mu)\,\gamma/\sqrt{\sigma^2 + \epsilon} + \beta$) because
`nn.BatchNorm1d` rejects the $(S, N, C)$ shapes that vectorised sampling produces.

`--norm none` drops BatchNorm entirely, but its arms are then not comparable to
BatchNorm arms at equal $\mathrm{var}_0$ — the prior lives in post-BN coordinates.

## 6. Arms are not methods

Each arm is fitted with the **whole set** of inference methods, so results form a
grid — arms × methods. The arm asks *how much of the network is Bayesian*; the method
asks *how do we approximate that posterior*. [Q2](../open-questions.md#q2) needs both
axes to separate "the disagreement is Laplace's fault" from "the disagreement is the
last-layer restriction's fault".

Which methods exist per arm is itself a result, not a configuration choice:

| method | `ll` | `last2` | `full` |
|---|---|---|---|
| `Laplace` (exact Hessian) | ✔ | ✗ not positive definite | ✗ not positive definite |
| `Laplace (GGN)` | ✔ | ✔ | ✔ |
| `Laplace (GGN, linearised)` | ✔ | ✔ | ✔ |
| `VI mean-field` | ✔ | ✔ | ✔ |
| `VI full-rank` | ✔ | ✔ | expensive: $P(P+1)/2 \approx 125\,000$ |
| `NUTS` | ✔ | ✔ | ✔ |

The exact Hessian exists only for `ll`, where the frozen features make the problem a
convex logistic regression — [I11](../gotchas/full-bnn-laplace-not-pd.md). That is why
the ladder carries a GGN Laplace as well
([0012](../decisions/0012-ggn-laplace-for-deep-arms.md)), and what each curvature
matrix is is [M7](../methods/curvature-approximations.md). A missing row is recorded
in `method_status` inside each arm's `artifacts/config.json`.

## 7. Running one

```bash
uv run python -u full_bnn_ablation.py --arms ll
uv run python -u full_bnn_ablation.py --arms last2
uv run python -u full_bnn_ablation.py --arms full --skip-full-rank
```

Output goes to `ablations/<UTC-stamp>_bayes_depth[_tag]/arms/arm<i>_<name>_d<P>/`, so
the folder name carries the arm and its latent dimension
([0011](../decisions/0011-sweep-layout.md) for the layout).

## Regenerating these counts

```python
from full_bnn_ablation import PartiallyBayesianMLP, ARMS, site_shapes
for name in ("ll", "last2", "full"):
    m = PartiallyBayesianMLP(n=2, h=20, k=1,
                             bayes_depth=ARMS[name].bayes_depth, norm="batchnorm")
    print(name, m.latent_dim, site_shapes(m),
          sum(p.numel() for p in m.deterministic_parameters()))
```

If `Config.h` changes, every number on this page changes with it — the counts above
are for $h = 20$.
