# M6 — The mode is not where the posterior mass is

**Source: measured** (2026-08-21, run 2: 5 chains x 500 draws, $\hat r \le 1.004$,
$n_{\text{eff}}$ 1270–2122, 0 divergences). Numbers from `artifacts/`.

The single fact that explains the LLLA-vs-NUTS gap in
[M2](laplace-vs-vi-vs-mcmc.md). It is a property of high-dimensional geometry, not of
this dataset. Every symbol used below is listed in [Symbols](#symbols).

---

## 1. Mass = density × volume

Let

- $d$ — the number of weight dimensions (here $d = 20$, the last layer).
- $w$ — a weight vector, $\lVert w \rVert$ its Euclidean length ("radius").
- $s$ — the prior standard deviation per coordinate (here $s = 44.72$).
- $r$ — a radius, i.e. a candidate value of $\lVert w \rVert$.

For an isotropic Gaussian $\mathcal{N}(0, s^2 I_d)$, the *density* at radius $r$ falls
off like $e^{-r^2/2s^2}$ — but the *volume* of the thin shell at that radius grows
like $r^{d-1}$. What you actually sample is the product:

$$\underbrace{P(\lVert w \rVert \approx r)}_{\text{how much mass at radius } r}
\;\propto\;
\underbrace{e^{-r^2 / 2s^2}}_{\text{density: peaks at } r=0}
\;\times\;
\underbrace{r^{\,d-1}}_{\text{shell volume: grows}}$$

Setting the derivative of the log of that product to zero,

$$\frac{d}{dr}\left[-\frac{r^2}{2s^2} + (d-1)\log r\right] = -\frac{r}{s^2} + \frac{d-1}{r} = 0
\qquad\Longrightarrow\qquad
r^\star = s\sqrt{d-1} \;\approx\; s\sqrt{d}$$

so the mass concentrates in a thin shell at radius $r^\star$ — the **typical set** —
and *not* at the density peak $r = 0$. Drawn as profiles against $r$:

```
     density  e^(-r²/2s²)        shell volume  r^(d-1)        product = where mass is
  ┤█                          ┤                             ┤
  │ ██                        │                     ▂▄▆█    │            ▁█▖
  │   ███                     │              ▁▂▄▆█▀▀        │           ▄▀ ▝█▖
  │      ████▄▄               │       ▁▂▄▆█▀▀               │          ▄▘   ▝█▄
  ┤          ▀▀▀▀▀▀▀▄▄▄▄▄▄▄   ┤▁▂▄▆█▀▀                      ┤▁▁▁▁▁▁▁▁▄▀▘      ▀▀▄▄▄
  └─┬────────────────────►    └──────────────────────►      └──────────┬──────────►
    0            r                          r                  0      s√d       r
    ▲                                                                  ▲
    the mode lives here                                    every single draw lives here
```

For $d = 20$, $s = 44.72$: $s\sqrt{d-1} = 194.9$ and $s\sqrt{d} = 200.0$.

| quantity | value |
|---|---|
| typical prior radius, $s\sqrt{d}$ | 200.0 |
| median $\lVert w \rVert$ of NUTS draws | **198.3** |
| $\lVert w_{\text{MAP}} \rVert$ | **17.1** |
| closest of 2000 draws to $w_{\text{MAP}}$ | 81.8 |
| median distance to $w_{\text{MAP}}$ | 188.2 |

The posterior radius (198.3) is indistinguishable from the prior's (194.9–200.0):
**the data does not constrain the magnitude of $w$ at all**, only its direction. The
Hessian says the same thing from the other side — 8 of 20 curvature directions sit
within $2\times$ of pure prior curvature ([M2](laplace-vs-vi-vs-mcmc.md)).

---

## 2. This is textbook, not pathology

For a $d$-dimensional Gaussian, a typical draw's negative log density sits about
$d/2$ **nats** above the value at the mode (a nat is a unit of log-probability in
base $e$):

| | value |
|---|---|
| excess of $-\log p$ over the mode, median draw | 9.72 nats |
| Gaussian prediction, $d/2$ with $d = 20$ | 10.0 nats |

Nothing is broken. What fails is the intuition that the mode represents the
distribution. And $w_{\text{MAP}}$ genuinely *is* the mode — its
$-\log p(w, \mathcal{D})$ of 0.468 beats every one of 2000 NUTS draws (best: 2.136).
**Being the mode and being representative are different properties.**

### Why typical draws fit the data *better*

Split the negative log posterior into its two parts,

$$-\log p(w \mid \mathcal{D}) \;=\;
\underbrace{\text{NLL}}_{\text{data fit}} \;+\;
\underbrace{\tfrac{1}{2s^2}\lVert w \rVert^2}_{\text{prior term}} \;+\; \text{const}$$

and evaluate each at the mode and at a typical draw:

| | NLL (data fit) | prior term |
|---|---|---|
| at the mode | 0.394 | 0.073 |
| typical draw | **0.052** | **9.833** |

The data is separable, so a larger $\lVert w \rVert$ gives sharper, more confident, *correct*
logits and therefore a **lower** NLL. The mode is where that trade-off balances
pointwise; the mass sits where it balances *after volume is accounted for*.

---

## 3. Two traps this creates

### Trap 1 — pair plots cannot show it

Every one-dimensional shadow looks healthy while the joint picture does not:

```
  per-coordinate view (what a corner plot shows)    joint view (what matters)
  ─────────────────────────────────────────────    ────────────────────────────
   coord  1 : w_map inside central 90%   ✓           ||w_map||             =  17.1
   coord  2 : w_map inside central 90%   ✓           nearest of 2000 draws =  81.8
      ⋮                    ⋮             ⋮           median distance       = 188.2
   coord 20 : w_map inside central 90%   ✓
  ─────────────────────────────────────────────    ────────────────────────────
   20 of 20 marginals look fine                     no draw is anywhere near it
```

Both columns are true at once. To be *near* the mode, every coordinate has to be
small **simultaneously**, and that conjunction is vanishingly rare:

$$P(\text{all } d \text{ coordinates within } 1\,\text{sd}) = 0.68^{20} = 4.5\times10^{-4}$$

A corner plot only ever shows 1-D and 2-D shadows, and each shadow looks fine.

### Trap 2 — the posterior mean is not typical either

| quantity | value |
|---|---|
| $\lVert \overline{w} \rVert$ — length of the mean of the draws | 125.2 |
| $\overline{\lVert w \rVert}$ — mean of the lengths of the draws | 198.2 |

Averaging vectors that point in different directions shrinks the result inward, off
the shell. The mean's $-\log p$ (3.917) is worse than the best single draw's (2.136).
**No point estimate summarises this posterior** — which is why methods here must be
compared through the *predictive*, never through a guide's `loc`
([decision 0009](../decisions/0009-uniform-sampled-predictive.md)).

---

## 4. Testable prediction — made, then confirmed

Shrinking $s$ moves the typical radius $s\sqrt{d}$ toward the mode, so the
Laplace-vs-NUTS gap should close — [Q8](../open-questions.md#q8).

**Measured 2026-08-24** (8 prior scales, everything else fixed): the disagreement
$\text{mean}\,\lvert \bar p - \bar p_{\text{NUTS}} \rvert$ falls monotonically from
0.2542 at $s = 100$ to 0.0030 at $s = 0.1$. The mechanism in this note made a
quantitative prediction that could have failed and did not — see
[prior-scale-calibration](prior-scale-calibration.md#measured-sweep).

The sweep also re-tests the shell radius itself at four scales, not one. If the
draws really sit at $r^\star = s\sqrt{d-1}$, then the median radius divided by the
prior scale must be constant at $\sqrt{19} = 4.36$:

| $s$ | 10 | 31.6 | 44.7 | 100 |
|---|---|---|---|---|
| median $\lVert w \rVert$ (NUTS) | 45.16 | 139.60 | 197.77 | 438.20 |
| $\div\, s$ | **4.52** | **4.41** | **4.42** | **4.38** |

against the predicted 4.36 — converging toward it as $s$ grows and the prior
dominates. Below $s = 10$ the ratio rises (up to 7.5 at $s = 0.1$) because there the
*likelihood* still constrains the magnitude and the posterior is no longer
prior-shaped.

---

## Symbols

| symbol | meaning |
|---|---|
| $d$ | number of weight dimensions (here 20) |
| $w$, $\lVert w \rVert$ | weight vector; its Euclidean length ("radius") |
| $w_{\text{MAP}}$ | the trained weights — the posterior mode |
| $s$ | prior standard deviation per coordinate (here 44.72) |
| $r$, $r^\star$ | a radius; the radius where posterior mass concentrates |
| $\mathcal{N}(0, s^2 I_d)$ | isotropic Gaussian, zero mean, variance $s^2$ per coordinate |
| $\mathcal{D}$ | the training data |
| NLL | negative log likelihood — the data-fit term |
| nat | unit of log-probability in base $e$ |
| $\bar p$ | posterior-mean predicted probability at a grid point |
| $\overline{w}$, $\overline{\lVert w \rVert}$ | mean of the draws; mean of the draws' lengths |
| $\hat r$, $n_{\text{eff}}$ | MCMC convergence diagnostics: split-$\hat R$ and effective sample size |

**Transferable?** yes — none of this is specific to two moons or to last-layer Bayes.
Any BNN posterior with tens of weights and a weakly-informative prior behaves this
way.
