# M6 — The mode is not where the posterior mass is

**Source: measured** (2026-08-21, run 2: 5 chains x 500 draws, `r_hat` <= 1.004,
`n_eff` 1270-2122, 0 divergences). Numbers from `artifacts/`.

The single fact that explains the LLLA-vs-NUTS gap in
[M2](laplace-vs-vi-vs-mcmc.md). It is a property of high-dimensional geometry,
not of this dataset.

## Mass = density x volume

In `d` dimensions the volume of a shell at radius `r` grows like `r^(d-1)`, which
outruns the falloff in density. Samples therefore come not from the density peak
but from a thin shell where the product peaks — the **typical set**. For
`N(0, s^2 I_d)` it sits at radius `s*sqrt(d)`.

Here `d = 20`, prior `s = 44.72`:

| quantity | value |
|---|---|
| typical prior radius `s*sqrt(d)` | 200 |
| `\|\|w\|\|` of NUTS draws (median) | **198.3** |
| `\|\|w_map\|\|` | **17.1** |
| closest of 2000 draws to `w_map` | 81.8 |
| median distance to `w_map` | 188.2 |

The posterior's radius is indistinguishable from the prior's: **the data does not
constrain the magnitude of `w` at all**, only its direction. Same fact the Hessian
shows from the other side — 8 of 20 curvature directions sit within 2x of pure
prior curvature ([M2](laplace-vs-vi-vs-mcmc.md)).

## This is textbook, not pathology

```
excess of -log p over the mode, median draw : 9.72 nats
Gaussian prediction in d = 20,  d/2         : 10.0 nats
```

A 20-D Gaussian puts typical draws ~`d/2` nats above its own mode. Observed 9.72.
Nothing is broken; the intuition that the mode represents the distribution is what
fails.

`w_map` **is** the mode — its `-log p(w, D)` of 0.468 beats every one of 2000 NUTS
draws (best: 2.136). Being the mode and being representative are different things.

## Why typical draws fit the data better

|  | NLL (data fit) | prior term |
|---|---|---|
| at the mode | 0.394 | 0.073 |
| typical draw | **0.052** | **9.833** |

The data is separable, so larger `\|\|w\|\|` gives sharper, more correct logits and a
*lower* NLL. The mode is where that trade balances pointwise; the mass sits where
it balances after volume is accounted for.

## Two traps this creates

1. **Pair plots cannot show it.** `w_map` lies inside the central 90% marginal for
   **all 20 coordinates**, yet no draw is within 81 units of it. To be near the mode
   every coordinate must be small *simultaneously*: for a 20-D Gaussian,
   `P(all coords within 1 sd) = 0.68^20 = 4.5e-4`. A corner plot only ever shows
   1-D and 2-D shadows, and each shadow looks fine.
2. **The posterior mean is not typical either.** `\|\|mean of draws\|\| = 125.2` while
   `mean of \|\|draw\|\| = 198.2` — averaging vectors with differing directions shrinks
   the result inward, off the shell. Its `-log p` (3.917) is worse than the best
   single draw's (2.136). **No point estimate summarises this posterior**, which is
   why methods must be compared through the predictive, not through `loc`.

## Testable prediction

Shrinking the prior scale moves the typical radius `s*sqrt(d)` toward the mode, so
the Laplace-vs-NUTS gap should close — see [Q8](../open-questions.md#q8).

**Transferable?** yes — none of this is specific to two moons or to last-layer
Bayes. Any BNN posterior with tens of weights and a weakly-informative prior
behaves this way.
