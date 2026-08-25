# M1 — Prior scale sets LLLA calibration

**Source: measured** (2026-08-18/19, two moons, `h = 20`, last layer only).

With a prior $w \sim \mathcal{N}(0, s^2 I)$ on the last-layer weights $w$:

| prior scale $s$ | effect on the confidence map |
|---|---|
| too small | posterior covariance is prior-dominated and tight → **overconfident** far from data |
| too large | the predictive logit variance $v(x) = \phi(x)^\top \Sigma \phi(x)$ blows up, the probit shrinks logits → **underconfident** everywhere |

where $\phi(x)$ is the frozen feature vector at input $x$, $\Sigma$ the posterior
covariance over $w$, and $I$ the identity matrix.

Current setting:

$$\mathrm{var}_0 = \frac{1}{\lambda} = \frac{1}{5 \times 10^{-4}} = 2000,
\qquad s = \sqrt{\mathrm{var}_0} \approx 44.7$$

with $\lambda$ the MAP baseline's weight decay and $\mathrm{var}_0$ the prior
variance ([I1](../gotchas/prior-scale-units.md)). That match is a *consistency*
argument, not a calibration argument — the weight decay was itself inherited from the
paper's code, not tuned for calibration.

## Measured sweep

**Source: measured** (2026-08-24, `prior_scale_sweep.py`, all 8 prior scales;
`sweeps/2026-08-24T0852_prior_scale_q8/`). Everything but $\mathrm{var}_0$ held
fixed — same seed, same data, `map_weight_decay` pinned at
$\lambda = 5 \times 10^{-4}$.

| $s$ | train acc | $\lVert w_{\text{MAP}} \rVert$ | Laplace conf | NUTS conf | $\text{mean}\,\lvert \bar p - \bar p_{\text{NUTS}} \rvert$ | Laplace $w$ sd | NUTS $w$ sd | max $\hat r$ |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.865 | 0.62 | 0.861 | 0.861 | **0.0030** | 0.095 | 0.094 | 1.0000 |
| 0.316 | 0.975 | 1.88 | 0.935 | 0.936 | 0.0042 | 0.282 | 0.280 | 1.0008 |
| 1 | 0.995 | 4.64 | 0.916 | 0.926 | 0.0123 | 0.826 | 0.830 | 1.0031 |
| 3.16 | 1.000 | 8.86 | 0.836 | 0.879 | 0.0481 | 2.476 | 2.508 | 1.0039 |
| 10 | 1.000 | 14.29 | 0.723 | 0.854 | 0.1346 | 7.600 | 7.744 | 1.0025 |
| 31.6 | 1.000 | 16.94 | 0.630 | 0.849 | 0.2202 | 21.085 | 24.542 | 1.0022 |
| 44.7 (current) | 1.000 | 17.14 | 0.616 | 0.845 | 0.2300 | 27.747 | 34.716 | 1.0036 |
| 100 | 1.000 | 17.30 | 0.594 | 0.845 | **0.2542** | 50.025 | 77.778 | **1.0122** |

$\bar p$ is the posterior-mean predicted probability at a grid point, so the sixth
column is the grid-average absolute disagreement with the NUTS reference. The "$w$ sd"
columns are the posterior standard deviation per weight coordinate, averaged over the
20 coordinates. The $s = 100$ row is the only one with $\hat r > 1.01$ — it is the
hardest geometry to sample (typical radius $s\sqrt{d} \approx 447$) and should be
treated as the weakest row of the eight.

Four things this settles or sharpens:

1. **[Q8](../open-questions.md#q8) is confirmed.** The Laplace/NUTS disagreement is
   monotone in the prior scale and spans two orders of magnitude. At $s = 0.1$ the
   Gaussian approximation is essentially exact — the two mean confidences agree to
   three decimals — which is what [M6](mode-vs-typical-set.md) predicts once the
   typical set collapses onto the mode.
2. **The two tightest rows are training failures and must not be read as
   calibration wins.** At $s = 0.1$ the network misclassifies 13.5% of its own
   training set (accuracy 0.865) yet is *more* confident in the far field (0.897)
   than over the grid as a whole (0.861) — an underfitted model that is confident
   away from the data, which is the paper's own failure mode reached from the
   opposite direction. The cause is the declared prior alone: precision
   $1/\mathrm{var}_0 = 100$ confines the weights to about $\pm 0.1$, so the logits
   $\phi(x)^\top w$ cannot reach the magnitudes the data needs.

   Laplace matching NUTS to three decimals there is a statement about the
   *approximation* — a prior-dominated posterior really is nearly Gaussian — and not
   about the model.

   **The finding does not depend on those rows.** Restricted to $s \ge 3.16$, where
   train accuracy is 1.000 throughout, the gap still moves 0.0481 → 0.2542.
3. **The weight decay pins the Laplace *centre*, not the posterior width — and the
   gap does not saturate.** Two different things scale differently:

   | quantity | behaviour as $s$ grows | evidence |
   |---|---|---|
   | $\lVert w_{\text{MAP}} \rVert$ — the Laplace centre | **plateaus** at ~17 | 14.29 → 16.94 → 17.14 → 17.30 |
   | posterior width | **grows $\propto s$**, uncapped | NUTS sd $/\,s$ = 0.774, 0.776, 0.776, 0.778 |

   The weight decay $\lambda$ enters **only stage 1**, where precisions add
   ($1/\mathrm{var}_0 + \lambda \to \lambda$), so it caps where the MAP lands. The
   Laplace Hessian, the VI objective and the NUTS target all use the model prior
   alone, so nothing caps the *width* — see
   [I10](../gotchas/prior-enters-map-fit.md).

   This is exactly the ratio mechanism of [M2](laplace-vs-vi-vs-mcmc.md#23-the-two-combine-as-a-ratio):

   - **Laplace** — centre pinned, width $\propto s$ ⟹ ratio $\to 0$ ⟹ confidence
     $\to 0.5$, so the gap keeps growing (0.2202 → 0.2300 → **0.2542**).
   - **NUTS** — centre *and* width both $\propto s$ ⟹ ratio constant ⟹ confidence
     frozen at 0.845 from $s = 10$ upward.

   > **Retracted.** An earlier version of this note predicted the top of the range
   > would saturate, on the grounds that $\lambda$ caps the effective prior variance
   > at $1/\lambda = 2000$. The $s = 100$ run falsified it: the gap grew and the
   > widths nearly doubled. The cap is real but applies to the MAP location only.
4. **VI does not follow the same curve.** Mean-field confidence sits at ~0.92 from
   $s = 3.16$ upward regardless of the prior, and its $w$ sd stays near 1.8 while
   Laplace's reaches 27.7. VI's failure is not a prior-scale failure.

The direction of the table also matches the "too small / too large" summary above:
confidence rises to ~0.94 near $s = 0.3$ and falls away in both directions.

Consequences:

- Any comparison of LLLA against VI or MCMC is only meaningful **at the same prior**
  ([M2](laplace-vs-vi-vs-mcmc.md)); the prior is shared, so a prior-driven
  disagreement would show up in all three.
- The knob is a hyperparameter that ought to be *fit*, not guessed. Ritter et al.
  (2018) tune it by marginal likelihood / validation — **not yet tried**,
  [Q1](../open-questions.md#q1).

Do not fan-in scale the prior — [I1](../gotchas/prior-scale-units.md).
