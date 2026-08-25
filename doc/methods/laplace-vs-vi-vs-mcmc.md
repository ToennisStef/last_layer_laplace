# M2/M3 — LLLA vs. VI vs. MCMC on two moons

**Source: measured.** Rewritten 2026-08-21 after
[two_moons_comparison.py](../../two_moons_comparison.py) replaced the notebook run.
Same model, same frozen feature map, same prior $\mathcal{N}(0, s^2)$ with
$s = 44.72$ on the last-layer weights $w \in \mathbb{R}^{20}$. Artifacts in
`artifacts/`. Symbols are listed in [Symbols](#symbols).

> **This note previously concluded the opposite.** The earlier version had MCMC
> siding with VI and inferred that Laplace was "bloated". That run seeded NUTS at
> $w_{\text{MAP}}$; with `init_to_sample()` and converged chains the ordering
> changes. Kept visible on purpose — the retraction is the finding. See
> [learning-log](../learning-log.md).

---

## 1. Result

Run 2: 5 chains × 500 draws, $\hat r \le 1.0036$, $n_{\text{eff}}$ 1270–2122,
0 divergences.

| Inference | $\lVert \mathbb{E}[w] \rVert$ | mean sd | ratio $\lVert \mathbb{E}[w] \rVert / \text{sd}$ | mean conf | mean epistemic (nats) |
|---|---|---|---|---|---|
| Laplace (`AutoLaplaceApproximation`) | 17.1 | 27.7 | 0.13 | **0.616** | **0.637** |
| VI mean-field (`AutoNormal`) | 57.0 | 1.8 | 4.47 | 0.919 | 0.175 |
| VI full-rank (`AutoMultivariateNormal`) | 56.0 | 7.8 | 0.90 | 0.910 | 0.202 |
| **NUTS (reference)** | 125.2 | 34.7 | 0.70 | **0.845** | **0.328** |

where

- $\lVert \mathbb{E}[w] \rVert$ — length of the posterior mean weight vector.
- mean sd — the posterior standard deviation per coordinate, averaged over the 20
  coordinates.
- mean conf — grid-average of $\max(\bar p, 1 - \bar p)$, with $\bar p$ the
  posterior-mean predicted probability at a grid point.
- epistemic — mutual information $I[y; w \mid x]$, in nats. Its maximum possible
  value is $\ln 2 = 0.693$.

**NUTS sits between Laplace and VI.** Laplace is the *least* confident, VI the most.
Laplace and NUTS reproduce to three decimals across two independent runs (4 vs 5
chains); the VI numbers move (mean-field sd $3.03 \to 1.78$ once the VI optimiser got
learning-rate decay) but their *ordering* does not.

All four are scored with the identical sampled predictive
([decision 0009](../decisions/0009-uniform-sampled-predictive.md)), so the
differences are posterior differences, not estimator differences.

---

## 2. Why Laplace ends up least confident

Two effects compound, and neither is "the approximation is bad".

### 2.1 The covariance is mostly the prior

The Laplace covariance is the inverse Hessian of the negative log posterior at the
mode:

$$\Sigma = H^{-1}, \qquad
H = \underbrace{\Phi^\top S\, \Phi}_{\text{likelihood}} + \underbrace{\tfrac{1}{\mathrm{var}_0} I}_{\text{prior}}$$

- $\Phi$ — the $N \times 20$ matrix of frozen features, row $n$ being $\phi(x_n)^\top$.
- $S$ — diagonal $N \times N$ matrix with entries $\sigma_n(1 - \sigma_n)$, where
  $\sigma_n$ is the predicted probability at $x_n$.
- $\mathrm{var}_0 = s^2 = 2000$ — the prior variance; $1/\mathrm{var}_0$ is the prior
  precision.

The network **separates** the data, so every $\sigma_n \to 0$ or $1$, hence
$\sigma_n(1-\sigma_n) \to 0$ and the likelihood term collapses:

| quantity | value |
|---|---|
| smallest eigenvalue of $H$ | $5.049 \times 10^{-4}$ |
| largest eigenvalue of $H$ | $1.401$ |
| prior's contribution $1/\mathrm{var}_0$ to *every* eigenvalue | $5.000 \times 10^{-4}$ |
| directions within $2\times$ of pure prior curvature | **8 of 20** |

In the flattest direction the likelihood supplies about 1% of the curvature. Laplace's
mean sd of 27.7 is **62% of the prior scale** — it largely hands the prior back.

### 2.2 The mode is deep inside the typical set

$\lVert w_{\text{MAP}} \rVert = 17.1$ against a sampled shell at
$\lVert w \rVert \approx 198$. Full account in [M6](mode-vs-typical-set.md).

### 2.3 The two combine as a ratio

Confidence is driven by the predictive logit's mean divided by its spread — in the
probit form, $m(x) \big/ \sqrt{1 + \pi v(x)/8}$, with $m(x)$ the mean logit and
$v(x)$ its variance. **It is the ratio that matters, not the spread alone.** Using
the weight-space numbers as a proxy:

```
                centre ‖E[w]‖              spread (mean sd)        ratio
  Laplace       17.1                       27.7                    0.13
                0    17
                ├────●──────────────┤      centre buried inside the spread
                                           → averaged sigmoids collapse toward ½

  NUTS          125.2                      34.7                    0.70
                0                125
                ├─────────────────●─────┤  centre well outside the spread
                                           → sigmoids stay saturated
```

Laplace has a small centre and a prior-sized spread; NUTS has a *larger* spread yet
is more confident, because its centre is $7\times$ further out. **Ranking methods by
covariance magnitude alone will mislead you.**

---

## 3. Why full-rank VI does not recover Laplace

It lands on mean-field instead, so the mean-field independence assumption is *not*
what separates VI from Laplace. The two optimise different things:

- **Laplace** takes local curvature at the mode and minimises no divergence at all.
- **VI** minimises $\mathrm{KL}(q \,\Vert\, p)$ over the whole family, where $q$ is the
  approximating Gaussian and $p$ the true posterior. A $q$ as wide as the prior would
  make $\mathbb{E}_q[\log p(y \mid w)]$ terrible, so the ELBO shrinks it — and shrinks
  it further the longer it converges (learning-rate decay tightened both guides
  without moving confidence).

The textbook expectation that a full-covariance Gaussian recovers Laplace holds only
when the posterior is Gaussian *in shape near the mode and dominated by it*. Here it
is a shell.

---

## 4. Current reading

- **Laplace is underconfident here**, but "bloated covariance" is the wrong
  diagnosis: the covariance is roughly the prior, and the centre is unrepresentative.
- **VI is overconfident**, and converging it further makes that worse, not better.
- **NUTS is the only one describing the actual posterior**, and it is well converged
  by every available diagnostic.
- None of this says Laplace is a bad method in general. It says that on a
  **separable** problem with a **weakly-informative** prior the Laplace assumptions
  fail in a specific, predictable way.

## 5. Still open

- No calibration metric has been computed — everything here is confidence and entropy
  on a grid, not ECE/NLL/OOD ([Q3](../open-questions.md#q3),
  [M4](../lessons-methodology.md)).
- ~~The prediction that a tighter prior closes the gap is untested~~ — **answered
  2026-08-24**, it does: see
  [prior-scale-calibration](prior-scale-calibration.md#measured-sweep) and
  [Q8](../open-questions.md#q8).
- Whether any of this transfers beyond last-layer Bayes ([Q2](../open-questions.md#q2)).

---

## Symbols

| symbol | meaning |
|---|---|
| $w$, $d$ | last-layer weight vector; its dimension (20 here) |
| $w_{\text{MAP}}$ | trained weights = posterior mode |
| $s$, $\mathrm{var}_0 = s^2$ | prior standard deviation (44.72) and variance (2000) |
| $\Sigma$, $H$ | posterior covariance; Hessian of the negative log posterior, $\Sigma = H^{-1}$ |
| $\phi(x)$, $\Phi$ | frozen feature map; the $N \times d$ matrix of features over the training set |
| $S$ | diagonal matrix of $\sigma_n(1-\sigma_n)$ — the likelihood's curvature per data point |
| $\sigma_n$ | predicted probability at training point $n$ |
| $m(x)$, $v(x)$ | mean and variance of the predictive logit at input $x$ |
| $\bar p$ | posterior-mean predicted probability at a grid point |
| $q$, $p$ | the variational approximation and the true posterior |
| $\mathrm{KL}(q \,\Vert\, p)$ | Kullback–Leibler divergence, what VI minimises |
| $I[y; w \mid x]$ | mutual information — the epistemic part of predictive uncertainty |
| nat | unit of log-probability in base $e$; $\ln 2 = 0.693$ is the binary maximum |
| $\hat r$, $n_{\text{eff}}$ | MCMC diagnostics: split-$\hat R$ and effective sample size |
