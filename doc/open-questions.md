# Open questions

Every entry follows the same shape:

| field | meaning |
|---|---|
| **Status** | one line: `OPEN` / `IN PROGRESS` / `ANSWERED` / `DROPPED`, with a date once it moves |
| **Question** | the research question, stated explicitly as a question — the title alone is only a label |
| **Why it matters** | what hangs on the answer |
| **Answer** | present only once answered: the result, with numbers, self-contained |
| **Learnings** | cross-references to the lessons pulled out of the answer (gotcha / method / decision) |
| **Next step** | what would actually move it, while it is still open |

An `OPEN` entry is **not evidence** — it is a plan. An `ANSWERED` entry carries its
result here *and* points at the durable lesson in
[lessons-implementation](lessons-implementation.md),
[lessons-methodology](lessons-methodology.md) or [decisions/](decisions/).

## At a glance

| # | Question | Status |
|---|---|---|
| [Q1](#q1) | Does tuning the prior precision fix calibration? | OPEN |
| [Q2](#q2) | Is the gap the Laplace approximation, or the last-layer restriction? | IN PROGRESS |
| [Q3](#q3) | Which posterior is better calibrated under a real metric? | OPEN |
| [Q4](#q4) | Are all methods scored with the same estimator? | **ANSWERED** 2026-08-21 |
| [Q5](#q5) | Are the NUTS chains converged enough to be ground truth? | **ANSWERED** 2026-08-21 |
| [Q6](#q6) | Is our Laplace a method property or a port bug? | OPEN |
| [Q7](#q7) | Does a bounded 2-D grid test the paper's claim at all? | OPEN |
| [Q8](#q8) | Does a tighter prior close the Laplace/NUTS gap? | **ANSWERED** 2026-08-24 |

---

<a id="q1"></a>

## Q1 — Tuning the prior precision

**Status:** OPEN — paper not read, not implemented.

**Question:** Does tuning the prior precision $\tau = 1/\mathrm{var}_0$ by marginal
likelihood or validation NLL, as Ritter et al. (2018) do, produce a *calibrated*
last-layer Laplace posterior — or does it merely re-fit an already misspecified
Gaussian to the wrong shape?

**Why it matters:** [Q8](#q8) showed
the prior scale moves the Laplace-vs-NUTS disagreement across the entire range
0.0030–0.2542, so "which value of $s$" is no longer a detail — it is the dominant
free choice in the whole comparison
([M1](methods/prior-scale-calibration.md)). Current practice here is to inherit
$s$ from the MAP baseline's weight decay
([0003](decisions/0003-prior-scale-from-weight-decay.md)), which is a *consistency*
argument and not a calibration one.

**Next step:** read the paper, implement the tuning, re-run the comparison at the
tuned scale, and check it against the swept curve already measured. Note the
distinction that Q8 exposed: the weight decay caps the MAP location but not the
posterior width ([I10](gotchas/prior-enters-map-fit.md)), so "tuning the prior" and
"tuning the weight decay" are not the same experiment.

**Related:** [M1](methods/prior-scale-calibration.md),
[M2](methods/laplace-vs-vi-vs-mcmc.md) ·
Ref: <https://openreview.net/forum?id=Skdvd2xAZ> ·
`laplace-torch` implements `optimize_prior_precision`:
<https://aleximmer.github.io/Laplace/>

---

<a id="q2"></a>

## Q2 — Full BNN vs last-layer only

**Status:** IN PROGRESS (2026-08-24) — `full_bnn_ablation.py` written and
smoke-tested; no full-settings run yet.

**Question:** Is the Laplace-vs-NUTS disagreement a property of the **Laplace
approximation itself**, or of **restricting Bayesian treatment to the last layer**?

**Why it matters:** it decides how far [M2](methods/laplace-vs-vi-vs-mcmc.md)
generalises. If the gap survives when every layer is Bayesian, it is a statement
about Gaussian approximations; if it disappears, it is a statement about last-layer
Bayes and the paper's headline method specifically.

**How it is being tested:** one architecture, three rungs of *Bayesian depth* — `ll`
(output layer only, 21 latent dimensions), `last2` (441), `full` (501) — each fitted
with the same set of inference methods, so only the depth varies.

**Partial findings already banked** (each cost a design decision):

1. **Exact-Hessian Laplace does not exist beyond the last layer.** 79 of 500
   eigenvalues are negative, so the Cholesky refuses — and the notebook's
   `torch.inverse` route silently returns 60 negative variances instead of failing.
   → [I11](gotchas/full-bnn-laplace-not-pd.md)
2. **So the ladder carries a GGN Laplace**, with both a sampled and a linearised
   predictive, validated against the exact Hessian on the `ll` arm (agreement to
   $1.2\times10^{-10}$ in eigenvalues). → [0012](decisions/0012-ggn-laplace-for-deep-arms.md)
3. **The authors never inverted a full exact Hessian either** — their deep variants
   are diagonal MC Fisher, Kronecker MC Fisher, or dense GGN.
   → [M7](methods/curvature-approximations.md)
4. **BatchNorm stays in but is never Bayesian** — frozen in eval mode it is an affine
   map with fixed constants, so the likelihood still factorises over data points; a
   *Bayesian* BN would break that and add a non-identifiable ridge. Rationale in the
   `full_bnn_ablation.py` module docstring.

**Next step:** run the ladder at full settings, one arm at a time
(`--arms ll`, then `last2`, then `full --skip-full-rank`). Then the open sub-question:
weight-space $\hat r$ is meaningless for the deep arms (permutation and sign
symmetries), so a function-space convergence diagnostic is needed before the `full`
row can be trusted.

**Related:** [M2](methods/laplace-vs-vi-vs-mcmc.md),
[M3](methods/laplace-vs-vi-vs-mcmc.md)

---

<a id="q3"></a>

## Q3 — Real calibration metrics

**Status:** OPEN — not computed. Everything so far is judged by eye from confidence
maps.

**Question:** Under a proper metric — held-out NLL, ECE / reliability diagrams, and
an OOD detection measure such as AUROC against a far-field or second-dataset control
— which of the four posteriors is actually best calibrated, and does the reading
"Laplace underconfident, VI overconfident" survive?

**Why it matters:** without a metric, [M2](methods/laplace-vs-vi-vs-mcmc.md) and
[M4](lessons-methodology.md) are qualitative, and "over/underconfident" is an
impression rather than a result. It also gates the poster claim: a confidence map is
a picture, not a measurement.

**Next step:** implement held-out NLL and ECE first (cheap, no new data needed), then
an OOD split. Hendrycks & Gimpel (2017), the standard baseline for the AUROC-style
measures, is in `literature/`.

**Related:** [M4](lessons-methodology.md),
[M5](methods/reproduction-scope.md), [Q7](#q7)

---

<a id="q4"></a>

## Q4 — One predictive estimator for every method

**Status:** **ANSWERED** (2026-08-21).

**Question:** Are the four methods scored through the *same* predictive estimator, so
that differences between their confidence maps are differences in the posterior
rather than in how logits were turned into probabilities?

**Why it mattered:** the paper scores Laplace with MacKay's probit approximation,
which deliberately shrinks logits toward zero, while VI and MCMC were being scored by
sampling. Any part of the confidence gap could have been the estimator.

**Answer:** Yes, now. `two_moons_comparison.py` scores all four methods by sampling
weights from the posterior and averaging sigmoids, so the maps differ only through
the posterior. The paper's `Model.predict` keeps the probit form, used only for
reproducing the original figure.

**Learnings:** [0009](decisions/0009-uniform-sampled-predictive.md) (the decision and
its consequences) · [0007](decisions/0007-probit-predictive.md) (the superseded probit
choice, kept for the reproduction) ·
[M2](methods/laplace-vs-vi-vs-mcmc.md) now states its numbers under one estimator.

---

<a id="q5"></a>

## Q5 — Is NUTS trustworthy as ground truth?

**Status:** **ANSWERED** (2026-08-21) — with one later caveat, below.

**Question:** Are the NUTS chains converged well enough for NUTS to serve as the
reference posterior the other three methods are judged against?

**Why it mattered:** the entire comparison is anchored on NUTS. An earlier run seeded
the chains at $w_{\text{MAP}}$, which biases them toward reporting a narrow posterior
simply because they never leave the mode — and that produced a conclusion that later
had to be retracted.

**Answer:** Yes, at the default prior. Seeded with `init_to_sample()` rather than the
MAP, 5 chains × 500 draws give $\hat r \le 1.0036$, $n_{\text{eff}}$ 1270–2122 of
2500, and **0 divergences**; the numbers reproduce across an independent 4-chain run.
Diagnostics are written to `artifacts/mcmc_diagnostics.json` on every run.

Consequence: [M2](methods/laplace-vs-vi-vs-mcmc.md) can be trusted, and the *old* M3
conclusion — derived from `w_map`-seeded chains — is retracted and kept visible.

**Caveat added 2026-08-24:** convergence is not automatic at every prior scale. In
the Q8 sweep the loosest prior ($s = 100$) reached $\hat r = 1.0122$, the only run
above 1.01, because its typical set sits at radius $s\sqrt{d} \approx 447$ and needs
long trajectories. Convergence must be re-checked per configuration, not assumed from
this one audit.

**Learnings:** [M3 retraction](methods/laplace-vs-vi-vs-mcmc.md) ·
[0006](decisions/0006-nuts-as-ground-truth.md) ·
[I6](gotchas/mcmc-setup.md) (chains, seeding, diagnostics to check) ·
[I7](gotchas/windows-multiprocessing-mcmc.md) (Windows spawn needs a `__main__`
guard or chains silently collapse to one)

---

<a id="q6"></a>

## Q6 — `laplace-torch` as an independent cross-check

**Status:** OPEN — not used.

**Question:** Does an independent, well-tested Laplace implementation reproduce our
numbers on the same trained network — i.e. is "Laplace is underconfident here" a
property of the *method*, or a bug in our port?

**Why it matters:** every conclusion in [M2](methods/laplace-vs-vi-vs-mcmc.md) rests
on one hand-written Hessian and one hand-written predictive. A second implementation
is the cheapest available falsification test, and the repo's own README recommends
using that library instead of this code.

**Next step:** run it from a **separate script**, so this repo's scripts stay
dependency-free. [M7 §7](methods/curvature-approximations.md) records the
configuration mapping: our `Laplace (GGN)` corresponds to *all weights + full
structure + GGN*, and the paper's headline method to *last layer + Kronecker*.

**Related:** [M7](methods/curvature-approximations.md),
[I11](gotchas/full-bnn-laplace-not-pd.md) ·
<https://github.com/AlexImmer/Laplace>

---

<a id="q7"></a>

## Q7 — Is the ReLU framing testable on this grid?

**Status:** OPEN — open thought, nothing run.

**Question:** The paper's result is *asymptotic* — MAP confidence tends to certainty
as $\lVert x \rVert \to \infty$, and any Gaussian last-layer posterior bounds that
limit away from 1. Does a bounded 2-D grid test that claim at all, and if not, what
would?

**Why it matters:** the test grid here is $[-5, 5]^2$, so
$\lVert x \rVert \le 7.07$ — close to the data by the standards of an argument about
$\lVert x \rVert \to \infty$. Confidence differences measured on that grid may be
outside what the theorem speaks to, which would make a matching figure evidence about
the *implementation* rather than about the claim
([M5](methods/reproduction-scope.md)).

**Next step:** the cheapest decisive version is an explicit far-field sweep —
confidence along a ray at $\lVert x \rVert = 10, 100, 1000$ — which is what the
theorem is actually about. Alternatives: a GP-limit comparison, or deep ensembles as
a second reference posterior.

**Related:** [M5](methods/reproduction-scope.md),
[Q3](#q3)

---

<a id="q8"></a>

## Q8 — Does a tighter prior close the Laplace/NUTS gap?

**Status:** **ANSWERED — yes** (2026-08-24). Sweep in
`sweeps/2026-08-24T0852_prior_scale_q8/`.

**Question:** [M6](methods/mode-vs-typical-set.md) says the posterior mass sits on a
shell of radius $s\sqrt{d}$ while the mode sits much further in, at
$\lVert w_{\text{MAP}} \rVert \approx 17$. Does shrinking the prior scale $s$ pull
that shell onto the mode and so shrink the Laplace-vs-NUTS disagreement — and does it
raise the likelihood's share of the Hessian?

**Why it mattered:** it was a falsifiable prediction of the mechanism in M6, cheap to
run, and therefore the most informative next experiment available.

**Answer: yes, monotonically, over two orders of magnitude.** Eight prior scales,
everything else held fixed. The predictive disagreement
$\text{mean}\,\lvert \bar p - \bar p_{\text{NUTS}} \rvert$ falls from **0.2542** at
$s = 100$ to **0.0030** at $s = 0.1$, and at the tight end the two mean confidences
agree to three decimals — the Gaussian approximation becomes essentially exact once
the typical set collapses onto the mode. Full table:
[M1 § measured sweep](methods/prior-scale-calibration.md#measured-sweep).

Three things the sweep added beyond the prediction:

1. **The gap does *not* saturate at the loose end.** A prediction that it would —
   on the grounds that the pinned weight decay caps the effective prior variance —
   was **falsified** by the $s = 100$ run: the gap grew and the widths nearly
   doubled. The cap is real but applies to the **MAP location only**
   ($\lVert w_{\text{MAP}} \rVert$ plateaus at 14.29 → 17.30), because
   `weight_decay` acts in stage 1 alone.
2. **The mechanism is the ratio, confirmed.** Laplace's centre is pinned while its
   width grows $\propto s$, so its ratio $\to 0$ and its confidence $\to 0.5$; NUTS
   scales centre *and* width together, so its confidence freezes at 0.845 from
   $s = 10$ upward.
3. **M6's shell radius holds at four scales, not one.** Median
   $\lVert w \rVert / s$ = 4.52, 4.41, 4.42, 4.38 against the predicted
   $\sqrt{d-1} = 4.36$.

**What this does *not* say: that a tight prior is a good setting.** The two tightest
scales are **training failures**, and the vanishing gap there is an artefact of that,
not a success:

| $s$ | train accuracy | conf (whole grid) | conf (far field) |
|---|---|---|---|
| 0.1 | **0.865** | 0.861 | **0.897** |
| 0.316 | **0.975** | 0.936 | 0.946 |
| 3.16 | 1.000 | 0.879 | 0.863 |

At $s = 0.1$ the network misclassifies 13.5% of its own training set on a task where
1.000 is achievable, and is *more* confident far from the data (0.897) than over the
grid as a whole. That is the paper's own failure mode, arrived at from the opposite
direction — it is not calibration.

The cause is the declared prior, on its own. With precision $1/\mathrm{var}_0 = 100$
at $s = 0.1$, the weights are confined to roughly $\pm 0.1$, the logits
$\phi(x)^\top w$ cannot reach the magnitudes the data needs, and the fit collapses.
Note that this is **not** the double-counting of
[I10](gotchas/prior-enters-map-fit.md): the weight decay's share of the total stage-1
precision, $\lambda / (1/\mathrm{var}_0 + \lambda)$, is 0.000% at $s = 0.1$ and only
becomes material at the *loose* end (50% at $s = 44.7$, 83% at $s = 100$). Tight-end
breakage is the model prior; double-counting is a loose-end effect.

Two consequences for reading the answer above:

1. **Approximation fidelity is not calibration.** Laplace matching NUTS to three
   decimals at $s = 0.1$ is a true statement about the *approximation* — a
   prior-dominated posterior really is nearly Gaussian, so a Gaussian fits it — and
   says nothing good about the *model*.
2. **The claim rests on the well-fitting rows.** Restricted to $s \ge 3.16$, where
   train accuracy is 1.000 throughout, the gap still moves 0.0481 → 0.2542, a factor
   of 5. Q8's answer therefore does not depend on the two broken rows at all.

Whether any scale in the well-fitting window is actually *well calibrated* remains
open and needs a real metric — [Q3](#q3).

**Learnings:** [M1 § measured sweep](methods/prior-scale-calibration.md#measured-sweep)
(the table, and the retraction of the saturation prediction) ·
[M6 § prediction confirmed](methods/mode-vs-typical-set.md) (the shell radius
re-tested at four scales) ·
[I10](gotchas/prior-enters-map-fit.md) (the prior enters the MAP fit through the
stage-1 ELBO; `weight_decay` on $w$ is a second copy of the prior and caps the mode
but not the width) ·
[0011](decisions/0011-sweep-layout.md) (how swept runs are stored) ·
[Q5 caveat](#q5) ($\hat r = 1.0122$ at
$s = 100$)
