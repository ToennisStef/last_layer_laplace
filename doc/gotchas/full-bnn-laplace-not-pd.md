# I — The full-BNN Hessian is indefinite; only one of our two code paths says so

**Source: measured** (2026-08-24, two moons `n = 200`, `h = 20`, MAP by SGD 3000
steps, `var0 = 2000`; exact Hessians via `hessian.exact_hessian`, spectra via
`torch.linalg.eigvalsh`).

The same Hessian, three parameter sets, at the same MAP point:

| Hessian over | d | negative eigenvalues | min eigenvalue |
|---|---|---|---|
| last layer only (the notebook's case) | 20 | **0** | +6.4e-4 |
| all Linear weights (what the ablation samples) | 500 | **79** | -143.5 |
| all Linear weights + BatchNorm affines | 580 | **118** | -144.7 |

## Why the last layer is the lucky case
With the features frozen, the last-layer model is **logistic regression in $w$** with
a Gaussian prior — a convex problem, so its Hessian

$$H = \Phi^\top S\, \Phi + \tfrac{1}{\mathrm{var}_0} I$$

is positive definite everywhere and the Laplace approximation is well posed by
construction. ($\Phi$ is the matrix of frozen features, $S$ the diagonal matrix of
$\sigma_n(1-\sigma_n)$ curvatures, $\mathrm{var}_0$ the prior variance, $I$ the
identity — see [I5](hessian-scaling.md).) That is what makes LLLA cheap *and* safe,
and it is easy to mistake for a property of Laplace approximations in general.

Note *how narrowly* it holds. The smallest eigenvalue is
$6.4 \times 10^{-4}$, against a prior precision
$1/\mathrm{var}_0 = 5.0 \times 10^{-4}$:

$$\underbrace{6.4\times10^{-4}}_{\text{smallest eigenvalue of } H}
\;\approx\;
\underbrace{5.0\times10^{-4}}_{\text{prior alone}}
\;+\;
\underbrace{1.4\times10^{-4}}_{\text{all the likelihood contributes}}$$

The data is separable, so the likelihood contributes almost nothing in that
direction — **the prior is what keeps the matrix invertible**. Same fact
[M2](../methods/laplace-vs-vi-vs-mcmc.md) reports as "8 of 20 curvature directions
sit within $2\times$ of pure prior curvature".

A full BNN has no convexity to lean on: permutation and ReLU sign-flip symmetries
give exactly flat directions, and the surface has saddles. 79 of 500 directions are
genuinely negative — not round-off, and not fixable with more MAP steps. Which term
is responsible, and why the GGN drops exactly that term, is
[M7 §2.1](../methods/curvature-approximations.md).

## The two code paths diverge, and the louder one is the safer one

| route | last layer | full network |
|---|---|---|
| **notebook** (`torch.inverse`, [bnn_laplace.ipynb](../../bnn_laplace.ipynb)) | fine | **"succeeds"** — returns a Sigma with 79 negative eigenvalues and **60 negative variances** |
| **pyro** (`AutoLaplaceApproximation.laplace_approximation`) | fine | raises `_LinAlgError` / `ValueError` from the Cholesky |

`torch.inverse` is LU-based and never checks definiteness, so the notebook's
implementation does not fail — it silently produces a "covariance matrix" with
negative variances. Pyro builds `MultivariateNormal(precision_matrix=H)`, whose
Cholesky factorisation *does* check, and refuses.

So the answer to "does the notebook's implementation fail or does pyro's?" is: only
pyro's fails, and that is the correct behaviour. The indefiniteness is a property of
the model, not of either library. **The notebook's route is not a workaround** —
sampling from that Sigma would yield `nan`, and the probit approximation
`v = diag(phi Sigma phi^T)` could return a negative variance.

## What the paper's own code does about it

**Source: read** (2026-08-24, `paper/` in this repo).

Kristiadi et al. never invert an exact full-network Hessian. Every *active*
`exact_hessian(...)` call in `paper/` is over the last layer alone —
`paper/laplace/llla_binary.py:34` on `[mu]` (the concatenated last-layer `w, b`),
and `[W]` in the three 2-D notebooks. Their three Laplace flavours are:

| file | curvature | scope |
|---|---|---|
| `laplace/llla.py` | KFAC via BackPack | last layer |
| `laplace/llla_binary.py` | exact Hessian of the **likelihood only** | last layer |
| `laplace/dla.py` | diagonal **MC Fisher** (`grad**2`, labels *sampled from the model*) | all layers |
| `laplace/kfla.py` | Kronecker-factored MC Fisher | all layers |
| `notebooks/laplace/diag_laplace.py` | **full GGN**, built column-by-column with `ggn_vector_product` | all layers |

What each of those matrices is, and what it gives up against a dense exact
Hessian, is written up in [M7](../methods/curvature-approximations.md). Note
`dla.py` samples its labels *from the model*, which makes it an MC estimate of the
true Fisher and **not** the empirical Fisher — an earlier version of this note had
that wrong, and it is precisely the distinction Kunstner et al. (2019) is about.

Three devices make it work, and all three are load-bearing:

1. **GGN or Fisher instead of the exact Hessian.** Both are PSD by construction.
   `notebooks/laplace/diag_laplace.py` is the closest thing to a full-network
   Laplace in the repo, and it assembles the GGN — with
   `h = exact_hessian(loss, self.net.parameters())` sitting **commented out on the
   next line**, beside a commented-out `torch.symeig(...)[0][:10]`. They tried the
   exact Hessian, looked at its spectrum, and moved to the GGN.
2. **The prior precision is added as damping, afterwards.** Every flavour forms
   `inverse(H + tau*I)` with `tau = 1/var0` from the *likelihood* curvature, rather
   than differentiating the log-posterior. PSD + `tau*I` is strictly positive
   definite, so the inverse always exists.
3. **A linearised predictive.** `forward_linearized` propagates the Jacobian,
   `z = f_map / sqrt(1 + pi/8 * diag(J Sigma J^T))`, instead of sampling weights and
   pushing them through the network.

Also worth noting for architecture comparisons: the paper's own 2-D two-moons
notebook has its `nn.BatchNorm1d` lines **commented out** — that figure is produced
by a plain `Linear -> ReLU` MLP.

So "the paper did full Laplace" is true only with the qualifier *full-network GGN
with prior damping*, which is a different object from the exact Hessian, and closer
to what `laplace-torch` does than to what `AutoLaplaceApproximation` does.

## What was *not* done about it
No damping, no jitter, no eigenvalue clipping: those redefine "Laplace" mid-
comparison. `laplace-torch` sidesteps it with a GGN/Fisher approximation, PSD by
construction — a genuinely different and defensible method, but a different one.
Pyro's autoguide offers no such fallback.

`full_bnn_ablation.py` records the failure in `method_status` in each arm's
`config.json` and continues with the remaining methods. **An arm reporting three
methods instead of four is a result, not a broken run.**

## Consequence for Q2
"Does the LLLA/NUTS gap come from Laplace or from the last-layer restriction?"
cannot be answered by comparing Laplace across depths, because Laplace does not
exist at the deeper ones. Either run the comparison through VI and NUTS, or adopt a
PSD-by-construction Laplace variant and label it as such.

**Transferable?** yes — convexity of the last-layer problem, and its loss under any
deeper Bayesian treatment, is a property of the model class, not of this dataset.
