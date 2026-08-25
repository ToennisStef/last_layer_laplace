# 0012 — GGN Laplace, with both predictives, for the deeper ablation arms

**Status:** Accepted · **Date:** 2026-08-24

## Context
The exact-Hessian Laplace that `AutoLaplaceApproximation` provides exists only for
the `ll` arm. Beyond it the Hessian is indefinite — 79 negative eigenvalues of 500 —
and the Cholesky refuses
([I-full-bnn-laplace](../gotchas/full-bnn-laplace-not-pd.md)). Without a substitute,
the Q2 ladder loses its Laplace column exactly where the question is being asked.

Reading `paper/` settled what the substitute should be: Kristiadi et al. never
invert an exact full-network Hessian either. Every active `exact_hessian(...)` call
in their code is last-layer only; their deeper variants use a diagonal MC Fisher, a
Kronecker-factored MC Fisher, or — in `notebooks/laplace/diag_laplace.py` — a full
**GGN**, with the exact-Hessian line commented out beside a commented-out eigenvalue
check. What each of those matrices is:
[M7](../methods/curvature-approximations.md).

## Decision
Add **`Laplace (GGN)`** to `full_bnn_ablation.py` as a method distinct from the
exact-Hessian `Laplace`, never as a silent replacement for it. The precision is

$$C + \tau I, \qquad
C = J^\top \operatorname{diag}\big(p_n(1-p_n)\big) J, \qquad
\tau = \frac{1}{\mathrm{var}_0}$$

- $J$ — the Jacobian $\partial(\text{logits})/\partial\theta$ evaluated at the MAP,
  shape $N \times P$ for $N$ data points and $P$ latent weights.
- $p_n$ — the predicted probability at training point $n$; $p_n(1-p_n)$ is the
  curvature of the binary cross-entropy in the logit.
- $\tau$ — prior precision; $\mathrm{var}_0$ the prior variance.
- $I$ — the $P \times P$ identity.

$C$ is positive semi-definite for *any* network because the loss is convex in the
logit, so every weight $p_n(1-p_n) \ge 0$ and $C$ is a sum of PSD rank-1 terms.
Adding $\tau I$ with $\tau > 0$ makes the precision strictly positive definite,
hence always invertible. Full derivation: [M7](../methods/curvature-approximations.md).

Report **both** predictives as separate rows, because the GGN is the exact Hessian
of the *linearised* model and the two ways of spending it are not equivalent:

- `Laplace (GGN)` — sample weights, push them through the real nonlinear network.
  The estimator every other method uses ([0009](0009-uniform-sampled-predictive.md)),
  so the row is comparable across the whole table.
- `Laplace (GGN, linearised)` — propagate the Jacobian instead, making the logit at a
  test point Gaussian,

  $$f \sim \mathcal{N}\big(f_{\text{MAP}},\; \operatorname{diag}(J \Sigma J^\top)\big),
  \qquad \Sigma = (C + \tau I)^{-1}$$

  Self-consistent with the GGN, and what the paper does. Drawn by MC rather than
  evaluated with MacKay's closed form
  $\sigma\big(f_{\text{MAP}} / \sqrt{1 + \pi v / 8}\big)$, so that the same
  aleatoric/epistemic decomposition applies; the run log prints the gap between the
  two as a check.

The Jacobian is taken through `poutine.condition` + `poutine.trace` on the existing
model, so there is exactly one definition of the network.

## Validation
On the `ll` arm the logit is linear in `w`, so the term the GGN drops is identically
zero, linearisation is exact, and three things *must* hold. Measured, at 4000
predictive draws:

| check | expected | measured |
|---|---|---|
| GGN precision vs exact Hessian (eigenvalues) | identical | max rel. diff **1.2e-10** |
| `Laplace` vs `Laplace (GGN)` vs `(GGN, linearised)` confidence | identical | 0.675 / 0.676 / 0.676 |
| MC linearised vs MacKay probit | within MC noise | mean 0.0065, max 0.0259 (noise ~0.0079) |

The eigenvalue comparison runs automatically on every `ll` arm and its result is
recorded in `method_status`.

## Alternatives
- **Damp the exact Hessian** (jitter, eigenvalue clipping) — redefines "Laplace"
  mid-comparison and has no counterpart in the literature being reproduced.
- **`laplace-torch`** — implements this and more, but adds a dependency and a second
  architecture definition to keep in sync. Kept as the *independent* cross-check
  ([Q6](../open-questions.md#q6)), to be run from a separate script.
- **Only the linearised predictive**, as in the paper — would score one method with
  a different estimator than the rest, reintroducing precisely the confound
  [0009](0009-uniform-sampled-predictive.md) removed.
- **Diagonal or KFAC GGN**, as in their `dla.py` / `kfla.py` — structural
  approximations that exist to make large networks tractable. At `d = 501` the full
  GGN is cheap, so accepting their approximation error would buy nothing.

## Consequences
- The ablation table has up to six method rows; `Laplace` appears only for `ll`,
  the two GGN rows for every arm.
- The GGN and linearised rows share a posterior, so their `w_std` is identical by
  construction — any difference between those rows is the predictive alone. That is
  the point of carrying both.
- Cost is one Jacobian over the training set (`N x P`) plus one per predictive chunk
  (`chunk x P`), so the grid dominates: `grid_n = 100` means 10k rows at `P = 501`.
  Chunked through `cfg.pred_chunk`; lower `--grid-n` if it bites.

## Revisit if
Arms grow past a few thousand latent dimensions — then the dense `P x P` GGN stops
fitting and the diagonal/KFAC structure the paper uses becomes necessary rather than
optional.
