# I5 — What the Hessian must be taken of

**Source: measured** (porting `hessian.py` / `Model.get_covariance`).

The Laplace posterior covariance is the **inverse Hessian of the negative log
posterior at the MAP**, not of the training loss:

```
-log p(w|D) = NLL(w) + 0.5 * wᵀ (1/var0) I w
Σ = H⁻¹,  H = ∇²(-log p(w|D)) |_{w_MAP}
```

Two scaling traps:

- the likelihood term must be **summed** over the data (`reduction="sum"`);
  a mean-reduced BCE scales `H` by `1/N` and inflates `Σ` by `N`.
- the prior term must use the same `var0` as the model — see [I1](prior-scale-units.md).

Mixed reductions are easy to end up with: the SGD training loop here uses the
default *mean* BCE, while `neg_log_likelihood` deliberately uses `sum` for the
Hessian. That is intentional; the mean/sum choice only rescales the optimiser's
effective learning rate, but it changes `Σ` outright.

Prediction then uses the probit approximation
`sigmoid(m / sqrt(1 + (π/8) v))`, `v = diag(φ Σ φᵀ)` (MacKay 1992) — see
[references](../references.md).
