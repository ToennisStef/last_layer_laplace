# I5 — What the Hessian must be taken of

**Source: measured** (porting `hessian.py` / `Model.get_covariance`).

The Laplace posterior covariance is the **inverse Hessian of the negative log
posterior at the MAP**, not of the training loss:

$$-\log p(w \mid \mathcal{D}) \;=\; \underbrace{\mathrm{NLL}(w)}_{\text{summed over data}}
\;+\; \underbrace{\tfrac{1}{2}\, w^\top \left(\tfrac{1}{\mathrm{var}_0} I\right) w}_{\text{neg log prior}}$$

$$\Sigma = H^{-1}, \qquad H = \nabla^2_w \big[-\log p(w \mid \mathcal{D})\big]\Big|_{w = w_{\text{MAP}}}$$

- $w$, $w_{\text{MAP}}$ — the last-layer weights, and their MAP value.
- $\mathcal{D}$ — the training set, of size $N$.
- $\mathrm{NLL}(w)$ — negative log likelihood, here binary cross-entropy.
- $\mathrm{var}_0$ — prior variance; $1/\mathrm{var}_0$ is the prior precision.
- $I$ — identity matrix; $\Sigma$ — posterior covariance; $H$ — the Hessian.

## Two scaling traps

**1. The likelihood term must be *summed* over the data** (`reduction="sum"`). A
mean-reduced BCE computes $\tfrac{1}{N}\mathrm{NLL}$, which scales the whole Hessian:

$$H_{\text{mean}} = \tfrac{1}{N} H_{\text{sum}}
\qquad\Longrightarrow\qquad
\Sigma_{\text{mean}} = N \, \Sigma_{\text{sum}}$$

so the posterior comes out $N$ times too wide.

**2. The prior term must use the same $\mathrm{var}_0$ as the model** — see
[I1](prior-scale-units.md).

Mixed reductions are easy to end up with: the SGD training loop here uses the default
*mean* BCE, while `neg_log_likelihood` deliberately uses `sum` for the Hessian. That
is intentional — the mean/sum choice only rescales the optimiser's effective learning
rate, but it changes $\Sigma$ outright.

## Prediction

Turning the Gaussian over logits into a probability uses MacKay's probit
approximation:

$$p(y = 1 \mid x) \;\approx\; \sigma\!\left(\frac{m(x)}{\sqrt{1 + \tfrac{\pi}{8} v(x)}}\right),
\qquad
m(x) = \phi(x)^\top w_{\text{MAP}},
\qquad
v(x) = \phi(x)^\top \Sigma\, \phi(x)$$

- $\sigma(z) = 1/(1+e^{-z})$ — the sigmoid.
- $\phi(x)$ — the frozen feature vector at input $x$.
- $m(x)$, $v(x)$ — mean and variance of the logit at $x$.

MacKay (1992) — see [references](../references.md).
