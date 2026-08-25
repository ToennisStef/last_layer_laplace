# I1 — Prior scale: std vs. variance vs. precision

**Source: measured** (2026-08-18/19, [learning-log](../learning-log.md#2026-08-18--llla-reimplementation)).

Three different numbers describe the same Gaussian prior, and the codebase uses all
three. `pyro.distributions.Normal(loc, scale)` takes a **standard deviation**; the
paper's code parameterises by **variance** `var0`; the MAP baseline's weight decay is
a **precision**.

$$\underbrace{\lambda}_{\text{precision}} \;=\; \frac{1}{\mathrm{var}_0}
\qquad
\underbrace{\mathrm{var}_0}_{\text{variance}} \;=\; \frac{1}{\lambda}
\qquad
\underbrace{s}_{\text{std}} \;=\; \sqrt{\mathrm{var}_0} \;=\; \frac{1}{\sqrt{\lambda}}$$

- $\lambda$ — weight decay / prior precision. Here $\lambda = 5\times10^{-4}$.
- $\mathrm{var}_0$ — prior variance, $1/\lambda = 2000$.
- $s$ — prior standard deviation, $\sqrt{2000} \approx 44.7$. **This** is what
  `dist.Normal` wants.

```
   λ = 5e-4  ──1/λ──►  var0 = 2000  ──√──►  s ≈ 44.7  ──►  dist.Normal(0, s)
      precision            variance          std             what Pyro takes
```

Passing $1/\lambda$ where a scale is expected is a silent $\sqrt{2000} \approx 45\times$
mis-scaling — no error, just a differently calibrated model. See
[two_moons_comparison.py](../../two_moons_comparison.py)
(`var0 = 1/5e-4; std0 = math.sqrt(var0)`).

## The same number enters twice, so an error hits twice

1. the prior term of the negative log posterior → shifts the MAP;
2. the prior contribution to the Hessian, $\tfrac{1}{\mathrm{var}_0} I$ → shifts the
   posterior covariance (see [I5](hessian-scaling.md)).

Do **not** fan-in scale the *prior*. That heuristic exists to keep activation
variance stable at initialisation and belongs in the guide init instead
([I2](pyro-autoguide-init.md)).

Calibration consequence: [M1](../methods/prior-scale-calibration.md).

- API: <https://docs.pyro.ai/en/stable/distributions.html#pyro.distributions.Normal>
