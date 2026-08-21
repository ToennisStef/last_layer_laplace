# I1 — Prior scale: std vs. variance vs. precision

**Source: measured** (2026-08-18/19, [learning-log](../learning-log.md#2026-08-18--llla-reimplementation)).

`pyro.distributions.Normal(loc, scale)` takes a **standard deviation**.
The paper's code parameterises the prior by a **variance** `var0`, and the MAP
baseline uses weight decay `λ`, which is a **precision**:

```
λ = 5e-4  ->  var0 = 1/λ = 2000  ->  scale = sqrt(var0) ≈ 44.7
```

Passing `1/λ` where a scale is expected is a silent ~45× mis-scaling — no error,
just a differently calibrated model. See [two_moons_comparison.py](../../two_moons_comparison.py)
(`var0 = 1/5e-4; std0 = math.sqrt(var0)`).

Two places the same number enters, so an error hits twice:

1. the prior term of the negative log posterior → shifts the MAP;
2. the prior contribution to the Hessian → shifts the posterior covariance
   (see [I5](hessian-scaling.md)).

Do **not** fan-in scale the *prior* — that heuristic is for activation variance at
init and belongs in the guide init instead ([I2](pyro-autoguide-init.md)).

Calibration consequence: [M1](../methods/prior-scale-calibration.md).

- API: <https://docs.pyro.ai/en/stable/distributions.html#pyro.distributions.Normal>
