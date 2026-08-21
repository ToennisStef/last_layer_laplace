# I2 — Autoguide `init_loc_fn` for neural-network weights

**Source: read** (pyro 1.9.1 `pyro/infer/autoguide/guides.py`) **+ measured**.

Defaults, verified in pyro 1.9.1:

| Guide | default `init_loc_fn` |
|---|---|
| `AutoDelta` | `init_to_median` |
| `AutoNormal` | `init_to_feasible` |
| `AutoContinuous`, `AutoMultivariateNormal`, `AutoDiagonalNormal` | `init_to_median` |
| `AutoLaplaceApproximation` (a subclass of `AutoContinuous`) | `init_to_median` |

For a symmetric `Normal(0, s)` weight prior, `init_to_median` returns **0 for every
weight**: all hidden units start identical, symmetry is never broken.
`init_to_feasible` only guarantees support, not a sensible scale.

Override with an explicit fan-in draw (as in `BayesianMLP.init_loc_fn_guides`):

```python
from pyro.infer.autoguide import AutoLaplaceApproximation, init_to_value

init = init_to_value(values={"w": (torch.rand(h) * 2 - 1) * h**-0.5})
guide = AutoLaplaceApproximation(model, init_loc_fn=init)
```

Reference scheme: `nn.Linear` draws from `U(-1/sqrt(fan_in), +1/sqrt(fan_in))`
(fan_in = `in_features`; the `a=sqrt(5)` in its `kaiming_uniform_` call is a
backward-compatibility artifact, not real He init).

Fan-in scaling belongs **here**, not in the prior — [I1](prior-scale-units.md).

- API: <https://docs.pyro.ai/en/stable/infer.autoguide.html#module-pyro.infer.autoguide.initialization>
