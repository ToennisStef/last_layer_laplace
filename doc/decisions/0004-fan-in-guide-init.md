# 0004 — Fan-in `init_to_value` for all guides

**Status:** Accepted · **Date:** 2026-08-18

## Context
Pyro autoguides default to `init_to_median` / `init_to_feasible`. With a symmetric
`Normal(0, s)` weight prior, `init_to_median` starts **every** weight at 0
([I2](../gotchas/pyro-autoguide-init.md)).

## Decision
Pass an explicit fan-in draw to every guide, so Laplace and MVN-VI start from the
same distribution of initial points:

```python
init_to_value(values={"w": (torch.rand(h) * 2 - 1) * h**-0.5})
```

and seed NUTS separately from the fitted `w_map` ([0006](0006-nuts-as-ground-truth.md)).

## Alternatives
- Accept the defaults — degenerate start, and silently different between guide types.
- Initialise every method at `w_map` — makes SVI results depend on the SGD baseline
  and hides optimisation failures.

## Consequences
- Guide init and prior scale stay separate concerns; the fan-in rule appears only
  here and never in the prior ([0003](0003-prior-scale-from-weight-decay.md)).
- Results are seed-dependent — a seed sweep is not yet part of the comparison.

## Revisit if
SVI results move noticeably across seeds; then the init, not the method, is what is
being compared.
