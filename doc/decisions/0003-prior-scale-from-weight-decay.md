# 0003 — Prior scale tied to the MAP weight decay

**Status:** Accepted, provisional · **Date:** 2026-08-19

## Context
The prior scale is the dominant calibration knob ([M1](../methods/prior-scale-calibration.md)),
and it must be *some* number before anything can be compared. Weight decay
`λ = 5e-4` was inherited from the paper's training setup.

## Decision
Derive the prior from the weight decay rather than picking it independently:

```python
var0 = 1 / 5e-4   # = 2000
std0 = math.sqrt(var0)   # ≈ 44.7  -> dist.Normal(0, std0)
```

The same `var0` enters the neg-log-prior, the Hessian, and all three inference
methods, so the comparison holds the prior fixed.

## Alternatives
- Tune it by marginal likelihood (Ritter et al. 2018) — the principled option,
  parked as [Q1](../open-questions.md#q1) because the paper was not read yet.
- Pick a round number (`s = 1`) — arbitrary, and inconsistent with the MAP baseline.
- Fan-in scale it — **rejected outright**, wrong tool: [I1](../gotchas/prior-scale-units.md).

## Consequences
- Consistency with the MAP baseline, *not* calibration. `λ` was never itself tuned
  for calibration, so "well-calibrated at this prior" is unclaimed.
- Any statement about LLLA vs. VI vs. MCMC is a statement *at this prior*.
- Unit discipline required in three places → [I1](../gotchas/prior-scale-units.md).

## Revisit if
[Q1](../open-questions.md#q1) is implemented — a tuned prior supersedes this by
construction.
