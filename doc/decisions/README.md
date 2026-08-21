# Decision records

One file per **choice that could reasonably have gone the other way**. Numbered,
append-only: a superseded decision is marked `Superseded by 00NN`, never deleted —
the reason it was once right is the useful part.

Not for: things learned ([lessons-implementation](../lessons-implementation.md),
[lessons-methodology](../lessons-methodology.md)), things not yet tried
([open-questions](../open-questions.md)), or what happened when
([learning-log](../learning-log.md)).

Template: [_template.md](_template.md).

| # | Decision | Status |
|---|---|---|
| [0001](0001-last-layer-only.md) | Last-layer Bayes only, deterministic feature map | Accepted, under review |
| [0002](0002-pyro-alongside-paper-code.md) | Keep the paper's exact-Hessian code as a reference implementation next to Pyro | Accepted |
| [0003](0003-prior-scale-from-weight-decay.md) | Prior scale `s = sqrt(1/λ) ≈ 44.7`, tied to the MAP weight decay | Accepted, provisional |
| [0004](0004-fan-in-guide-init.md) | Fan-in `init_to_value` for all guides, not the autoguide defaults | Accepted |
| [0005](0005-freeze-feature-map.md) | Freeze the feature map with `eval()` **and** `requires_grad_(False)` | Accepted |
| [0006](0006-nuts-as-ground-truth.md) | NUTS treated as the reference posterior for the comparison | Accepted, contested |
| [0007](0007-probit-predictive.md) | LLLA scored with the probit approximation | Accepted, to be revisited |
| [0008](0008-notes-live-in-repo.md) | Notes that die with the code live in `doc/`; nothing here mirrors a vault | Accepted |
| [0009](0009-uniform-sampled-predictive.md) | Score all four methods with the same sampled predictive | Accepted, supersedes 0007 for the comparison |
| [0010](0010-artifacts-and-figures-layout.md) | Runs persist to `artifacts/`, figures to `figures/` | Accepted |
