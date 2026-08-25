# setup/ — what the experiments *are*

Descriptive reference for the apparatus: architectures, what a configuration means,
parameter counts, what each script writes and what it costs. **Not findings** — those
live in [../methods/](../methods/) and [../gotchas/](../gotchas/) with a `Source:`
line saying how they were learned.

Each file carries a `Describes:` line naming the script it documents. The test for
whether something belongs here: *would it go stale when the **code** changes, rather
than when a **belief** changes?* If yes, it is setup.

| File | Describes | The experiment |
|---|---|---|
| [two-moons-comparison.md](two-moons-comparison.md) | `two_moons_comparison.py` | Four posteriors — Laplace, VI ×2, NUTS — over one frozen feature map. The base experiment everything else builds on. |
| [prior-scale-sweep.md](prior-scale-sweep.md) | `prior_scale_sweep.py` | The base experiment repeated once per prior scale. |
| [bayesian-depth-arms.md](bayesian-depth-arms.md) | `full_bnn_ablation.py` | The base architecture at three *Bayesian depths* — last layer, last two, all. |

Reading order for someone new to the repo: base experiment → then either the sweep
(varies the prior) or the arms (varies how much is Bayesian). Both of those import the
base experiment rather than reimplementing it.

Every parameter count on these pages is for the default `Config` — notably
$h = 20$ hidden units. Change `Config.h` and the numbers change with it; each page
carries the snippet that regenerates its own counts.
