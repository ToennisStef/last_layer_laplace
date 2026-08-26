# Promotions — repo → knowledge vault

Ledger of `Transferable? yes` items promoted out of this repo into the central
knowledge vault (`02_Notes`), per [decision 0008](decisions/0008-notes-live-in-repo.md).
The vault note is the concept restated in own words; this repo stays the
system-of-record for the *measured evidence*. Bidirectional: each vault note links back
to the repo ID below and to its `[[citekey]]`.

Promotion pass (2026-08-25):

| Repo ID | Detail file | Vault note (in `02_Notes`) | Lit link |
|---|---|---|---|
| M6 | methods/mode-vs-typical-set.md | Mode is not the posterior mass | Betancourt2017 |
| M7 | methods/curvature-approximations.md | (Laplace) Curvature Taxonomy | Martens2015, Botev2017, Kunstner2019, Dangel2020 |
| M2/M3 | methods/laplace-vs-vi-vs-mcmc.md | Laplace covariance is mostly the prior *(status: working)* | ritter2018a |
| M1 | methods/prior-scale-calibration.md | Prior scale is the calibration knob | MacKay1992, ritter2018a |
| M5 | methods/reproduction-scope.md | Reproducing a figure is not reproducing a claim | Kristiadi2020 |
| 0012 | decisions/0012-ggn-laplace-for-deep-arms.md | GGN is the exact Hessian of the linearized model | Immer2021 |
| I11 | gotchas/full-bnn-laplace-not-pd.md | Exact-Hessian Laplace is well-posed only for a last layer | Kunstner2019 |
| I5 | gotchas/hessian-scaling.md | Laplace needs the Hessian of the summed neg-log-posterior | MacKay1992b |
| I1 | gotchas/prior-scale-units.md | (Pyro) Normal takes std, not variance | — |
| I2 | gotchas/pyro-autoguide-init.md | (Pyro) autoguide init_to_median collapses NN weights | — |
| I3 | gotchas/pyro-laplace-guide-workflow.md | (Pyro) AutoLaplaceApproximation is two-phase | — |
| I10 | gotchas/prior-enters-map-fit.md | (Pyro) SVI weight_decay double-counts the prior | — |
| I7 | gotchas/windows-multiprocessing-mcmc.md | (Pyro) multi-chain MCMC on Windows needs a main guard | — |
| I9 | gotchas/param-store-reload.md | (Pyro) param-store reload breaks under torch 2.6 | — |
| Q5 | learning-log.md (2026-08-21 retraction) | MAP-seeding can manufacture a narrow posterior | Betancourt2017, Hoffman2011 |
| — | learning-log.md (2026-08-24) | MC predictive noise on a grid scales as 1 over sqrt(S) | — |
| I12 | gotchas/fork-deadlock-multichain-nuts.md | Never fork after a threaded OpenMP/BLAS runtime | — |
| I13 | gotchas/thread-cap-placement-spawn.md | Config in the `__main__` guard never reaches spawned children | — |
| I14 | gotchas/log-streams-and-buffering.md | tqdm writes to stderr; `-u` does not cover a wrapper's file handle | — |
| — | learning-log.md (2026-08-26) | Detaching a long job from an SSH/VSCode session (cgroups, not `nohup`) | — |

Standing rule for the next promotion pass: for each `Transferable? yes` line, (1)
ensure a `[[citekey]]` stub exists in `06_Literature/notes/`; (2) write/extend the
atomic vault note in own words with `source-repo` / `source-lit`; (3) add it to the
domain MOC; (4) add a row here.
