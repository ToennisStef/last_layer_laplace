# I12 — Multi-chain NUTS hangs forever if pyro forks after torch has warmed its thread pool

**Source: measured** (2026-08-25, `full_bnn_ablation.py --arms ll last2 full` on Linux,
16 cores, python 3.13.11 / torch 2.13.0+cu130 / pyro 1.9.1; repro at 4 latent dims).

The `ll` arm finished stages 1-2, printed `stage 3: NUTS, 5 chains x 500 samples`, then
did **nothing for 31 minutes**. All five workers were alive and idle -
`wchan: futex_do_wait`, `Threads: 1`, **CPU time `0:00`**: not one gradient evaluation.
The parent (20 threads) sat in `poll_schedule_timeout` polling their queue.

## Mechanism
With `mp_context` unset, `MCMC` uses the bare multiprocessing module
(`self.ctx = mp`, `pyro/infer/mcmc/api.py:263`), i.e. the platform default start method - `fork` on
Linux; pyro switches to spawn only for CUDA `initial_params` (same file, line 487). By stage 3,
5000 MAP steps and 20000 VI steps have warmed torch's 16-thread OpenMP pool.
**libgomp is not fork-safe**: the child inherits its mutexes but none of the threads
holding them, so the child's first parallel region blocks on a futex nobody will
release. Nothing is raised, so nothing catches it - the run hangs instead of failing.

Reproduced with a 4-dimensional logistic model, warm OMP pool, 2 chains:

| variant | result |
|---|---|
| `fork` (the default) | **hang**, killed at 90 s, no progress |
| `mp_context="spawn"` | completed in 22.5 s |
| `OMP_NUM_THREADS=1` + fork | completed in 2.7 s |

Dimension, `var0` and Bayesian depth are irrelevant - the `--quick` `ll` arm at d = 21
hung identically.

## Why it appeared now
[artifacts_run1_4chains](../../artifacts_run1_4chains/config.json) records a
**completed** 4-chain run on 2026-08-21 under python 3.12.11 / torch 2.13.0**+cpu**.
`torch` is unpinned in `pyproject.toml` and there is no `.python-version`, so a
re-resolve moved the venv to torch 2.13.0**+cu130**, whose CUDA-linked runtime makes the
fork unsafe even though every tensor here is on the CPU.

## The fix, and what it costs
`mp_context="spawn"` on the multi-chain `MCMC`. Both consequences are shared with
[I7](windows-multiprocessing-mcmc.md) - the same trap from the other side, where spawn
was the platform's choice and the problem, and here is the cure: `if __name__ ==
"__main__"` becomes load-bearing on **every** platform, and the model must stay
picklable (`PartiallyBayesianMLP` is a `PyroModule`, so it is - a lambda in `model()`
would not be).

`OMP_NUM_THREADS=1` also fixes it, with no code change, and was fastest above: these are
200x20 matmuls, so 16-thread intra-op parallelism is mostly thrashing. Better *launch*
flag; spawn is the better *default*, since it does not rely on remembering it.

**Transferable?** yes. "Never fork after using a threaded BLAS/OpenMP runtime" is a
libgomp property, not a repo one - as is reading zero worker CPU time as a fork-safety
bug rather than a slow sampler.

- API: <https://docs.pyro.ai/en/stable/mcmc.html#pyro.infer.mcmc.api.MCMC>
- <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>
