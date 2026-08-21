# I7 — Multi-chain MCMC needs a `__main__` guard on Windows

**Source: measured** (2026-08-21, hit twice while wiring up `two_moons_comparison.py`).

`MCMC(..., num_chains=N)` with `N > 1` runs chains in separate **processes**. On
Windows there is no `fork`, so Python uses `spawn`, which re-imports the entry
module in each child. Without a guard the child re-runs the whole script and
multiprocessing aborts:

```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

The fix is in the **entry script**, not in the library code:

```python
if __name__ == "__main__":
    main()
```

Consequences that are easy to miss:

- Running `python two_moons_comparison.py` is fine (it has the guard), but
  `from two_moons_comparison import main; main()` in a **notebook or an unguarded
  script** trips it — the same code, a different caller.
- `main()` therefore catches this one `RuntimeError` and falls back to a single
  chain of `mcmc_samples * mcmc_chains` draws, printing a warning. The samples
  survive; **cross-chain `r_hat` does not**, which is the whole reason for
  running several chains ([Q5](../open-questions.md#q5)).
- The model must also be picklable for spawn to work. `BayesianMLP` is an
  `nn.Module`, so it is — but adding a lambda or a local closure to it would
  silently break multi-chain sampling.

- API: <https://docs.pyro.ai/en/stable/mcmc.html#pyro.infer.mcmc.api.MCMC>
- <https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods>
