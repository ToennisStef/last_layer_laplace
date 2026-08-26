# I14 — The chain progress bars never reached `run.log`, and `run.log` can lose its tail

**Source: measured** (2026-08-26, pyro 1.9.1, python 3.13.11).

Two facts about the logging path, both discovered while trying to quieten the MCMC
output. They pull in opposite directions: `run.log` sees *less* than expected during
the run, and less than expected after an abort.

## 1. Progress bars go to stderr, so `Tee` never saw them
`Tee` ([prior_scale_sweep.py](../../prior_scale_sweep.py)) wraps **stdout only**.
pyro's per-chain bars are tqdm, which writes to **stderr**. Measured on a 3-chain run:

| | stdout | stderr |
|---|---|---|
| `disable_progbar=False` (default) | 1 line | **21 lines of bars** |
| `disable_progbar=True` | 1 line | **0 lines** |

So `run.log` was always clean; the clutter appears only in a launch log that folds
stderr in with `2>&1`, and on the terminal. With `mcmc_chains` bars interleaved and
rewritten in place with ANSI cursor codes, that log becomes unreadable.

**Fix:** `disable_progbar=True` on `MCMC`. Nothing is lost — the `stage 3: NUTS ...`
line and `mcmc.summary()` are both stdout, so start and finish are still recorded.

## 2. `Tee.flush()` is defined but never called
Nothing in either script calls it: no `flush=True`, no `reconfigure()`, no explicit
flush. `python -u` unbuffers the real `sys.stdout` — that is `Tee._stream` — but
`Tee._handle` is a plain `open()`, block-buffered at ~8 KB. Consequences:

- A **hard kill can truncate `run.log`** by up to 8 KB. The two aborted runs of
  2026-08-25 survived intact only because the interpreter unwound and closed the file;
  a `SIGKILL` would not have.
- The shell-redirected launch log is therefore the *more* trustworthy record, since
  `-u` does cover it. Worth knowing when the thing being diagnosed is a hang, where
  the last line before the freeze is the whole diagnosis ([I12](fork-deadlock-multichain-nuts.md)).

**Fix if wanted:** `buffering=1` on the `open()`, or call `Tee.flush()` after each
write. Not applied — the launch log covers it, and per-print flushing on a
50k-line run is not free.

**Transferable?** yes, both. "tqdm writes to stderr, so a stdout-only tee misses it"
and "Python block-buffers stdout the moment it is not a tty, and `-u` only covers the
real stream, not a wrapper's second handle" are library properties, not repo ones.

- API: <https://docs.pyro.ai/en/stable/mcmc.html#pyro.infer.mcmc.api.MCMC>
- <https://docs.python.org/3/using/cmdline.html#cmdoption-u>
