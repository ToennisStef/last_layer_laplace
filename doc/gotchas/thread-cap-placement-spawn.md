# I13 — A thread cap set inside `main()` never reaches a spawned chain worker

**Source: measured** (2026-08-26, 16 cores, python 3.13.11 / torch 2.13.0+cu130 /
pyro 1.9.1). Sibling of [I12](fork-deadlock-multichain-nuts.md): same `spawn`
mechanics, opposite direction — there spawn was the cure, here it defeats the fix.

`torch.set_num_threads(1)` is the in-code equivalent of launching with
`OMP_NUM_THREADS=1`. **Where it is written decides whether it does anything.**
Measured, parent and one spawned child, printing `torch.get_num_threads()`:

| where the cap is written | parent | spawned child |
|---|---|---|
| nowhere | 16 | 16 |
| `torch.set_num_threads(1)` inside `main()` | 1 | **16** |
| `torch.set_num_threads(1)` at module level | 1 | 1 |
| `OMP_NUM_THREADS=1` in the environment | 1 | 1 |
| `os.environ[...]` **before** `import torch` | 1 | 1 |
| `os.environ[...]` **after** `import torch` | **16** | 1 |

## Mechanism
A spawned worker is a fresh interpreter. It **re-imports the entry module** (as
`__mp_main__`, so module-level statements run) but **never executes the
`if __name__ == "__main__"` block** — so a cap applied in `main()` is invisible to it.
Environment variables need no such care: children inherit the process environment.
Row 6 is the mirror image — libgomp reads `OMP_NUM_THREADS` when it initialises, so
setting it after `import torch` is too late for the parent, while each child still
imports torch fresh and picks it up.

## Two separate failure modes, both fixed by the same cap

Torch spreads a single matmul across all 16 cores. That is the right instinct for the
large matrices of a big network's training step, and the wrong one twice over here.

### F1 — Cross-process contention: every chain claims every core
The NUTS chains run **in the children**; the parent only marshals results. Each child
is a fresh process with its own OpenMP pool, and each pool sizes itself to the *whole
machine* - it has no idea the other nine chains exist. So `mcmc_chains x 16` threads
compete for 16 cores, and every thread spends most of its life descheduled, waiting
for a core rather than computing. The chains do not run faster for having asked for
more cores; they run slower than if each had quietly taken one. Every context switch
also evicts the cache these tiny matrices depend on.

This is what makes row 2 the expensive one: capping only the parent caps the single
process that does no sampling, and nothing errors, so the log looks like the cap was
applied.

### F2 — Per-operation coordination overhead, paid on tiny matmuls
Splitting a matmul is not free: the pool has to be woken, the output rows divided, and
every thread has to arrive at a barrier before the result exists. That cost is roughly
fixed per operation, so it is negligible against a large matmul and ruinous against a
small one. Measured, single process, no contention at all:

| matmul | 16 threads | 1 thread | |
|---|---|---|---|
| **200x20** (this network) | 12332 us | **4.4 us** | 1 thread ~2800x faster |
| 1200x1200 | **13792 us** | 32301 us | 16 threads ~2.3x faster |

A 200x20 matmul is ~80k flops - a few microseconds of arithmetic. The coordination
around it costs three orders of magnitude more. The crossover is real, which is why
the same flag would be wrong for a large network.

F1 and F2 compound: F2 makes each chain's own arithmetic slower, F1 makes the chains
slow each other down on top of that. One thread per process removes both, and turns
chain count into a free parameter - one chain per core at no extra wall-clock, so 10
chains cost what 5 did.

## The fix
Module level, above `main()`, in the entry script (`full_bnn_ablation.py`). Preferred
over the launch flag for the same reason `mp_context="spawn"` is preferred over
remembering it: a flag you must remember will be forgotten, and the failure is silent.
`OMP_NUM_THREADS` remains the broader hammer — it also caps other OpenMP-linked
libraries in the process, which `torch.set_num_threads` does not.

**Transferable?** yes. "Process-wide runtime config set inside the `__main__` guard is
invisible to spawned children, while environment variables survive" is a
multiprocessing property, not a repo one.

- API: <https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html>
- <https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods>
