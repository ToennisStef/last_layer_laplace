# doc/ — lessons learned

Working notes for this repo (re-implementation of *Being Bayesian, Even Just a Bit,
Fixes Overconfidence in ReLU Networks*, Kristiadi et al., ICML 2020) in Pyro.

Entry points:

| File | Contains |
|---|---|
| [decisions/](decisions/) | **Why** a choice was made, one numbered record each. Append-only. |
| [lessons-implementation.md](lessons-implementation.md) | Coding / library gotchas (Pyro, PyTorch). One line each → detail file. |
| [lessons-methodology.md](lessons-methodology.md) | What the *methods* actually do — calibration, approximation quality. |
| [open-questions.md](open-questions.md) | Ideas and TODOs **not yet tried**. Nothing here is verified. |
| [learning-log.md](learning-log.md) | The lab log — chronological: what was run, what it showed, which lesson came out. |
| [references.md](references.md) | Papers, code, API docs. |

Capture happens in [../NOTES.md](../NOTES.md) (scratch, gets emptied); that file
carries the routing table for emptying it.

Conventions:

- Detail files live in [gotchas/](gotchas/) and [methods/](methods/).
- Every lesson records **how it was learned** (`Source:` line): *measured*, *read*,
  *inferred*, or *assumed* — so unverified beliefs stay visibly unverified.
- Keep entries short. If it needs more than ~40 lines, it is a method note, not a gotcha.
- Scope: everything here would need editing if the code changed. Notes that would
  survive this repo (what a paper proves, general method claims) belong outside it —
  [decision 0008](decisions/0008-notes-live-in-repo.md); the seam is the
  `Transferable?` line in the [learning-log](learning-log.md).
