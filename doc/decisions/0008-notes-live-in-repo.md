# 0008 — Notes that die with the code live in `doc/`

**Status:** Accepted · **Date:** 2026-08-21

## Context
Two kinds of note get produced here: ones a refactor would invalidate (a Pyro API
trap, why this guide, what a run showed) and ones that outlive the repo entirely
(what a paper proves, why reverse KL is mode-seeking).

## Decision
Split by that test — *would a refactor force me to edit this note?*

- **Yes → this repo**, under `doc/`: [decisions/](.), [learning-log](../learning-log.md)
  (the lab log), [gotchas/](../gotchas/), [open-questions](../open-questions.md).
- **No → outside the repo** (personal vault / AI-OS knowledge base): literature notes,
  concept claims, method comparisons.

`doc/` does not mirror, symlink, or duplicate anything external. The seam is the
`Transferable?` line on each [learning-log](../learning-log.md) entry: `yes` means it
is owed a note *elsewhere*, restated in own words, never copy-pasted.

## Alternatives
- Everything in the repo — concept notes rot with a project that ends.
- Everything in a vault — decisions and gotchas lose the code they refer to.
- Symlink `docs/` into the vault — sync tooling and git fight over the same files,
  and it collapses the split that is doing the work.

## Consequences
- Requires a periodic promotion pass, or the `Transferable? yes` lines just accumulate.
- Cross-repo lessons are invisible from inside this repo by design; only the
  external notes see them.

## Revisit if
The vault side never actually materialises — then folding the transferable material
back into `doc/` beats losing it.
