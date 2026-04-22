# ADR-003: Enclosure detection — trigger check + iterative BFS with scratch stamp

**Status:** Accepted
**Date:** 2026-04-22

## Context

When a player's snake path closes a loop, the EMPTY cells trapped
inside become that player's claimed territory. Every `step` has to ask
"did this move just enclose any cells?" and, if so, flood-fill the
pocket.

Hot-path constraints:

- Perf target: enclosure resolution < 200 µs on 40×40 (see
  `CLAUDE.md`). Per-turn overhead dominates training-loop throughput
  (~10⁶ turns/second target).
- Most turns do *not* trigger an enclosure. The common case must cost
  ~nothing.
- Python recursion limits: the default 1000-frame stack dies on a
  40×40 = 1600-cell flood fill. Recursion is off the table.

Alternatives considered:

- **Per-turn full flood fill from every EMPTY cell.** Correct but
  wastes 99% of the work; blows the perf budget by orders of
  magnitude.
- **Recursive flood fill on trigger.** Dies on large boards.
- **Trigger check + iterative BFS + fresh `np.bool_` mask per call.**
  Correct, safe, but allocation-heavy in tight loops.
- **Trigger check + iterative BFS + preallocated scratch with a
  stamp counter** (the choice).

## Decision

Two-phase detection:

1. **Cheap trigger check** (constant-time, no allocation): the placed
   cell triggers an enclosure only if it is adjacent to another
   same-player path cell other than its immediate predecessor. That's
   at most 4 set-membership lookups against `PlayerState.path_set`.

2. **Boundary-seeded BFS on trigger.** An iterative `collections.deque`
   BFS walks over EMPTY cells only, treating every non-empty cell
   (including opponents' paths) as a wall. The BFS uses a
   preallocated `np.bool_` mask on `GameState` with a stamp counter:
   each BFS increments the counter and marks visited cells with the
   current counter value, so no `memset` is needed between calls.

Attribution rule: enclosed cells always go to the triggering player,
even if opponent path tiles form part of the pocket boundary.

## Consequences

- **No-trigger cost ≈ 4 hashtable lookups.** Common case is essentially
  free.
- **Trigger cost < 200 µs on 40×40.** Verified against
  `benchmarks/bench_engine.py`; the optimized snapshot at
  `benchmarks/optimized.json` is the committed reference.
- **Safe on any board size.** No recursion; iterative queue bounded by
  board area.
- **Scratch buffer lives on `GameState`** — `reset()` keeps it alive
  across a training loop's million-game run, avoiding per-reset
  allocation churn.
- **Trade-off:** mask-stamp approach reuses memory, but a buggy mutation
  that leaks the wrong stamp value would silently corrupt the BFS.
  Mitigated by unit tests on every enclosure-detection path.

## References

- `src/territory_takeover/engine.py::detect_and_apply_enclosure`.
- `benchmarks/OPTIMIZATION_REPORT.md` — before/after BFS perf numbers.
- Commit `02144fa` (perf: scratch+stamp BFS, empty-count early exit).
