# ADR-002: State split — grid + redundant per-player caches

**Status:** Accepted
**Date:** 2026-04-22

## Context

ADR-001 established the `np.int8` grid as the single source of truth
for board contents. But several hot-path engine operations need O(1)
answers the grid alone can't provide cheaply:

- "Is this cell on player X's path?" — enclosure detection reads this
  on every placed cell and every BFS boundary step.
- "What is player X's head cell?" — needed to validate move legality.
- "How many cells has player X claimed?" — needed for win detection
  on every terminal check.

Computing any of these from the grid alone requires a full scan; doing
so per move would destroy the sub-millisecond per-move budget the
tournament harness needs.

Alternatives considered:

- **Grid-only, scan every lookup.** Clean invariant (one source of
  truth) but blows the perf target.
- **Grid-only with memoized scans.** Invalidation logic fragile; hard
  to keep correct under the mutation-in-place style used in `step`.
- **Grid + redundant caches on `PlayerState`** (the choice).

## Decision

Each `PlayerState` carries redundant, grid-derivable fields:

- `path: list[tuple[int, int]]` — ordered, for head / predecessor.
- `path_set: set[tuple[int, int]]` — O(1) membership.
- `head: tuple[int, int]` — current head cell.
- `claimed_count: int` — must equal `(grid == CLAIMED_CODES[pid]).sum()`.

**Contract:** every mutation that writes to the grid *must* update the
affected player's caches in lockstep. `engine.detect_and_apply_enclosure`
spells out the caller's contract for this explicitly (see
`CLAUDE.md` "Engine entry points" section).

`GameState.copy()` does a numpy `memcpy` of the grid and **shallow**
copies of each path list/set. Tuples of ints are immutable, so sharing
at copy time is safe; any subsequent mutation allocates a fresh
container.

## Consequences

- **Perf target met:** legal-move generation and enclosure trigger
  checks are O(1), under the 1 µs and 4-lookup budgets respectively
  (see `CLAUDE.md` performance-targets section).
- **Invariant overhead:** every new mutation path needs a code review
  for cache consistency. The cost is paid once per feature; the
  benefit repeats every game.
- **Cheap clone preserved:** the `copy()` profile is dominated by the
  grid `memcpy`; path list/set copies are cheap and the counts are
  value copies.
- **Debug audit:** test suite asserts `claimed_count` matches the
  grid-derived count on every invariant check.

## References

- `src/territory_takeover/state.py` — `PlayerState`, `GameState.copy`,
  `GameState.reset`.
- `src/territory_takeover/engine.py::detect_and_apply_enclosure` —
  caller contract.
- `CLAUDE.md` — "State split" section.
