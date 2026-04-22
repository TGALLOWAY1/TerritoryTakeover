# ADR-006: SeedSequence spawn tree + seat rotation for reproducibility

**Status:** Accepted
**Date:** 2026-04-22

## Context

The tournament harness must satisfy three reproducibility demands:

1. **Single-integer reproducibility.** A whole tournament
   (hundreds of games, many agents) should replay bit-identically
   from one root seed.
2. **Parallel ≡ serial.** Results under `--parallel` must match
   serial runs exactly, not just statistically — otherwise CI
   investigations become painful.
3. **Seat-symmetry.** Spawn corners (top-left, bottom-right, etc.)
   carry meaningful asymmetry on this board. A head-to-head
   tournament must not report "agent A is stronger" when really
   "agent A got the better corner more often."

Naive approaches (a single `np.random.seed()`, agents sharing one
RNG, fixed seat assignment) fail at least one of these.

## Decision

**SeedSequence spawn tree.** `numpy.random.SeedSequence(root_seed)`
seeds the tournament. For each game, `ss.spawn(1)` produces one
child sequence. That child `spawn(1 + num_players)` yields one seed
for `new_game()` plus one seed per seat for per-agent RNG reset. The
spawn tree is deterministic, so serial and multiprocessing runs with
the same root seed produce bit-identical game logs.

**Per-worker isolation.** `run_match` pickles the agent list once.
Each multiprocessing worker deserializes a fresh copy and receives
seeds by value through the job args — nothing mutable crosses the
process boundary.

**Cyclic seat rotation.** With `swap_seats=True`, seat `k` in game
`i` gets agent `(k + i) mod N`. If `num_games % N == 0`, every agent
visits every seat equally often; the harness enforces this constraint
and raises otherwise.

**Wilson 95% CIs in-process.** `_wilson_ci(k, n)` is computed in
plain Python / math, no scipy dependency, so test and production
code agree on the intervals.

## Consequences

- **One integer reproduces everything.** The committed baseline
  reports (`docs/baseline_report*.md`) and the curriculum sweep
  (`docs/curriculum_puct_scaling.md`) each cite one `--seed` and
  any reader can regenerate them byte-identically.
- **Parallel safe.** `--parallel` pickles the agents and fans games
  out to `os.cpu_count()` workers without changing results.
- **Seat bias quantifiable.** Seat-rotation lets the harness attribute
  any remaining spawn asymmetry to actual strategic differences, not
  luck of assignment.
- **Constraint:** `games_per_pair` / `num_games` must be a multiple
  of the number of agents. For 2-agent head-to-heads this collapses
  to "must be even." Callers see a clear error instead of silent
  bias.

## References

- `src/territory_takeover/search/harness.py::run_match`,
  `::round_robin`, `::_seat_assignment`, `::_reseed_agent`,
  `::_wilson_ci`.
- `tests/test_harness.py` — seat-rotation balance, CI math, and
  serial/parallel equivalence assertions.
- `scripts/run_baseline_report.py`, `scripts/run_puct_scaling.py` —
  downstream consumers that inherit the reproducibility guarantee.
