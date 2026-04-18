# Phase 1.5 Hot Spots

Derived from cProfile on `bench_rollout_throughput` (200 random rollouts from a mid-game 40x40 state, 4 players).

Ranked by cumulative time in `territory_takeover` code. These are the functions that a Cython or C++ port should prioritize.

| Rank | Function | File:Line | Calls | tottime (s) | cumtime (s) |
|------|----------|-----------|-------|-------------|-------------|
| 1 | `step` | `engine.py:275` | 22003 | 0.1069 | 21.4638 |
| 2 | `detect_and_apply_enclosure` | `engine.py:163` | 22003 | 19.6562 | 21.1183 |
| 3 | `legal_actions` | `actions.py:46` | 44406 | 0.0930 | 0.1280 |

## Notes on Cython / C++ candidacy

### 1. `step`

**Moderate candidate.** `step` itself is glue — its tottime is mostly attribute lookups (`state.players[...].path.append`) and dict construction for `info`. A Cython cdef class for GameState would collapse the attribute chain; worth doing once the core primitives are ported.

### 2. `detect_and_apply_enclosure`

**Strong C/Cython candidate.** The BFS boundary flood over an `np.bool_` mask is allocation-heavy (fresh mask + deque per enclosure check) and dominates step cost on non-trivial loops. A C implementation with a stack-allocated visited mask and explicit (row, col) queue would be 5-20x faster.

### 3. `legal_actions`

**Moderate candidate.** Already ~0.6 µs per call (good), but called once per step so total contribution is meaningful. Easy win in Cython: the four-way bounds+EMPTY check maps directly to a typed memoryview.
