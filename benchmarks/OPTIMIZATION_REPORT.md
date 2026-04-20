# Territory Takeover — Simulation Engine Optimization Report

**Date:** 2026-04-20
**Branch:** `claude/optimize-simulation-engine-pqdxl`
**Baseline:** `benchmarks/baseline_pre_optimization.json` (2026-04-20)
**Optimized:** `benchmarks/optimized.json` (2026-04-20)
**Constraints:** Pure Python + NumPy only (Cython/C deferred).

---

## A. Baseline — where time went

Measured on 20×20 / 30×30 / 40×40 boards, 4 players, uniform-random policy.
Per-step numbers include the full `step()` pipeline (legality, placement,
enclosure detection, turn advance, info-dict build).

| Metric                                 | 20×20      | 30×30      | 40×40      | Target   |
| -------------------------------------- | ---------- | ---------- | ---------- | -------- |
| `step` (no enclosure) — min            | 5.01 µs    | —          | —          | <20 µs   |
| `step` (enclosure trigger) — min       | 126.4 µs   | —          | —          | <250 µs  |
| `state.copy()` — min                   | 6.0 µs     | —          | —          | <50 µs   |
| `legal_actions` — min                  | 0.59 µs    | —          | —          | <2 µs    |
| Full random game — min / median        | 6.5 / 22.5 ms | 16.8 / 81.1 ms | 29.4 / 162.4 ms | — |
| Turns/game (mean)                      | 164.6      | 249.0      | 301.2      | —        |
| Games/min (random rollout fast path)   | 2,791      | 893        | 896        | ≥60,000  |
| Trigger fires / steps                  | 2,491 / 8,067 (31%) | 3,874 / 11,898 (33%) | 4,572 / 14,014 (33%) | — |
| Wasted fires (trigger but 0 enclosed)  | **85.2%**  | **87.3%**  | **87.2%**  | —        |

### Hottest functions (cProfile, 200 × 40×40 random rollouts, baseline)

From `benchmarks/HOTSPOTS.md`:

- `detect_and_apply_enclosure`: 19.66 s `tottime` / 21.12 s `cumtime` (of ~22 s total)
- `_advance_turn`: heavy because it re-ran `legal_actions` N times per step
- `step`: heavy glue — `legal_action_mask` allocating `np.zeros(4)` per call; `info` dict + `sum(1 for p in players if p.alive)` rescan; `_advance_turn` calling `legal_actions` building a list

The dominant cost was **`detect_and_apply_enclosure` allocating a fresh
`(H, W) np.bool_` reachability buffer and flooding the entire exterior on
every trigger fire — even though the trigger is over-approximate and
~87% of fires claim zero cells.**

---

## B. Root-cause analysis

1. **Wasted BFS on 87% of trigger fires.** The cheap trigger (placed cell
   adjacent to a non-predecessor same-player path tile) fires far more
   often than an actual enclosure occurs. Every fire paid the full
   `np.zeros((H, W))` allocation + deque alloc + perimeter-BFS cost,
   regardless of outcome. For 40×40 random play this is the #1 hotspot
   by a wide margin.

2. **Per-step allocation churn in `step()`.** `legal_action_mask` returned
   a fresh `np.zeros(4, dtype=np.bool_)` only to be read once; the
   `info: dict[str, Any]` was built for every step even when the caller
   discarded it; `sum(1 for p in state.players if p.alive)` rescanned all
   seats to decide terminal state. Net: ~5 µs per `step` even on the
   no-enclosure path.

3. **`_advance_turn` double-scanning legality.** The existing
   `legal_actions(state, candidate)` call in `_advance_turn` built a
   `list[int]` of up to 4 entries just to test truthiness, then threw it
   away.

4. **No fast-path rollout API.** MCTS / bulk self-play went through the
   full `step()` pipeline one Python frame at a time, including every
   `StepResult` + `info` alloc, even though rollouts don't need any of
   that observability.

---

## C. Implemented improvements

Each change was validated against the legacy `detect_and_apply_enclosure`
via `tests/test_engine_equivalence.py` (50 seeds × {10, 20, 40} boards).

### C.1 Scratch reachability buffer + monotonic stamp

*Files:* `src/territory_takeover/state.py`,
`src/territory_takeover/engine.py`.

The `(H, W)` reachability mask is now preallocated as
`GameState._scratch_reachable: NDArray[int32]` and reused across calls.
A monotonically increasing `_enclosure_stamp` counter replaces
per-call `np.zeros((H, W))`: a cell is "reached this call" iff
`scratch[r, c] == stamp`. Overflow handling (`stamp >= 2**31 - 1`) resets
the buffer — never observed in practice at 2 B calls' headroom.

**Effect:** eliminates one H×W numpy allocation per enclosure check.
Biggest single contributor to the 1.3× rollout gain.

### C.2 Incremental `empty_count` + zero-enclosed early exit

*Files:* `src/territory_takeover/state.py`,
`src/territory_takeover/engine.py`, `src/territory_takeover/rollout.py`.

The engine now maintains `state.empty_count` incrementally (seeded in
`_seed_player_state`, decremented on each placement and each claim). At
the end of BFS, if `reachable_count == empty_count` we **return 0
immediately** — skipping the `(grid == EMPTY) & (scratch != stamp)` mask
build, `.sum()`, and the masked assignment. This is the hot path:
~85-87% of trigger fires hit it.

**Effect:** the no-enclosure branch (7 of 8 fires) avoids two full-grid
numpy passes. Directly targets the "wasted BFS" root cause.

### C.3 Preserve legacy semantics exactly

*Files:* `src/territory_takeover/engine.py`,
`tests/test_engine_equivalence.py`.

An early attempt at a *localized* BFS (flood from interior seeds near
`placed_cell`) turned out to diverge from the legacy semantics in a
subtle case: the legacy implementation claims **every** EMPTY region
currently unreachable from the perimeter on any trigger fire — including
pockets that were closed earlier but hadn't yet been swept. We kept the
full-board BFS to preserve those semantics exactly and verified with
`_legacy_detect_and_apply_enclosure_full_bfs` (a byte-for-byte copy of
the original routine, kept in `engine.py` for test-only use).

### C.4 Trim `step()` per-call allocations

*File:* `src/territory_takeover/engine.py`.

- Inline 4-line legality check (bounds + `grid.item` EMPTY) replaces
  `legal_action_mask(...)[action]` → eliminates the `np.zeros(4)` alloc.
- `state.alive_count` field on `GameState`, maintained incrementally
  (decrements on `alive = False` transitions in `step()` illegal path
  and in `_advance_turn`'s no-moves path) → replaces the O(N)
  `sum(1 for p in state.players if p.alive)` scan.
- `info` dict is still built (API compat), but `simulate_random_rollout`
  avoids it entirely.

### C.5 `has_any_legal_action` short-circuit in `_advance_turn`

*File:* `src/territory_takeover/actions.py`,
`src/territory_takeover/engine.py`.

New `has_any_legal_action(state, pid) -> bool` returns on the first
legal direction. Replaces `if legal_actions(state, candidate):` in
`_advance_turn` — no `list[int]` allocation, no extra direction scanned
once one has been found.

### C.6 `simulate_random_rollout` fast-path API

*Files:* `src/territory_takeover/rollout.py` (new),
`src/territory_takeover/__init__.py`.

```python
def simulate_random_rollout(state: GameState, rng: random.Random) -> int
```

Drives `state` to terminal under uniform-random legal policy, in place.
No `StepResult`, no `info` dict, no `reward` return — just the total
number of half-moves executed. Inlines the legality check, coord math,
path append, enclosure call, and turn-advance in one tight Python loop.
Uniform random action via `rng.randrange(len(acts))`.

Semantics are asserted-identical to a `step()`-based reference loop with
the same action stream (`tests/test_rollout_api.py`, 5 cases incl.
50-seed 20×20 and 30-seed 40×40 equivalence suites).

### Before / after headline numbers

| Workload                 | Baseline g/min | Optimized g/min | Speedup |
| ------------------------ | -------------- | --------------- | ------- |
| 20×20 rollout throughput | 2,791          | 3,836           | **1.37×** |
| 30×30 rollout throughput | 893            | 1,184           | **1.33×** |
| 40×40 rollout throughput | 896            | 1,152           | **1.29×** |
| 20×20 full-game random   | 2,657          | 3,627           | 1.37×   |
| 30×30 full-game random   | 735            | 982             | 1.34×   |
| 40×40 full-game random   | 346            | 440             | 1.27×   |

| Primitive               | Baseline min | Optimized min | Speedup |
| ----------------------- | ------------ | ------------- | ------- |
| `step` (no enclosure)   | 5.01 µs      | 2.54 µs       | **1.97×** |
| `step` (enclosure)      | 126.4 µs     | 110.5 µs      | 1.14×   |
| `state.copy()`          | 5.99 µs      | 6.12 µs       | ~flat   |
| `legal_actions`         | 0.59 µs      | 0.60 µs       | ~flat   |

The `step` speedup is primarily from the inlined legality check and
incremental `alive_count` (alloc elimination); the enclosure and
rollout speedups come from the scratch buffer + empty-count early-exit
combo.

---

## D. Remaining bottlenecks (ordered by ROI)

1. **Python-level BFS loop in `detect_and_apply_enclosure` (≈75-85%
   of rollout time).** Even with scratch reuse and early-exit, every
   boundary BFS on a 40×40 board visits ~1,400 cells through Python
   for-loops. The next 3-5× gain almost certainly requires a Cython /
   `cdef` port of the enclosure routine (or the hot inner loop).
   Recommended as the next engineering step.

2. **`state.copy()` for MCTS tree search (~6 µs / copy).** Dominated by
   the numpy `grid.copy()` memcpy and the per-player `path_set.copy()`
   calls. For shallow MCTS playouts (where only the leaf state is
   rolled out), `simulate_random_rollout` on a single copied state
   already amortizes this. For deep search, a structural-sharing /
   diff-log representation is the follow-up but it's invasive.

3. **`step()` glue (`info` dict, `StepResult` wrapping).** ~1-1.5 µs
   per call even after trimming. `simulate_random_rollout` skips this
   entirely; RL callers that don't read `info` could use an
   `info=None` opt-out to save another ~15-20%.

4. **(Minor.)** `_advance_turn` still scans up to `N` candidates per
   call. On a 4-player game it's a no-op; the `has_any_legal_action`
   change already shaved its per-call cost by ~30%.

---

## E. Throughput target assessment

The original task brief quoted a `bench_rollout_throughput` target of
**≥1,000 games/sec (60,000 g/min)** on 40×40, which was ~50× short on
baseline. After the pure-Python/NumPy optimizations landed:

| Board | Baseline g/min | Optimized g/min | Original target | Recommended realistic target<br>(pure Python + NumPy) |
| ----- | -------------- | --------------- | --------------- | -------------------------------------------------- |
| 20×20 | 2,791          | **3,836**       | 60,000          | 5,000 – 12,000 g/min (next step: Cython BFS) |
| 30×30 | 893            | **1,184**       | —               | 1,500 – 3,500 g/min (Cython BFS) |
| 40×40 | 896            | **1,152**       | 60,000          | 500 – 2,000 g/min (Cython BFS); ≥ 5,000 g/min only with native code |

The original 1,000 g/s target was unrealistic under pure-Python
constraints (the BFS alone is ≥500 µs of Python frames on 40×40, which
caps us below 2,000 g/s of pure enclosure checks — and random rollouts
need 300+ of them). A reasonable staged plan:

- **Near term (landed in this PR):** ~1.3× across the board via
  allocation elimination and the rollout fast path. 20×20 clears
  3,000 g/min; 40×40 clears 1,000 g/min.
- **Next step (out of scope):** Cython port of
  `detect_and_apply_enclosure` — realistic 3-5× on top of current,
  which would bring 40×40 into the 3,000-6,000 g/min range and 20×20
  near or past 15,000 g/min.
- **Longer term:** batch-vectorize rollouts (multiple games per numpy
  call with SIMD) — this is the only path to the original 60,000 g/min.

### Correctness guarantee

Every optimization is gated by `tests/test_engine_equivalence.py`,
which replays 50 random seeds × {10, 20, 40} board sizes × up to 200
plies through both the optimized engine and a legacy copy preserved as
`_legacy_detect_and_apply_enclosure_full_bfs`, asserting identical
`grid`, per-player `path` / `claimed_count`, `alive`, `winner`, and
`done` at every ply. `tests/test_rollout_api.py` adds a
`simulate_random_rollout` ↔ `step()` loop equivalence harness for the
fast path. Both suites are green; the full `pytest tests/` suite passes
49/49 on the engine-adjacent tests.

### Follow-ups (not in this PR)

- Cython port of `detect_and_apply_enclosure` — highest ROI next step.
- Batched rollouts — many games per numpy op for MCTS self-play.
- `step()` `info=None` opt-out for RL callers that don't read it.
- Diff-log state representation for MCTS tree search (removes
  `grid.copy()` from the hot path entirely).
