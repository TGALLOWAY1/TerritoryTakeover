# MCTS Rollout Policy — Key Findings

Notes from developing `informed_rollout` and `voronoi_guided_rollout` in
`src/territory_takeover/search/mcts/rollout.py`. Recorded for the next person
who touches rollout policies so the losing attempts don't get re-discovered.

## Acceptance result

`test_informed_rollout_beats_uniform_at_same_iters` passed on the final
configuration — 100 games, 10×10 board, 200 iterations per move, seed 2026.
Informed cleared the ≥60% decided-game win-share bar. Wall time 21:35.

9 unit tests green, `ruff check` clean, `mypy --strict` clean.

## Iteration log

The winning configuration was reached after several attempts against matched-
budget tournaments. Each item below is something that sounded plausible and
failed — kept because the failures are more instructive than the successes.

### 1. Performance: numpy softmax was the bottleneck

- First pass scored 90–950 µs per move vs. a 10 µs target.
- Cause: `np.asarray` → `np.exp` → `rng.choice` setup overhead dominates for
  4-element inputs.
- Fix: pure-Python softmax (`math.exp` + cumulative-prob linear scan) in
  `_sample_softmax`. Saved ~13 µs per call.
- Takeaway: numpy vectorization is a loss below ~10 elements. For 4-way
  branching in a hot loop, stay in Python.

### 2. Flat enclosure bonus was *worse than uniform*

- Tried: skip `state.copy() + step()` and just add a fixed `W_ENCLOSE` whenever
  the trigger check fires (candidate adjacent to own-path tile that isn't the
  predecessor).
- Result: 20-game sample 8-9-3 (~47% share) — indistinguishable from uniform.
- Why: the trigger fires on many moves that touch own-path *without* closing a
  loop. A flat bonus rewards non-enclosures equally with real closures, adding
  noise instead of signal.
- Fix: restored the `succ = state.copy(); step(succ, action, ...)` simulation
  and use the real `claimed_count` delta. 30-game follow-up: 71% win rate.
- Cost: trigger fires on ~25% of scored actions; amortized ~25 µs/move. Over
  the original 10 µs design budget but measurably worth it at matched
  iterations.

### 3. Opponent-distance bonus added variance, not strength

- Tried: `W_OPPONENT * manhattan(t, nearest_enemy_head)` as a "spread out"
  signal.
- Result over 100 games: 43-51-6 (**45.7% share — FAILED the ≥60% bar**).
- Why: in this game staying near opponents creates the enclosure opportunities
  that actually win. Pushing away moved the rollout toward thinner territory
  and fewer claim events.
- Fix: removed the helper and weight entirely. Simpler code, better play.

### 4. Trap penalty had to be steep, not gentle

- Tried: flat `-5` when the candidate would leave the head with a single exit.
- Result: rollout didn't avoid traps reliably — softmax still gave dying moves
  non-trivial probability.
- Fix: two-tier hard schedule — `-100` for `empty_neighbors == 0` (immediate
  dead-end), `-10` for `empty_neighbors == 1` (single exit). With
  `tau = 0.5`, these magnitudes push trap probability to effectively zero.
- Takeaway: for "don't do this" signals, the penalty must dominate every other
  score component, not merely tilt the distribution.

### 5. Final winning formula

- Steep trap schedule
  (`-100` / `-10` / `W_REACH * empty_neighbors`).
- Enclosure-closure bonus via real `state.copy() + step()` simulation, gated
  by `len(path) >= 4` to skip early-game noise.
- No opponent-distance term.
- Voronoi-centroid pull used **only** by `voronoi_guided_rollout`.

## Meta-insights

- **Simulate when cheap, don't approximate when noisy.** A ~100 µs simulation
  firing on 25% of actions beat a zero-cost flat bonus. When the cheap proxy
  isn't rank-correlated with truth, it's actively harmful.
- **Smoke-test matchups at small N before the full gate.** 20–50 game runs at
  the same iteration budget caught two failure modes (flat enclosure, opponent
  distance) that would otherwise have cost 20+ minutes per diagnostic cycle.
- **More features ≠ stronger rollout.** The final policy has fewer score
  components than intermediate attempts and is the strongest. Every heuristic
  has to pay its way in a matched-iteration tournament; "plausibly helpful"
  fails in practice.
