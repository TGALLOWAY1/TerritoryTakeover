# Evaluator Weight Tuning — Key Findings

Notes from building `tune_weights` in `src/territory_takeover/eval/tuning.py`
and the `scripts/tune_weights.py` two-stage CLI. Written for the next person
who touches the tuner so the losing attempts don't get rediscovered.

## Acceptance result

Two-stage CLI run on 10×10, seed 2026, wall clock **1m13s**:

- **Stage A** (greedy vs `UniformRandomAgent`, 12 gen × 10 pop × 12 games):
  tuned greedy beats default greedy **85 / 100** in the validation match.
  Best weights saturate `territory_total = +2.0` and
  `enclosure_potential_area = -2.0` at the bounds.
- **Stage B** (paranoid-d2 vs Stage-A-tuned greedy, 8 gen × 8 pop × 10
  games): tuned paranoid-d2 beats default paranoid-d2 **100 / 100** in
  the validation match. Best weights look **nothing like Stage A's** —
  `mobility` saturates positive instead of `territory_total`,
  `choke_pressure` swings hard negative, `opponent_distance` becomes
  large positive.

Smoke unit test (`tests/test_tuning.py::test_tune_weights_smoke_beats_default`,
3 gen × 4 pop × 4 games on 10×10, seed 7) green in 3.7 s. Replay
reproducibility test green. `ruff check` and `mypy --strict` clean
on the new module and tests.

## Design decisions (and their trade-offs)

### 1. (1+λ)-ES, not CMA-ES

- **Choice**: vanilla (1+λ)-ES with per-axis Gaussian mutation, scaled
  by each axis's bound width so a single scalar σ does the right thing
  across dimensions of different scale.
- **Rejected**: CMA-ES via the `cmaes` package. Would have added a
  runtime dep to a project whose entire aesthetic is "numpy + stdlib".
  With only six weight axes the covariance machinery is overkill —
  the space is too small for CMA-ES's adaptation to pay for itself in
  generations before convergence.
- **Takeaway**: when the search space has ≤ ~10 dims and the oracle is
  noisy and expensive, a dumb (1+λ) with generation-to-generation
  parent re-evaluation is the right default. Mirrors the
  `ROLLOUT_FINDINGS.md` meta-insight that simpler rollouts beat
  clever ones under matched budgets.

### 2. Parent re-evaluated every generation

- **Choice**: candidate 0 of every generation re-evaluates the current
  parent with that generation's fresh seed. A child is promoted only
  if its win rate strictly exceeds the parent's *current* (re-measured)
  win rate.
- **Rejected**: cache the parent's first-ever win rate. Seductive —
  saves one eval per generation. But fitness is stochastic: a parent
  that got 9/12 on its first eval and 5/12 on a fresh seed would look
  unbeatable by any child's first-try 8/12. The cache locks in first-
  generation variance.
- **Cost**: one extra eval per generation. On 10×10 at games_per_eval=12
  that's ~1 s per gen — negligible vs. the benefit of unbiased
  selection pressure.

### 3. First candidate pinned to `default_evaluator()` weights

- **Choice**: gen-0 parent is always the hand-tuned default (clipped to
  bounds). The search starts from the best known point and can only
  move "outward" as σ allows.
- **Rationale**: guarantees the returned weights are never worse than
  the baseline even when mutation noise produces only regressions.
  Without this pin, seed 123 at the tiny smoke budget returned weights
  that *lost* 5/100 to default — strictly worse than doing nothing.
- **Empirical check**: in the full run, the default came in at ~14/12
  wins vs random in the gen-0 eval (7 out of 10 seeds found a better
  child in gen 0 alone; the remaining 3 stayed on the pin).

### 4. Bound-normalized per-axis σ

- **Choice**: `child_vec = parent_vec + σ · (upper − lower) · 𝒩(0, 1)`
  where the `(upper − lower)` multiplier is per-axis. Default bounds
  are `[-2, +2]` for all six features — so the axis span is uniform,
  but the generalized formulation makes it trivial to tighten one axis
  (e.g. `choke_pressure: (-1, 0)`) without rebalancing σ.
- **Rejected**: fixed σ in absolute units. Works today because every
  axis happens to use the same bounds, but breaks the moment a caller
  passes asymmetric bounds — a ±2.0 perturbation on a `(0, 0.1)` axis
  would be almost guaranteed to clip.

### 5. JSONL log with logged seed = replayable evaluation

- **Choice**: one JSON object per evaluation, append-only, containing
  `generation, candidate_index, weights, wins, games, win_rate, seed,
  elapsed_s`. The `seed` is the exact integer that flowed into the
  `SeedSequence.spawn` → `run_match` chain.
- **Tested**: `test_tuned_weights_replay_reproducible` picks arbitrary
  log lines, reconstructs the candidate, re-derives the per-opponent
  sub-seeds (via `SeedSequence(seed).spawn(len(opponents))`), and
  asserts the replay win count matches the logged one.
- **Rejected**: logging `(opponent_idx, sub_seed)` pairs. Tempting but
  redundant — the sub-seeds are a pure function of the root seed and
  the opponent list length. Logging one scalar keeps each line small
  and the spawn strategy canonical.

### 6. Two-stage curriculum in the CLI

- **Choice**: Stage A tunes greedy vs `UniformRandomAgent`; Stage B
  tunes paranoid-d2 vs a frozen Stage-A-tuned greedy.
- **Rationale observed in practice**: Stage A and Stage B produce
  **materially different** optimal weights (see acceptance result).
  Tuning paranoid-d2 directly against random would waste gradient
  budget — paranoid-d2 beats random at ~100% regardless of weights,
  so there's no signal to follow. Bootstrapping through a
  stronger-than-random opponent is the only way to get Stage B
  gradient information.
- **Cost**: Stage A's tuned weights get baked into Stage B's opponent,
  so Stage B is conditional on Stage A. This is a feature, not a bug —
  it's precisely the "weights good at 1-ply are different from weights
  good at 2-ply" phenomenon the user warned about, and the
  curriculum surfaces it directly.

### 7. Name-aware validation helper

- **Gotcha caught during CLI smoke**: `run_match._aggregate` keys its
  per-agent accumulator by `agent.name`. A validation match with two
  agents that share a name (both built from the same factory with
  `name="greedy-cand"`) collapses both seats into the same aggregator
  entry; `per_agent[0].wins` then reports **0** because every win
  flowed to index 1. Fix: `_validate_vs_default` builds candidates
  with `name="tuned"` and opponents with `name="default"`.
- **Takeaway**: duplicated agent names are a silent correctness bug in
  the harness, not a crash. Worth noting in `HARNESS_FINDINGS.md` as a
  future hardening target (validate name uniqueness at
  `run_match` entry).

## Things that will bite the next person

1. **`games_per_eval` must be even** for 2-player matches. `run_match`
   with `swap_seats=True` rejects `num_games % len(agents) != 0`; the
   tuner validates this up front with a matching error message.
2. **Stage A and Stage B must write to different log paths**. The
   default `resume=False` truncates the log at start of each
   `tune_weights` call. The CLI enforces this by writing to
   `stage_a.jsonl` / `stage_b.jsonl` separately.
3. **Factory closures run in the main process only**; only the
   *resulting* `Agent` crosses the pickle boundary. Lambdas as
   factories are fine. But the returned agent must itself be
   picklable, which rules out closures stashed on `_evaluator` or
   similar.
4. **Weight bounds and `DEAD_SENTINEL`**: default `[-2, 2]` is six
   orders below `-1e6`. Pushing bounds past ~`±1e4` would start to
   interact with the dead-player sentinel and produce nonsensical
   evaluations.
5. **Tiny-budget runs often return the default unchanged**. Not a
   bug — the pin in decision #3 means a generation where every child
   underperforms simply returns the default weights. Expect this at
   3 gen × 4 pop × 4 games and don't interpret it as tuner failure.
6. **Saturation at bounds is informative, not an error**. Stage A's
   best weights hit `+2`/`-2` on two axes. This typically means either
   (a) the feature's natural scale wants a larger weight and the
   bounds are the binding constraint, or (b) the feature is
   miscalibrated vs. its intended direction. Worth widening bounds
   and re-running if the saturated axis matters downstream.
7. **CLI and test use different opponents on purpose**. The unit test
   tunes against a default-weighted greedy for a clean "beat default"
   gradient at a tiny budget. The CLI's Stage A tunes against
   uniform-random as the task spec requires. These are the same
   `tune_weights` code path exercised with different oracle pools.

## Meta-insights

- **The oracle shapes the optimum.** Stage A's tuned weights are
  essentially useless as Stage B's starting point — the two landscapes
  disagree on which features matter most. If you ever want to reuse
  weights across search depths, re-tune; don't transfer.
- **A baseline pin is worth one search run.** Pinning gen-0 to
  `default_evaluator()` costs one evaluation and prevents every
  "tuner made things worse" failure mode that shows up at tiny
  budgets. Do it by default, not as an opt-in.
- **Log everything, compute derivatives later.** The JSONL log has
  ~300 bytes per evaluation and contains enough state to replay any
  evaluation deterministically. That's a free regression-test fixture
  for any future change to the tuner or the harness — if the replay
  assertion starts failing, something in the determinism chain broke.
- **Simpler optimizer wins when the oracle is noisy and expensive.**
  Same as `ROLLOUT_FINDINGS.md` meta-insight #3: the win rate oracle
  is so noisy at small `games_per_eval` that CMA-ES's covariance
  adaptation has no signal to work with. Plain Gaussian (1+λ) handles
  this gracefully — worst case it wastes a few mutations on nothing.
