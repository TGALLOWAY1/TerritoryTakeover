# Optimization / Search Bottleneck Analysis

**Branch:** `claude/analyze-optimization-bottlenecks-KWyXw`
**Deliverable:** report (this file) + four supporting notes under
`docs/optimization_analysis/` + a flag-gated prototype of the top fix
(Chunk 6: AlphaZero n-step bootstrapped value target).

**TL;DR.** The dominant bottleneck is **not** engine throughput and is
**not** search depth per se. It is a **method/game mismatch**:
AlphaZero regresses a 300-step-trajectory value head against a single
terminal label per trajectory while the engine is already emitting a
dense per-step reward that Tabular Q and PPO both consume. Fix the
value target first (n-step bootstrap), then port the enclosure BFS to
Cython, then scale the curriculum. The rest of this document is the
evidence.

## Method summary

- Built `benchmarks/bench_pipeline.py` — method-level cProfile harness
  that partitions tottime into named buckets (enclosure / mcts /
  torch_ops / etc.) and a top-unmatched list for sanity-checking the
  bucket predicates.
- Profiled rollout (board 10/20/40), UCT-vs-Random (board 10 with
  200 iters, board 20 with 100 iters), and AlphaZero one-iteration
  (board 6 and 10, at Phase 3d curriculum net/train scale).
- Cross-referenced existing engine benchmarks
  (`benchmarks/optimized.json`, `HOTSPOTS.md`, `OPTIMIZATION_REPORT.md`)
  and Phase 3 lab notes (`KEY_FINDINGS.md`).
- Raw JSON summaries are committed under `benchmarks/profiles/`.
- Four supporting notes:
  - [01_cost_breakdown.md](optimization_analysis/01_cost_breakdown.md)
  - [02_reward_design.md](optimization_analysis/02_reward_design.md)
  - [03_strategy_comparison.md](optimization_analysis/03_strategy_comparison.md)
  - [04_experiment_plan.md](optimization_analysis/04_experiment_plan.md)

## Answers to the six-question decomposition

### A — Engine throughput

Engine is well-optimised for its current shape. `legal_actions` at
~0.6 µs, `GameState.copy()` at 6 µs on 40×40, `step` (no-claim) at
~2 µs. The single hot spot is `detect_and_apply_enclosure` — 63–76 %
of rollout tottime at board 10/20/40
([01 §1](optimization_analysis/01_cost_breakdown.md#1-engine-level-cost-pure-step-no-search)).
Already flagged for a Cython port in `benchmarks/HOTSPOTS.md`;
realistic 3–5× payoff.

This is a *real* bottleneck, but **second-rank.** See §F for why.

### B — Depth to terminal

All three rollout policies in `src/territory_takeover/search/mcts/rollout.py`
(lines 99, 263, 320) loop `while not state.done`. No horizon parameter.
At 40×40 `max_half_moves=480`; a PUCT tree that rolls every leaf to
480 is structurally expensive regardless of engine speed. Horizon
truncation with a leaf value estimator is a standard, cheap win — 3–10×
on search workloads, composes with (A).

Third-rank — important but subordinate to the learning-signal problem.

### C — Reward sparsity

Real only for AlphaZero. Tabular Q (`rl/tabular.py`) and PPO
(`rl/ppo.py`) already consume `engine.py:461`'s per-step
`1.0 + claimed_this_turn` reward. AlphaZero
(`alphazero/selfplay.py:59–66`, `alphazero/train.py:118–122`) regresses
the value head on the terminal-only normalized score. Subsumed by (F).

### D — Evaluation function cost

`eval/heuristic.py` (Voronoi + reach BFS + enclosure potential BFS) is
expensive (~200 µs/state on 40×40) but lives on the Max-N / Paranoid
critical path, not the RL training path. Separate optimisation lane —
a Cython port of the three BFS passes would buy ~5× on Max-N-d3 eval
but the RL loop does not call this code.

Not a bottleneck for the main training workload.

### E — State representation

6.2 µs per copy at 40×40 (`OPTIMIZATION_REPORT.md §E`). 0.1–0.8 % of
any profile at any scale
([01 §4](optimization_analysis/01_cost_breakdown.md#4-engine-level-anchor-from-benchmarksoptimizedjson)).
Already numpy-memcpy + shallow struct copy. **Not a bottleneck.**
Listed only because the task prompt named it.

### F — Method/game mismatch

**The dominant bottleneck.** Evidence:

- Games reach 80 / 160 / 240 / 480 half-moves at 10/15/20/40 (per
  `configs/phase3d_curriculum_fast.yaml`).
- The engine already knows, at every step, how many cells each player
  just claimed (`engine.py:461`: `reward = 1.0 + claimed_this_turn`).
- Tabular Q and PPO consume this dense signal directly.
- AlphaZero consumes only the terminal `2·score − 1 ∈ [−1, +1]` vector
  (`selfplay.py:59`, `train.py:118`). Every state in a 300-step
  trajectory gets the same scalar target.
- Phase 3d lab notes (`KEY_FINDINGS.md`): the curriculum beat direct
  training 0.708 vs 0.438 win rate (Cohen's d=+2.01) primarily by
  **shortening the effective episode**. The curriculum is a
  work-around for the variance problem induced by terminal-only
  training on long episodes, not a fix.

Cutting training throughput 3× (A) does not help if the gradient
points the wrong way. **F first.**

## Ranked bottleneck list

**F > A ≈ B > C > D > E.** Detail and evidence in
[01 §5](optimization_analysis/01_cost_breakdown.md#5-ranked-bottleneck-list).

## Recommended strategy — Hybrid (Option 3)

[03_strategy_comparison.md](optimization_analysis/03_strategy_comparison.md)
compares three directions (Search-heavy / RL-heavy / Hybrid) across
sample efficiency, implementation complexity, runtime cost, scalability
to 40×40, engine compatibility, and residual bottlenecks. Hybrid wins
on all axes for this phase of the project:

1. **Now:** flag-gated AlphaZero n-step dense-target prototype
   (Chunk 6 of this task). Fixes F with low risk.
2. **Next:** Cython port of `detect_and_apply_enclosure` (scheduled;
   already scoped in `HOTSPOTS.md`). Fixes A.
3. **Then:** scale curriculum to 15×15/4p and 40×40/4p on the
   combined stack. Fixes B via the existing `max_half_moves` cap once
   it is no longer the bottleneck.

Option 1 (pure search, Cython first) is a 3–5× win but leaves F
unfixed — the training stack would regress against terminal labels on
even longer trajectories. Option 2 (PPO scale-up) is sample-inefficient
for this game; AZ needs ~10× fewer environment steps to reach
comparable strength. Option 2 is reserved as a dense-reward ablation
oracle, not a main training stack.

## Recommended reward design

Change one thing, flag-gated:

```
G_t^(n) = Σ_{k=0..n-1} γ^k · r_{t+k}               (if t+n ≥ T)
        + γ^n · V̂_θ(s_{t+n})                        (bootstrap if t+n < T)
```

where `r_t = claimed_this_turn_t / board_area` (per seat), `n=16`,
`γ=0.99`. Value head architecture unchanged; policy-head target
unchanged. Clamp to `[−1, +1]` for target-distribution stability.
Defaults preserve the current terminal-only behaviour — the new target
is opt-in via `SelfPlayConfig.value_target_mode: nstep`.

Guardrails (full list in
[02_reward_design.md §Guardrails](optimization_analysis/02_reward_design.md#guardrails-this-is-the-important-part)):

1. Uses the engine's **natural** per-step reward, not a shaped
   auxiliary — optimal policy is unchanged by definition (same goal:
   maximize total claim).
2. Target clamped to `[−1, +1]` — match the value-head output range.
3. Reward divided by board area — scale-invariant across the
   curriculum ladder; no per-stage LR retune.
4. Flag-gated, default off — all 39 existing AlphaZero tests keep
   passing.
5. Replay buffer extended with two new columns (`per_step_reward`,
   `step_index`) — loud failure on old buffer loads, no silent zero
   substitution.
6. Terminal case exactly recovers the old target when `n → ∞, γ=1` —
   strict generalisation, not replacement.
7. Bootstrap from the *current* net, no target net. TD(n) stable at
   this horizon without the extra machinery.

## Experiment structure

Full plan in
[04_experiment_plan.md](optimization_analysis/04_experiment_plan.md).
Summary:

- **Ladder:** 10×10/2p (primary) → 15×15/2p (confirm) → 20×20/4p
  (stress) → 40×40/4p (aspirational, after Cython port).
- **Arms:** A0 terminal baseline vs A1 n-step (n=16, γ=0.99), plus
  sensitivity probes A2/A3/A4 at n=8 / 32 and γ=1.0.
- **Baselines:** Random / Greedy / UCT-32 / `v0.3d-reference-10x10-2p`.
- **Primary metric:** win rate vs Greedy on 200 swap-seat games per
  checkpoint. Secondary: first-enclosure half-move (sharper training
  signal per Phase 3d), training-steps-to-threshold, value-target
  variance.
- **Budget:** S0 ablation matrix ~6 h CPU; S1 ~24 h; S2 (only A0 vs
  A1 × 2 seeds) ~8 h.
- **Acceptance:** A1 ≥ A0 vs Greedy at p<0.05, faster first-enclosure
  improvement, strictly lower value-target variance, result holds at
  S1. Rejection on any seed divergence or >10 % wall-clock overhead.

## Deliverables in this branch

- `benchmarks/bench_pipeline.py` — method-level profiling harness.
- `benchmarks/profiles/*.{json,pstats}` — seven profile runs,
  committable and replayable.
- `configs/bench_alphazero_{6x6,10x10}_phase3d_like.yaml` —
  realistic-scale AlphaZero profile configs.
- `docs/optimization_analysis/01_cost_breakdown.md`
- `docs/optimization_analysis/02_reward_design.md`
- `docs/optimization_analysis/03_strategy_comparison.md`
- `docs/optimization_analysis/04_experiment_plan.md`
- `docs/OPTIMIZATION_ANALYSIS.md` (this file).
- **Prototype (Chunk 6):** `value_target_mode: nstep` in
  `SelfPlayConfig`, extended `Sample` + replay-buffer columns, n-step
  target computation in `train_step`, new config, new test.

## Next 3 implementation steps

1. **Land the n-step target prototype (this branch, Chunk 6).**
   Flag-gated, default off. Unit + smoke tests. Ships without
   regressing existing behaviour.
2. **Run the S0 10×10/2p ablation (A0 vs A1).** Three seeds. Gate the
   default flip on the acceptance criteria in §Experiment structure.
3. **Cython-port `detect_and_apply_enclosure`.** Already scoped in
   `HOTSPOTS.md` (strong candidate, 5–20× realistic on boundary BFS).
   Do this after the ablation lands so the engine change and the
   learning-signal change do not collide in measurement.

A follow-on fourth step — scale curriculum to 15×15/4p on the
combined stack — is the natural production target once steps 1–3
succeed, but is out of scope for this task.
