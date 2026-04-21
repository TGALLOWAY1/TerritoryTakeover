# 04 — Experiment Plan

Plan for validating the n-step dense-target prototype landed in
Chunk 6. Assumes `02_reward_design.md` as the change-under-test and
`03_strategy_comparison.md` Option 3 as the chosen strategy.

## Board-size ladder

| Stage | Board | Players | Max half-moves | Role | Games/sec (random rollout baseline, §01 §1) |
|-------|------:|--------:|----------------:|------|--------------------------------------------:|
| S0 | 10×10 |   2 |  80 | Primary ablation (fast iteration, clean signal) | 410 |
| S1 | 15×15 |   2 | 120 | Confirmation (method generalises; not a single-board fluke) | ~120 |
| S2 | 20×20 |   4 | 240 | Scale stress (4-player, interaction density) | 62 |
| S3 | 40×40 |   4 | 480 | Aspirational. Run **only** after the Cython BFS port lands. Enclosure is 76 % of engine cost at 40×40 (§01 §1); current throughput would make the full ablation matrix impractical. | 10.7 |

Stages S0 → S1 → S2 form the validation matrix. S3 is the production
target once Opt 1 lands.

## Baselines

The ablation must compare against *external* agents, not just
terminal-mode AlphaZero. External anchors protect against the failure
mode where both conditions degrade equally and appear "matched".

| Baseline | Source | Why included |
|----------|--------|--------------|
| `RandomAgent` | `agents/random.py` | Floor. Any competent learner should clear 90 %+ at S0. |
| `GreedyAgent(default_evaluator)` | `agents/greedy.py` + `eval/heuristic.py` | Mid-tier anchor. Phase 3a Tabular Q failed to beat it at 0.116; it is a real test. |
| `UCTAgent(iterations=32, rollout=uniform)` | `agents/uct.py` | Tree-search anchor. Matched against AZ's 32 PUCT iters at tournament time. |
| `v0.3d-reference-10x10-2p` | `checkpoints/v0.3d-reference-10x10-2p/` | The current curriculum-trained AZ checkpoint. Lets us compare "old method at this budget" vs "new method at this budget" apples-to-apples. |

## Conditions (ablation arms)

Each arm is a full training run, same compute budget, same seed ladder.

| Arm | `value_target_mode` | Horizon | Rollout in MCTS | Net | Notes |
|-----|---------------------|--------:|-----------------|-----|-------|
| A0 — terminal baseline | `terminal` | n/a | bootstrap via NN leaf (current) | 32c / 2 res | Current Phase 3d behaviour |
| A1 — n-step (primary) | `nstep` (n=16, γ=0.99) | bootstrap via NN leaf | same net | The proposed fix |
| A2 — n-step longer | `nstep` (n=32, γ=0.99) | same | same | Sensitivity to n |
| A3 — n-step shorter | `nstep` (n=8, γ=0.99) | same | same | Sensitivity to n |
| A4 — n-step no-discount | `nstep` (n=16, γ=1.00) | same | same | Is discounting load-bearing? |

Arms A2/A3/A4 are cheap follow-ups; the core decision is A0 vs A1.

## Metrics

Primary, ordered by importance:

1. **Win rate vs GreedyAgent** over 200 swap-seat games at each
   checkpoint. Greedy is the strongest external baseline on 10×10
   and differentiates learner strength cleanly.
2. **Win rate vs `v0.3d-reference-10x10-2p`** — directly answers "is
   the new method stronger than the old one at matched budget".
3. **First-enclosure half-move** (a.k.a. `first_enclosure` in
   `selfplay.play_game_instrumented`). Phase 3d surfaced this as a
   sharper training-progress signal than win rate — it saturates
   later and has lower variance. Expected to improve *faster* under
   A1 than A0 because the n-step target rewards claim events
   directly.
4. **Training-steps to 50 % win rate vs Greedy.** Sample-efficiency
   proxy. If A1 reaches the threshold in fewer updates, the
   variance-reduction argument is empirically confirmed.
5. **Games/sec during self-play.** Expected unchanged within ±2 %;
   tracks the claim in `03_strategy_comparison.md` that the extra
   bootstrap forward is negligible.

Secondary:

- **Elo vs a fixed pool** (Random / Greedy / UCT-32). Use the existing
  harness in `eval/tournament.py`.
- **Value-target variance.** Log the sample variance of the value
  regression target within each training batch. Direct measurement of
  the variance-reduction claim in `02_reward_design.md`.
- **Value-loss trajectory.** Running EMA of MSE(V̂, target). Under A1,
  initial loss is larger (targets are no longer all ≈±1) but trend
  should be smoother.

## Budget

At 10×10/2p S0 scale, Phase 3d config runs ~15 min on CPU for 20
iterations (`configs/phase3d_curriculum_fast.yaml` + curriculum stage
s2). Each arm × 3 seeds × 2 boards = 24 runs ≈ 6 hours. Fits inside
one overnight batch.

S1 (15×15) adds ~4× per iteration; 24 runs there ≈ 24 hours. Schedule
as a separate batch after S0 confirms direction.

S2 (20×20/4p) adds ~20× per iteration; run **only** the A0 vs A1
arms × 2 seeds, not the sensitivity sweep. That is still 4 runs ×
~2 hours = 8 hours. Affordable.

## Success criteria

The prototype is "accepted" if **all** of:

1. A1 ≥ A0 on win rate vs Greedy at S0 with p < 0.05 (binomial CI on
   200 swap-seat games, pooled across 3 seeds).
2. A1 ≥ A0 on first-enclosure half-move at S0 (faster improvement;
   statistical test via paired Wilcoxon across iteration-by-iteration
   checkpoints).
3. A1 shows value-target variance strictly lower than A0 at the same
   iteration count (direct verification of §02's mechanism claim).
4. A1 at S1 maintains or extends the advantage over A0 (no "tuned to
   one board").

Acceptance does not require the advantage to be large. A 10-percentage-
point win at matched budget would already be a strong outcome given the
change is one configuration flag.

## Rejection criteria

Kill the arm (leave `value_target_mode="terminal"` as default forever)
if **any** of:

1. A1 < A0 on win rate vs Greedy at S0 after 3 seeds. No fallback; the
   mechanism argument would be falsified.
2. A1 training diverges (value loss NaN or monotonic increase) in
   more than 1/3 seeds. Indicates the bootstrap destabilises at this
   horizon.
3. Wall-clock overhead > 10 % at S0. Indicates the extra evaluator
   call per training step is not free the way §03 claims.

## What this experiment is **not**

- **Not a Cython-port benchmark.** That is the separately-scheduled
  engineering step. If it lands during this experiment, note the
  throughput delta but do not let it interact with the A0 vs A1
  comparison.
- **Not a hyperparameter sweep.** A2/A3/A4 are one-axis sensitivity
  probes, not a full sweep over n, γ, horizon, learning rate.
- **Not a final tournament.** The winner of A0 vs A1 becomes the new
  default; a subsequent Elo tournament against Greedy / UCT-32 /
  Paranoid-d2 can be run then.

## Follow-up experiments (stretch, outside this task)

- **Cython-port throughput measurement.** Repeat S3 (40×40/4p) with
  the Cython-BFS-enabled engine; expected 3–5× rollout throughput
  (§01 §5). Record against the current 10.7 games/sec anchor.
- **15×15/4p curriculum continuation.** Extend Phase 3d to one more
  stage using the winning arm as the default. Phase 3d already has
  the curriculum plumbing; only the config grows.
- **Auxiliary potential head.** If A1 wins, consider an auxiliary
  loss on `Φ(s) = α·claimed/area + β·reachable/area` to inject the
  Voronoi/reach signal without adding it to the reward. Second-order
  improvement; run only after A1 is accepted.
