# Phase 3 synthesis — learning agents for Territory Takeover

Phase 3 ran from 2026-04-19 to 2026-04-20 in four sub-phases, each on
its own branch, each with its own entry in `KEY_FINDINGS.md`. This
file is the cross-phase narrative: where we started, what each
sub-phase actually shipped, what the reference deliverable is, and
what the next person picking this up should know before they start.

Read `KEY_FINDINGS.md` for per-phase detail. Read this file first if
you have never seen the project before.

---

## What the game is (engine recap)

TerritoryTakeover is a turn-based 4-direction grid game. Each seat
controls a snake-like head that drops a path tile every half-move.
When a seat's path encloses a region of empty cells (a boundary-BFS
flood fill from the grid's outer edge cannot reach those cells), the
enclosed cells convert to that seat's claimed territory and the seat's
path is consumed. The game ends when all seats are either dead
(self-trapped) or stalemated; the winner is the seat with the most
claimed territory. Board size and player count (2 or 4) are
configurable; the action space is fixed at 4 directions.

The *enclosure mechanic* is the strategic step-function of the game.
Without it, the agent's best strategy is roughly "stay alive longer
than the opponent," which random and greedy heuristics already
implement. With it, the agent has to plan loops, avoid getting
enclosed itself, and trade off risk against territory gain. Every
sub-phase of Phase 3 measures progress by how fast / how well the
agent discovers and exploits enclosures.

---

## Phase 3a — Tabular Q-learning baseline (2026-04-19)

**Branch:** `claude/tabular-q-learning-baseline-nUlc2`.

**What it did.** Stood up ε-greedy Q-learning with a hand-crafted
7-tuple state encoder (head position, 4-way neighbor classes, phase
bucket). Trained 500 000 episodes of self-play on 8×8 / 2p; ran a
100 000-episode diagnostic on 10×10 / 4p.

**What it found.** Targets missed across the board: 0.394 vs Random
(target 0.80), 0.116 vs Greedy (0.55), 0.150 vs UCT-32 (0.30). The
10×10 / 4p diagnostic beat both Random (0.420) and Greedy (0.306)
above the 0.25 uniform baseline, which was mildly encouraging.

**Why it failed.** State aliasing: 7 909 unique encoder keys over
500 k episodes means a vast number of distinct board configurations
collapse onto the same key, and the Q-function cannot distinguish
"about to close a big enclosure" from "about to walk into a trap" when
they share the same local neighborhood. The 7-tuple encoder is
fundamentally under-expressive for 8×8, let alone 10×10.

**What it bought.** (1) A concrete measurement of how under-expressive
hand-crafted state features are on this game. (2) A shared-Q
self-play training loop and a reward-shaping design (per-claim reward
+ trap penalty + terminal rank bonus) that proved stable. (3) An
engine bug noted: `_default_spawns(8, 2)` returns diagonally-adjacent
seats, which silently breaks any experiment that doesn't override
spawns. Not fixed this phase.

---

## Phase 3b — PPO primitives (infrastructure only)

**Branch.** Pre-dates this session. Inherited as infrastructure only.

**What shipped.** PPO training primitives (actor-critic network,
rollout buffer, clipped objective) under
`src/territory_takeover/rl/ppo/`. No trained checkpoint, no
evaluation numbers. The "AlphaZero must exceed PPO" acceptance check
from later phase plans is therefore moot until a PPO model is trained.

**What the next person should do.** Train a PPO model on 8×8 / 2p
with the same encoder shape as Phase 3c/3d so it can be dropped into
the Elo pool. The code is ready; only a training run is missing.

---

## Phase 3c — AlphaZero primitives (infrastructure only, 2026-04-20)

**Branch.** `claude/complete-key-findings-phase-3b-c4tFV` (stale name
from 3b; work is 3c).

**What shipped.** A full AlphaZero stack under
`src/territory_takeover/rl/alphazero/`: observation encoder (fixed
seat ordering, turn one-hot), ResNet policy/value network with a
per-seat value head (one normalized expected score per seat, not a
single scalar), PUCT MCTS with per-seat value backup, Dirichlet noise
at the root, an LRU-cached batched NN evaluator, a replay buffer with
`npz` round-trip, self-play orchestration with AlphaGo Zero's
temperature schedule, and a train loop with masked softmax-CE +
MSE + L2. 39 tests pass.

**What did not ship.** No trained checkpoint. No evaluation table.
The gating tournament is stubbed (always-promote); the
4-dim-vs-scalar value-head ablation is configured but not executed;
no network-size ablations. Every missing piece is called out in the
3c KEY_FINDINGS entry.

**What the next person should do.** Run the 8×8 / 2p baseline
training (`configs/phase3c_alphazero_8x8_2p.yaml`) to populate a
headline result table, replace the gating stub with a proper
≥55%-over-N-games promotion policy, and then run the value-head
ablation.

---

## Phase 3d — Curriculum learning (2026-04-20)

**Branch:** `claude/curriculum-learning-phase-3d-ERrhn` (this file is
part of that branch).

**What it did.**

1. Made the AlphaZero network board-size-agnostic by replacing the
   FC policy/value heads with conv + `AdaptiveAvgPool2d(1)` +
   `Linear(small)` heads, gated behind a new `head_type="conv"` flag.
   `head_type="fc"` remains the default so Phase 3c's 39 tests pass
   unchanged. This was the blocking prerequisite: no curriculum work
   was possible while the trained heads depended on board size.
2. Built a curriculum module (`src/territory_takeover/rl/curriculum/`)
   with typed stage/schedule/promotion dataclasses, a YAML schedule
   loader, shape-filtered weight transfer across stages, and a
   trainer that persists `curriculum_progress.yaml` after each
   evaluation.
3. Built pairwise Bradley–Terry Elo
   (`src/territory_takeover/rl/eval/elo.py` +
   `scripts/compute_elo.py`) with a fixed anchor and multi-player
   rank decomposition.
4. Instrumented the self-play loop with a first-enclosure detector
   (the half-move at which the first claim happens) at zero hot-path
   cost.
5. Ran the headline ablation: 3 seeds × (curriculum, direct) = 6
   training runs on 10×10 / 2p with a matched 1 600-step self-play
   budget.

**What it found.** Curriculum beats direct at matched budget on every
measured axis:

- Final win rate vs Random: **0.708** [0.625, 0.875] vs **0.438**
  [0.313, 0.563] (bootstrap 95% CI). Cohen's d = +2.01.
- First enclosure: **27.7** vs **57.3** half-moves. Cohen's d = −2.98.
  Mann–Whitney U = 0 / 9 — **every curriculum seed discovers the
  enclosure mechanic earlier than every direct seed**.

Both effect sizes are "huge" by Cohen's benchmark (d ≥ 0.8 is large).
The N=3 sample size makes formal p-values under-powered; the primary
evidence is the effect size and the complete rank separation on
first-enclosure.

**Archived reference.** `docs/phase3d/net_reference.pt` (170 KB — the
curriculum arm, seed 0 checkpoint). Reproduction via
`scripts/train_curriculum.py --config configs/phase3d_curriculum_fast.yaml
--seed 0`. Git tag: `v0.3d-reference-10x10-2p`.

**Honest deferrals.** The original prompt targeted 40×40 / 4p; that's
multi-day per seed on CPU. Phase 3d scoped down to 10×10 / 2p as the
"large" end, documented the architecture-and-schedule path to 40×40
(no code changes needed), and called out the compute deferral
up-front. The Elo round-robin at 4 games per ordered seating is noisy
— the per-seed Elo does not cleanly discriminate curriculum from
direct at that sample size, but the training-time win-rate and
first-enclosure metrics do.

---

## Cross-phase Elo picture

At Phase 3d close, **only the AlphaZero conv-head curriculum agents
have trained checkpoints**. Tabular-Q-3a does not transfer to 10×10
(its encoder is tied to the training board), Phase 3b PPO has no
checkpoint, Phase 3c AlphaZero has no trained checkpoint. The Elo
table in `docs/phase3d/elo_final.csv` is therefore a within-Phase-3d
ranking plus Random and Greedy anchors.

The most defensible cross-phase statement is: **the curriculum arm is
the first learning agent in this project to beat Random by a wide
margin at 10×10 / 2p** (win rate 0.708 at best seed 0.875 vs Random's
trivially 0.50 from symmetry). Tabular-Q-3a on 8×8 / 2p beat Random
at only 0.394. A like-for-like comparison would require training
Tabular-Q-3a on 10×10 (its encoder can run there but is expected to
alias even worse) and training PPO on 10×10. Both are recommended
follow-ups.

---

## Recommended next actions (one session of effort each)

1. **Train PPO on 10×10 / 2p.** Phase 3b primitives are ready. This
   tests the "AlphaZero exceeds PPO" acceptance criterion directly and
   adds a second trained agent to the Elo pool.
2. **Re-run the Elo round-robin at ≥ 50 games per pair and
   `puct_iterations=8`.** At the current 4 games / pair the Elo
   variance is ~±100 points, which obscures real ordering. A 50
   games-per-pair tournament is a couple of hours wall-clock.
3. **Scale the curriculum to 15×15 / 4p** using the same
   `phase3d_curriculum_fast.yaml` schedule with an additional stage.
   This verifies the curriculum scaling argument and produces a
   multi-player reference agent.
4. **Replace the stubbed gating tournament** (carryover from Phase
   3c). At reduced scope it didn't matter, but at longer runs
   always-promote dynamics are known to produce checkpoint regression.
5. **Execute the aspirational 8 000-step configs**
   (`configs/phase3d_curriculum.yaml` + `phase3d_direct.yaml`, not
   the `_fast` variants). The fast numbers are strong enough to carry
   the scientific claim, but the aspirational budget was the
   originally-designed experiment.

---

## Where things live

```
src/territory_takeover/
    engine.py, state.py, constants.py, actions.py    # core engine
    rl/
        tabular/     # Phase 3a
        ppo/         # Phase 3b primitives
        alphazero/   # Phase 3c primitives; 3d network.py extended
        curriculum/  # Phase 3d
        eval/elo.py  # Phase 3d
    search/          # Random / Greedy / UCT agents (pre-3)
configs/
    phase3a_*.yaml                             # Phase 3a
    phase3c_alphazero_*.yaml                   # Phase 3c
    phase3d_curriculum{,_fast}.yaml            # Phase 3d
    phase3d_direct{,_fast}.yaml                # Phase 3d (control)
    phase3d_elo_pool.yaml                      # Phase 3d
docs/
    phase3a/    # 3a training curves + eval CSVs
    phase3d/    # 3d reference checkpoint + ablation + elo artifacts
experiments/
    curriculum_ablation.py                     # Phase 3d headline harness
scripts/
    train_tabular_q.py, eval_tabular_q.py      # Phase 3a
    train_alphazero.py, eval_alphazero.py      # Phase 3c
    train_curriculum.py                        # Phase 3d
    compute_elo.py                             # Phase 3d
tests/
    test_rl_tabular_*.py                       # Phase 3a
    test_rl_ppo_*.py                           # Phase 3b
    test_rl_alphazero_*.py                     # Phase 3c (+3d variable-size)
    test_rl_curriculum_*.py                    # Phase 3d
    test_rl_eval_elo.py                        # Phase 3d
```

---

## One paragraph, if you read nothing else

Phase 3 established that (a) hand-crafted state features are too
under-expressive for this game even at 8×8 (Phase 3a), (b) a full
AlphaZero stack with a per-seat value head is now in the repo as
tested infrastructure (Phase 3c), and (c) a small-to-large curriculum
(6×6 → 8×8 → 10×10) produces agents that discover the enclosure
mechanic ~2× earlier and end with +27 percentage points higher win
rate vs Random than direct training at equal self-play budget, with
complete rank separation across seeds on the first-enclosure metric
(Phase 3d). The archived reference agent is
`docs/phase3d/net_reference.pt`, git tag `v0.3d-reference-10x10-2p`.
