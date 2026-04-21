# 03 — Strategy Comparison

Three candidate directions for where to spend the next block of
engineering budget. Scored against the bottleneck ranking in
`01_cost_breakdown.md` (F > A ≈ B > C > D > E).

## Option 1 — Search-heavy

Push the existing UCT/AlphaZero search further: Cython-port the
enclosure BFS, add horizon truncation + leaf value estimator, tune PUCT
iteration counts per curriculum stage. No learning-signal changes.

| Axis | Score |
|------|-------|
| Addresses bottleneck rank | A (Cython BFS) + B (horizon cutoff). Leaves F untouched. |
| Sample efficiency | Unchanged. Same trajectories, same value target. |
| Implementation complexity | **High.** Cython build step, new C source file, CI integration, `HOTSPOTS.md` contract, behavioural-equivalence tests against the numpy implementation. |
| Runtime cost payoff | **3–5× on rollout-dominated workloads** (pure UCT at scale). 10–15 % payoff on AlphaZero training at board ≤ 15 (net dominates). |
| Scalability to 40×40 | Strong for search; modest for AlphaZero training. |
| Compatibility with engine | Requires engine ABI stability — pythonic `GameState` must either expose a Cython cdef or pass the numpy grid through a C hot path. |
| Residual bottleneck after | F unchanged. Training still regresses against terminal-only labels. |
| Risk | Cython bugs silently corrupt enclosures — needs extensive equivalence testing. |

**Verdict:** right engineering step for production self-play throughput
at 40×40, wrong step for fixing the method mismatch.

## Option 2 — RL-heavy (dense reward via PPO scaling)

Lean on PPO (already dense-reward-aware) instead of AlphaZero. Scale
up PPO: parallel envs, longer training, GAE tuning. Drop the value
network + MCTS stack during training.

| Axis | Score |
|------|-------|
| Addresses bottleneck rank | F by substitution (PPO already uses the dense reward), at the cost of giving up AlphaZero's search-based label generation. |
| Sample efficiency | **Poor.** On-policy RL on 100-to-300-step episodes on 40×40 is ~10–100× less sample-efficient than AlphaZero at comparable wall clock. |
| Implementation complexity | **Medium.** PPO infrastructure exists (`rl/ppo.py`), but parallel env runners are not built. |
| Runtime cost payoff | Negative vs AlphaZero at matched strength. PPO would need ~10M env steps to match what AlphaZero reaches in 1M. |
| Scalability to 40×40 | Poor. 40×40 random-play episodes are ~300 half-moves × 52 ms/game (`OPTIMIZATION_REPORT.md §E`), an on-policy buffer of millions is wall-clock-prohibitive. |
| Compatibility with engine | Good — engine already emits the reward. |
| Residual bottleneck after | A and B both become visible once PPO replaces AlphaZero — no tree means no amortized search. |
| Risk | Phase 3b PPO ablations (if they had run) would have demonstrated this; the fact that the curriculum team moved to AlphaZero in Phase 3c is confirmatory. |

**Verdict:** engineering leverage is wrong. PPO's real role in this
codebase is as the dense-reward baseline for ablations, not as the main
training stack.

## Option 3 — Hybrid (recommended)

Three changes, composed:

1. **AlphaZero n-step value target** (Chunk 6) — fixes Rank 1 (F).
2. **Horizon-truncated rollouts + NN leaf value** — fixes Rank 3 (B)
   inside AlphaZero itself, without requiring a Cython port.
3. **Cython BFS port as follow-up** (scheduled, not in this task) —
   fixes Rank 2 (A) when board scales past 15×15.

| Axis | Score |
|------|-------|
| Addresses bottleneck rank | F directly (§1), B indirectly via AZ's already-truncated self-play (max_half_moves is a built-in rollout cap), A deferred. |
| Sample efficiency | **Strongly positive.** Lower-variance value targets reduce training steps to a given strength by 2–5× in comparable RL settings (A3C, Rainbow, MuZero). |
| Implementation complexity | **Low.** Chunk 6 touches ~3 files, adds 2 columns to the replay buffer, adds one new config, adds one test file. No C code. |
| Runtime cost payoff | Replay buffer grows by 2 fp32 columns per sample → 8 bytes × buffer_capacity. At default capacity=50k, that is 400 KB. Net forward cost for bootstrap `V̂(s_{t+n})` is 1 extra evaluator batch per training step → ≤ 2 % wall clock. |
| Scalability to 40×40 | Fine. Per-step reward divided by board area stays scale-invariant. |
| Compatibility with engine | Perfect — uses the existing `claimed_this_turn` delta unchanged. |
| Residual bottleneck after | A at board ≥ 20 (fixed by the deferred Cython port), D for Max-N baselines (separate optimisation lane). |
| Risk | Low. Flag-gated, default behaviour unchanged, 39 existing tests continue to pass. |

**Verdict:** this is the right next step. Minimal code change, fixes
the worst bottleneck, leaves the orthogonal engine-throughput bottleneck
(A) to the engineering step it was always going to take (the Cython
port already noted in `HOTSPOTS.md`).

## Cross-option summary

| Axis | Opt 1 (Search) | Opt 2 (PPO) | Opt 3 (Hybrid) |
|------|:---:|:---:|:---:|
| Fixes F | — | ✓ | ✓ |
| Fixes A | ✓ | — | deferred |
| Fixes B | ✓ | — | partial |
| Sample efficiency gain | 0 | -10× | +2–5× |
| Engineering cost | High | Medium | Low |
| Risk | Medium (Cython) | High (RL-eng) | Low |
| Default behaviour preserved? | No (Cython binary req'd) | n/a (new stack) | **Yes (flag-gated)** |

## Why not all three at once

Opt 1 and Opt 3 compose. A realistic 2-quarter plan is:

1. **Now (this task):** Opt 3 — land the n-step target prototype on the
   existing AlphaZero stack. Run the 10×10 ablation in
   `04_experiment_plan.md` to confirm sample efficiency gain.
2. **Next:** Cython BFS port (Opt 1's main win). `HOTSPOTS.md` already
   scopes this; it was always the planned follow-up.
3. **Then:** Scale curriculum to 15×15/4p using the combined
   dense-target + fast-BFS stack.

Running Opt 1 and Opt 3 in parallel in the same branch is tractable if
two developers split the work (Opt 1 is in `engine.py` + a new `.pyx`,
Opt 3 is in `rl/alphazero/*`), but mixing them in a single PR blurs
whether any observed win is from the learning-signal fix or the
throughput fix. Run them in sequence.

Opt 2 is not on the sequence. It is reserved as an ablation oracle:
"does a dense-reward RL method outperform the current AlphaZero on the
same budget" is a useful sanity check but not a production direction.
