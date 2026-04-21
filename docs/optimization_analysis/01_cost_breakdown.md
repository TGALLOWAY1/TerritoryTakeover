# 01 — Cost Decomposition & Bottleneck Ranking

Source data:

- Engine micro-benchmarks: `benchmarks/optimized.json`,
  `benchmarks/baseline_pre_optimization.json`, `benchmarks/OPTIMIZATION_REPORT.md`.
- Method-level profiles (this task): `benchmarks/profiles/*.json` produced by
  `benchmarks/bench_pipeline.py`. Buckets are a **disjoint partition** over
  `tottime` (self time), so columns sum to the profiled wall time.

All times are cPython 3.11, CPU only, single process. AlphaZero profiles include
a pre-warm `train_loop(num_iterations=1, …, train_steps=1, games=1)` call to
force torch lazy imports before measurement — without it, `posix.stat /
marshal.loads / sympy.*` dominate the first-iteration profile at ~90 %.

## 1. Engine-level cost (pure `step`, no search)

`simulate_random_rollout` is the anchor — one function call per full game,
board-dominated cost only.

| Board | Games | Wall (s) | g/sec | enclosure % | numpy_ops % | stdlib % | step_glue % | legal % | state_copy % |
|-------|------:|---------:|------:|-------------:|-------------:|----------:|-------------:|---------:|--------------:|
| 10×10 |    50 |     0.12 |   410 |        63.9 |         16.1 |       6.7 |          2.3 |      1.8 |           0.4 |
| 20×20 |    50 |     0.81 |    62 |        73.7 |         17.8 |       5.4 |          0.6 |      0.4 |           0.1 |
| 40×40 |    30 |     2.80 |  10.7 |        76.3 |         17.8 |       5.1 |         0.2 |      0.1 |           0.0 |

`numpy_ops` here is almost entirely `numpy.ndarray.item()` calls made from
inside `detect_and_apply_enclosure` while walking the BFS boundary. If
enclosure were ported to C, the `numpy_ops` share would collapse with it.
Treating that as a conservative merge, `enclosure + numpy_ops + stdlib` runs
**86.7 % / 96.9 % / 99.2 %** at 10 / 20 / 40 — the BFS flood is not just the
top item, it **is** the cost at realistic board sizes.

## 2. Method-level cost — UCT (search-heavy)

UCT-vs-`RandomAgent`, uniform rollouts to terminal, single game end-to-end.

| Config | Wall (s) | enclosure % | numpy_ops % | stdlib % | step_glue % | rollout_driver % | mcts_* total % | state_copy % |
|--------|---------:|-------------:|-------------:|----------:|-------------:|-----------------:|----------------:|--------------:|
| 10×10, 200 iter/move | 10.3 | 57.7 | 14.6 | 3.6 | 7.5 | 4.1 | 4.4 | 0.8 |
| 20×20, 100 iter/move | 82.7 | 73.2 | 17.1 | 4.8 | 1.9 | 1.2 | 0.3 | 0.1 |

`mcts_*` totals (`selection + expansion + backprop + node_bookkeeping`) are
**4.4 % at 10×10, 0.3 % at 20×20**. The tree-policy math — UCB1 scoring, node
allocation, visit-count updates — is negligible compared to the rollouts it
spawns. At 20×20 the profile degenerates to the engine profile: flooding
dominates, everything else is noise.

Takeaway: **search overhead is free**. If you want UCT to be faster you do
not optimise the selection policy, the node object, or the backprop — you
make the rollout cheaper or shorter. That splits two ways:

- **Short rollouts** (horizon truncation + leaf heuristic or NN value) —
  orthogonal to engine speed, 3–10× depending on horizon.
- **Fast rollouts** (Cython-port the BFS) — 3–5× on the engine side.

They compose.

## 3. Method-level cost — AlphaZero one iteration

Self-play (4 games, 8 PUCT iterations/move) + 40 train steps at batch 32,
on a 32-channel / 2-res-block conv-head net matching the Phase 3d
curriculum production scale.

| Board | Wall (s) | torch_ops % | torch_nn % | enclosure % | numpy_ops % | mcts_puct % | evaluator_cache % | nn_train % | replay % |
|-------|---------:|-------------:|------------:|-------------:|-------------:|-------------:|--------------------:|------------:|----------:|
| 6×6   |     1.08 |         62.7 |        12.5 |          2.5 |          2.0 |          3.1 |                2.6 |         1.2 |       0.2 |
| 10×10 |     2.55 |         53.1 |        10.8 |          9.7 |          4.1 |          3.7 |                3.3 |         0.5 |       0.2 |

At 6×6 the **net forward/backward (`torch_ops + torch_nn + nn_module`)
accounts for ~78 %**; self-play env cost (enclosure + numpy_ops + step_glue)
is **< 6 %**. At 10×10 env cost climbs to ~16 % while net cost falls to
~67 %. Extrapolating the trend linearly, the crossover happens somewhere
around board=16–20 when the net is this small; at the 32-channel scale
**the engine is not the bottleneck for AlphaZero training at
production-stage boards ≤ 15×15**.

Three non-obvious observations from the AZ profiles:

1. **`mcts_puct` is 3–4 %.** PUCT tree policy — the thing that distinguishes
   AlphaZero from plain batched inference — is a rounding error compared to
   the net forward it calls. Optimizing the search loop is not where budget
   goes.
2. **`evaluator_cache` is 3 %, a win already landed.** LRU cache lookup cost
   is visible as a named bucket and is *smaller* than the net forward cost
   it replaces on cache hits. The cache is paying for itself.
3. **`replay_buffer + observation_encode` = 1.7 %.** Data wrangling is
   cheap. If we later add a per-step reward column (Chunk 6), the cost is
   comfortably below the margin of measurement noise.

## 4. Engine-level anchor (from `benchmarks/optimized.json`)

Reconfirms the existing engine snapshot, unchanged by this task:

- `legal_actions`: ~0.6 µs/call (target < 1 µs — met).
- `GameState.copy()`: 6.2 µs at 40×40 (target < 50 µs — met by an order
  of magnitude).
- `step` no-claim: 2.3 µs. `step` with claim BFS: 24 µs at 40×40 avg
  (dominated by the BFS flood, as `HOTSPOTS.md` records).
- 40×40 rollout throughput: 1152 games/min ≈ 52 ms/game. The 76 % of this
  consumed by `detect_and_apply_enclosure` is the already-known hotspot.

## 5. Ranked bottleneck list

Ranking answers the task's A–F decomposition. The ranking is **dominant
over the productive workload** (i.e. what a user running the training stack
would actually cut against their wall-clock budget), not over arbitrary
micro-benchmarks.

### Rank 1 — F: Method/game mismatch (AlphaZero value target ignores dense reward)

Evidence:

- `src/territory_takeover/rl/alphazero/selfplay.py:59–66, 152–155` stores only
  the **normalized final score vector** `2*score − 1 ∈ [−1, 1]` as the
  per-sample value target.
- `src/territory_takeover/rl/alphazero/train.py:118–122` regresses the value
  head against that one terminal label for every state in the trajectory.
- `src/territory_takeover/engine.py:461` already emits the per-step reward
  `1.0 + claimed_this_turn` — the **engine already knows** how many cells
  the player just claimed. Tabular Q (`rl/tabular.py`) and PPO
  (`rl/ppo.py`) both consume this dense signal. Only AlphaZero does not.
- Games reach 300 half-moves on 40×40 (`configs/phase3d_curriculum_fast.yaml`
  `max_half_moves: 480`). Regressing 300 states against one scalar each
  gives `O(T)` target copies of the same label — the variance is enormous
  and scales with game length.
- Phase 3d lab notes (`KEY_FINDINGS.md`): the curriculum beat direct
  training 0.708 vs 0.438 win rate (Cohen's d=+2.01) mainly because the
  curriculum **shortens the effective episode**. That is a workaround for
  the variance problem, not a fix.

This is a learning-signal defect, not a throughput defect. Cutting
training wall-clock by 3× (Rank 2) does not help if the gradient points
the wrong way. **This is the bottleneck to fix first.**

### Rank 2 — A: Engine throughput (Python BFS in `detect_and_apply_enclosure`)

Evidence:

- 63–76 % of `simulate_random_rollout` tottime at 10/20/40 (§1).
- 57–73 % of UCT-vs-Random tottime at 10/20 (§2). Includes the rollout
  cost at 20×20 where it effectively is the profile.
- Only 2.5 / 9.7 % of AlphaZero-one-iteration at 6 / 10 (§3); this becomes
  Rank 1 for AlphaZero only when boards reach ≥ 20.
- Already flagged as the Cython candidate in
  `benchmarks/HOTSPOTS.md` and `benchmarks/OPTIMIZATION_REPORT.md §D1`.
  87 % of trigger fires claim zero cells — an early-out in C avoids
  allocating the visited mask and the deque entirely.

Realistic payoff: **3–5× on rollout-dominated workloads** (UCT at scale,
pure self-play benchmarks). Does **not** meaningfully help AlphaZero
training at board ≤ 15 where the net is the bottleneck.

### Rank 3 — B: Depth to terminal (rollouts run to terminal)

Evidence:

- `src/territory_takeover/search/mcts/rollout.py:99, 263, 320` — all three
  rollout policies loop `while not state.done`. No horizon parameter, no
  leaf evaluator called when time runs out.
- `configs/phase3d_curriculum_fast.yaml` caps at 480 half-moves at 40×40.
  A PUCT tree with rollouts running to move 480 at every leaf is
  structurally expensive even if the engine is free.
- Horizon truncation with a leaf value estimator is a standard,
  well-understood UCT extension — turns Rank 3 into Rank-2's alternative
  fix without needing the Cython port first.

Payoff: 3–10× on any rollout-using search workload depending on horizon
choice. Composes with Rank 2.

### Rank 4 — C: Reward sparsity (terminal-only target)

Subsumed by Rank 1 for AlphaZero. Tabular Q and PPO already see dense
reward; they are not affected by this. The fix is the same fix as Rank 1
(emit the engine's per-step reward into the training target).

### Rank 5 — D: Evaluator cost (Voronoi + reach BFS + enclosure potential)

Evidence:

- `src/territory_takeover/eval/heuristic.py` plus
  `src/territory_takeover/eval/voronoi.py` compute a four-player multisource
  BFS for distance, a reach-BFS for each player, and an enclosure-potential
  BFS per empty region.
- Not on the critical RL path (no heuristic evaluator in the AZ profile
  except indirectly through greedy opponents in tournaments).
- Relevant for Max-N / Paranoid agents at high depth. Separate optimisation
  lane.

Not part of the main recommendation; listed for completeness. A Cython port
of the three BFS passes would drop Max-N-d3 eval time ~5× but the RL loop
does not call this.

### Rank 6 — E: State representation (`state.copy()`)

Evidence:

- 6.2 µs at 40×40 (§4). 0.1–0.8 % of any profile at any scale.
- Already a numpy-memcpy on the grid plus shallow-copy of per-player
  structs; the numpy memcpy is the dominant cost inside that.

Not a bottleneck. Only listed because the task prompt named it.

## 6. One-sentence ranking

**F > A ≈ B > C > D > E.** F is the method/game mismatch that current
training is paying for in variance and sample count; A and B are
throughput bottlenecks that compose and unlock larger board scaling; C
disappears once F is fixed; D is scoped to Max-N; E is not a bottleneck.
