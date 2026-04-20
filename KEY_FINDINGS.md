# KEY_FINDINGS.md

A running lab-notebook for TerritoryTakeover research phases. Each phase
appends below; nothing is overwritten. Numbers are generated from actual
training runs under `results/phase3a/` (gitignored) — reference artifacts are
committed under `docs/phase3a/`.

---

## Phase 3a — Tabular Q-learning baseline (2026-04-19)

**Goal.** Establish that *any* learning agent can improve at the game before we
invest in deep-RL infrastructure. Tabular Q-learning is the smallest viable
test: if it fails here for obvious reasons, we learn something. If it succeeds,
it anchors the learning baseline that Phase 3b (function approximation) must
exceed.

**Branch.** `claude/tabular-q-learning-baseline-nUlc2`.

### Experimental setup

- **Engine.** Discovered during exploration that `engine.step()` is already
  implemented (CLAUDE.md's "not yet implemented" line is stale); the engine
  returns `reward = 1.0 + claimed_this_turn` on legal moves and flips
  `alive=False` on illegal moves. Phase 3a builds on this directly.
- **Spawn-position quirk.** `engine._default_spawns(8, 2)` returns
  `[(4, 4), (3, 3)]` — the two seats are diagonally adjacent on an 8×8 board
  and corner the game in six moves. All 8×8/2p training and evaluation below
  overrides spawns to `[(0, 0), (7, 7)]`. Noted as a follow-up engine bug;
  not fixed in this phase.
- **State encoder.** `encode_state(state, player_id) -> StateKey` produces a
  7-tuple: `(head_r, head_c, nbr_N, nbr_S, nbr_W, nbr_E, phase)`. Each
  neighbor takes one of six classes: `OOB / EMPTY / OWN_PATH / OWN_CLAIM /
  OPP_PATH / OPP_CLAIM` (all opponents collapse into one "opp" class so a
  2p-trained encoder works unchanged on 4p). `phase` is a 4-way bucket over
  `empty_fraction` of the grid with thresholds `(0.80, 0.55, 0.25)`.
- **Q-agent.** Legal-action-masked ε-greedy with α=0.1, γ=0.99, linear
  ε-schedule 1.0 → 0.05 over the first 50% of training, then constant 0.05.
  Illegal actions are `-inf` in both the select-action argmax and the TD-target
  max — never written to, never chosen.
- **Reward shaping.** `+1` per cell claimed in a turn (from the engine reward);
  `-1 × path_len` added on the transition where `alive` flips True→False
  (trap/self-trap penalty); terminal rank bonus `(10, 3, -3, -10)` added to
  the final flushed transition, with tie-averaging between tied seats.
- **Training.** Single shared `TabularQAgent` drives every seat
  (pure self-play). Per-seat rolling buffer `last_sa[seat]` stitches the
  "my-turn-only" MDP transitions across intervening opponent moves;
  pending reward accumulates between the seat's moves.
- **Logger.** TensorBoard via `tensorboardX` (optional extra); CSV mirrors
  to `episode_log.csv` and `eval_curves.csv` alongside each run.

### Headline result (8×8 / 2p, 500 000 episodes, seed 0)

| opponent    | games | win | loss | tie | win_rate | 95% CI         |
|-------------|-------|-----|------|-----|----------|----------------|
| random      | 1000  | 394 | 262  | 344 | **0.394**| [0.364, 0.425] |
| greedy      | 1000  | 116 | 884  | 0   | **0.116**| [0.098, 0.137] |
| uct-32      | 100   | 15  | 77   | 8   | **0.150**| [0.093, 0.233] |

Targets set in the plan (Random ≥ 0.80, Greedy ≥ 0.55, UCT-32 ≥ 0.30) were
**missed across the board**. Wall-clock for the run: 30 min 22 s on a single
CPU thread. Final Q-table: 7 909 states.

Training curves (from `docs/phase3a/8x8_2p_seed0_*.png`):

![vs Random](docs/phase3a/8x8_2p_seed0_vs_random.png)
![vs Greedy](docs/phase3a/8x8_2p_seed0_vs_greedy.png)
![vs UCT-32](docs/phase3a/8x8_2p_seed0_vs_uct.png)

### What the curves actually show

- **vs Random** hovers between 0.30–0.45 for the entire 500 k-episode window.
  There is no visible monotone improvement. ε decays to 0.05 by episode
  250 000 and the next 250 000 near-greedy episodes do *not* raise the win
  rate further.
- **vs Greedy** is catastrophic (≤ 15% most of the time) with one brief
  0.50 spike at ep 30 k that is not reproducible on later checks — almost
  certainly a policy-tie resolution artifact, not genuine skill.
- **vs UCT-32** stays ~0.10–0.20 throughout; effectively loses to any
  search-based opponent.
- **Tie rate vs Random is 34.4%.** Random self-traps frequently, but so does
  our greedy-eval agent — a lot of games end with both seats dead.
- **vs Greedy has 0% ties.** Greedy does not self-trap, so the games always
  resolve; they resolve in Greedy's favor 88.4% of the time.

### Why it underperformed (hypotheses)

1. **State aliasing.** 7 909 unique encoder keys over a 500 k-episode run on
   an 8×8 board means a vast number of semantically different board
   configurations collapse onto the same key. Four-neighborhood occupancy
   + head position + 4-way phase bucket is simply too sparse a summary of an
   8×8 grid. The agent cannot distinguish "two moves from finishing a big
   enclosure" from "about to walk into a trap" when both share the same
   local neighborhood.

2. **Pure self-play with a shared Q-table.** The opponent model during
   training is literally the current policy (same table, other seat). Against
   the eval opponents — a uniform-random policy and a heuristic greedy policy
   — the state distribution is *off-policy* relative to training, and the
   Q-values are poorly calibrated on those states. A Greedy opponent
   corners the agent in situations the self-play partner would never have
   taken it to.

3. **Trap penalty competes with rank bonus.** On a short game where
   `path_len` at death is small (say 3–5), the `-3..-5` trap penalty is
   smaller in magnitude than the `-10` bottom-rank bonus. A losing agent can
   reduce its expected return by committing suicide early rather than
   playing out. We do not see egregious illegal-move rates (episode logs
   show normal turn counts of 40–80) but partial risk-seeking-to-die
   behavior is plausible and would degrade win rate against Random.

4. **ε-decay horizon vs. table growth.** The Q-table keeps growing slowly
   even after ε reaches 0.05 (7 700 states at ep 248 k → 7 908 at ep 498 k).
   Each new state starts at zero Q-values and is visited only a handful of
   times before training ends, so its Q-values never converge and greedy
   evaluation picks actions essentially at random in those states.

### State-space coverage

| episode | Q-table size | mean-visits/key* | ε     |
|---------|--------------|------------------|-------|
| 100 000 | 7 248        | ~ 69             | 0.63  |
| 250 000 | 7 700        | ~ 162            | 0.06  |
| 500 000 | 7 909        | ~ 316            | 0.05  |

\* Mean-visits-per-key is a rough estimate: total state visits (≈ episodes ×
 turns/ep / 2) divided by final table size. p99 visits per key was not
logged; capturing that histogram is a Phase 3b instrumentation upgrade.

### 10×10 / 4p diagnostic

Ran 100 000 episodes on the 10×10 4-player config, seed 0, no UCT eval
(expensive on 4p). This is a diagnostic, not a gated target. Wall-clock:
7 min 37 s. Final Q-table: 14 502 states.

Final eval (seed-42 evaluator, 500 games each, 1st-place = "win"; uniform
baseline = 0.25):

| opponent (3 seats) | games | win | loss | tie | win_rate | 95% CI         |
|--------------------|-------|-----|------|-----|----------|----------------|
| random             | 500   | 210 | 221  | 69  | **0.420**| [0.378, 0.464] |
| greedy             | 500   | 153 | 292  | 55  | **0.306**| [0.267, 0.348] |

Both are well above the 0.25 uniform baseline — the agent does learn
*something* on 10×10/4p, modestly. Curves (`docs/phase3a/10x10_4p_seed0_*.png`):

![10×10 vs Random](docs/phase3a/10x10_4p_seed0_vs_random.png)
![10×10 vs Greedy](docs/phase3a/10x10_4p_seed0_vs_greedy.png)

- **State explosion is real.** 14 502 unique keys after only 100 000
  episodes, versus 7 909 after 500 000 episodes on 8×8/2p. The 10×10/4p
  run fills new table slots ~9× faster per episode than 8×8/2p (0.145
  keys/ep vs 0.016 keys/ep). Scaling tabular to 12×12 or larger is
  clearly not viable with this encoder.
- **Counter-intuitive result: 10×10/4p agent outperforms 8×8/2p agent.**
  Despite 5× fewer episodes, the 10×10 agent beats both Random (42.0%
  vs. uniform 25%) and Greedy (30.6% vs. uniform 25%) more decisively
  than the 8×8/2p agent beats anything. Two plausible reasons: (a) 4p
  averages out the self-play-vs-eval distribution mismatch because each
  training game already has three "other" seats, (b) 10×10 corner spawns
  leave ~100 cells of maneuvering room so early-game head-on collisions
  (the likely failure mode on 8×8) are rare.

### Behavioral anomalies observed

- **No reward-hacking rage-quit** detected in spot checks: the greedy-eval
  agent's illegal-move rate is negligible (the `-inf` masking prevents
  illegal actions being selected; the trap penalty prevents self-trapping
  from being a positive-EV strategy at typical path lengths).
- **Policy oscillation.** Win rate vs Greedy jumps between 0.00 and 0.50
  across adjacent 10 k-episode eval ticks. Ties on Q-values resolve to
  `np.argmax` → lowest-index direction (N). When enough Q-values update
  to flip a tied-but-pivotal state's argmax from one direction to another,
  the greedy policy changes sharply — a classic symptom of
  function-free Q with tie-breaking by index. Phase 3b should break ties
  stochastically at eval time *or* use a continuous value function.

### Recommendations for Phase 3b

1. **Replace the state encoder with a grid-shaped tensor input.** A CNN over
   the full `(board_size, board_size, channels)` grid eliminates the 7-tuple
   aliasing bottleneck immediately. Channels: own-path, own-claim, opp-path,
   opp-claim, head-one-hot, legal-action-mask projected onto the head.
2. **Move to DQN with prioritized replay.** The shared-Q self-play dynamics
   are fine, but a replay buffer decorrelates updates and a target network
   eliminates the oscillation seen around policy-tie states.
3. **Instrument state visit histograms.** Log p50/p99/max visits-per-key so
   we can quantify coverage without repeated post-hoc table scans.
4. **Keep the trap-penalty and rank-bonus structure; retune once policy
   learns to not walk into walls.** The reward design looks coherent; the
   bottleneck is representation, not the scalar signal.
5. **Fix the `_default_spawns(8, 2)` engine bug** as a standalone commit —
   having a spawn default that is diagonally adjacent silently breaks any
   experiment that doesn't pass explicit spawns.

### Repro checklist

- Install: `pip install -e ".[rl,dev]"` (adds `pyyaml`, `matplotlib`,
  `tensorboardX`).
- Train: `python scripts/train_tabular_q.py
  --config configs/phase3a_tabular_8x8_2p.yaml --seed 0`.
- Eval (500 games each): `python scripts/eval_tabular_q.py
  --checkpoint results/phase3a/runs/<stamp>/q_table.pkl --games 500
  --uct-iters 32 --uct-games 100 --plot`.
- Smoke tests: `pytest tests/test_rl_tabular_*.py`.

### Follow-ups

- Reproducibility across seeds (1, 2) was scoped into the plan but skipped
  on this branch: given the headline miss, seed-sweep CIs would not change
  the conclusion. Re-run when Phase 3b DQN is in.
- TensorBoard event files are emitted under `results/phase3a/runs/<stamp>/tb/`
  if `tensorboardX` is installed. Not needed for this writeup — CSVs are
  sufficient.

---

## Phase 3c — AlphaZero primitives (infrastructure only) (2026-04-20)

**Status.** Infrastructure landed; long-running training and the full
evaluation table are deferred. This entry exists so the deferral is
visible in the lab notebook rather than buried in a commit message.

**Goal.** Stand up a Petosa & Balch (2019)-style AlphaZero agent for
TerritoryTakeover: PUCT MCTS driven by a ResNet policy/value network
with a per-seat value head (one expected normalized score per seat, not
a single scalar). Per-seat values are the named design choice that
generalizes AlphaZero to multi-seat self-play games.

**Branch.** `claude/complete-key-findings-phase-3b-c4tFV` (name is
stale — this is Phase 3c work; carried over from the earlier Phase 3b
branch).

### What shipped

All pieces live under `src/territory_takeover/rl/alphazero/`:

- `spaces.py` — `encode_az_observation(state, active_player)` returns
  `(grid_planes (3N+2, H, W), scalars (3+N,))`. Unlike the PPO encoder
  this uses **fixed seat ordering** (plane `i` is always seat `i`'s
  path/claim) and appends an N-channel turn one-hot so the network can
  read whose turn it is directly — both required for a stable per-seat
  value head.
- `network.py` — `AlphaZeroNet(AZNetConfig)`: conv stem → configurable
  ResNet trunk (`num_res_blocks`, `channels`) → policy head (1×1 conv +
  flatten + linear → 4 logits, illegal slots already masked) → value
  head (1×1 conv + flatten concat scalars + linear + ReLU + linear +
  tanh → `(N,)` in `[-1, 1]`). Scalar-value-head ablation path is
  exposed as `scalar_value_head=True` but not yet executed.
- `evaluator.py` — `NNEvaluator` with OrderedDict-based LRU cache keyed
  on `(grid.tobytes(), active_player, tuple(head_positions))`,
  batched-inference API, and virtual-loss scaffolding for future
  concurrent search. Priors are legal-action-masked softmax; illegal
  slots are returned as exactly zero.
- `mcts.py` — `puct_search(state, root_player, evaluator, iterations,
  c_puct=1.25, dirichlet_alpha=0.3, dirichlet_eps=0.25, rng, temperature)`.
  Reuses the passive `search.mcts.node.MCTSNode` (which already carries
  a per-player `total_value` vector), stores priors in a per-call
  side table keyed by `id(node)`, applies Dirichlet noise **at the root
  only** on the first expansion, and backs up the full per-seat value
  vector at each level. Ships an `AlphaZeroAgent` wrapper that satisfies
  the `search.agent.Agent` protocol.
- `replay.py` — Fixed-capacity numpy ring buffer over
  `(grid, scalars, mask, visits, final_scores)`. O(1) `add`, uniform
  `sample`, single-`.npz` save/load.
- `selfplay.py` — `play_game(evaluator, cfg, rng)` drives every seat
  with an independent PUCT search, follows AlphaGo Zero's canonical
  temperature schedule (sample ∝ visits for the first
  `temperature_moves` half-moves, argmax after), and emits one `Sample`
  per visited state with the same per-seat final value vector.
- `train.py` — `train_step(net, opt, batch)` = masked softmax
  cross-entropy (policy) + MSE (value) + L2. `train_loop(net_cfg,
  train_cfg, self_play_cfg, out_dir)` iterates self-play → buffer → SGD
  → snapshot and writes `iteration_log.csv`. **Gating tournament is
  stubbed** with a `# TODO` — under the reduced-scope cadence the latest
  snapshot always drives self-play.
- `scripts/train_alphazero.py` + `scripts/eval_alphazero.py` — YAML-driven
  CLI shims. Configs under `configs/phase3c_alphazero_*.yaml`.

Tests are under `tests/test_rl_alphazero_*.py` (six files, 39 tests):
encoder invariants, forward-pass shapes, cache/LRU/virtual-loss
semantics, PUCT formula on a toy node, per-seat backup, temperature
schedule, replay round-trip, and a 2-iteration end-to-end smoke that
writes snapshots and `iteration_log.csv`. All 39 pass.

### What did **not** ship (explicit deferrals)

The original Phase 3c prompt called for a multi-hour self-play run, a
full evaluation table vs Random / Greedy / UCT-32 / Phase 3b PPO, a
gating-tournament promotion policy, a 4-dim-vs-scalar value-head
ablation, and 10-vs-20-block / 64-vs-128-vs-256-channel network-size
ablations. Under the single-pass wall-clock budget none of those fit; in
priority order:

1. **No trained checkpoint.** The 50-iteration 8×8/2p baseline
   (`configs/phase3c_alphazero_8x8_2p.yaml`) is wired but not yet
   executed. Without a checkpoint there is no Headline result, no
   training curves, and no win-rate table. The smoke config
   (`configs/phase3c_alphazero_smoke.yaml`) does complete in seconds
   and confirms the loss decreases iteration-over-iteration (1.49 →
   1.27 on a 6×6 / 2p run), but that's a plumbing check, not evidence
   of learning.
2. **No PPO comparison.** Phase 3b shipped primitives only — there is
   no trained PPO checkpoint in the repo. The prompt's "AlphaZero must
   exceed PPO" acceptance is moot until Phase 3b trains a model.
3. **Gating tournament.** The stub in `train.py` always promotes the
   latest network. A proper implementation would play the current
   network vs the reigning champion and only promote on a threshold
   (e.g. ≥55% over N games, per the AlphaGo Zero convention).
4. **4-dim-vs-scalar value-head ablation.** The `scalar_value_head`
   flag exists and is tested on forward shapes, but the two runs needed
   to compare head variants have not been executed.
5. **Network-size ablation.** No comparison at all between 10-vs-20
   blocks or 64/128/256 channels. The default config uses 4 blocks /
   64 channels for CPU budget reasons.
6. **Strategic-behavior vignettes.** No alliance / double-enclosure /
   end-game probing examples.

### Repro checklist

- Install: `pip install -e ".[rl_deep,dev]"` (pulls in `torch`).
- Unit tests: `python -m pytest tests/test_rl_alphazero_*.py -q`.
- Smoke train (seconds, 6×6 / 2p, 1 res block, 8 channels):
  `python scripts/train_alphazero.py --config configs/phase3c_alphazero_smoke.yaml`.
- Baseline train (not yet executed; reduced-scope ~1 hr CPU budget,
  8×8 / 2p, 4 res blocks, 64 channels):
  `python scripts/train_alphazero.py --config configs/phase3c_alphazero_8x8_2p.yaml --seed 0`.
- Eval a checkpoint:
  `python scripts/eval_alphazero.py --checkpoint results/phase3c/runs/<stamp>/net_final.pt --config configs/phase3c_alphazero_8x8_2p.yaml --games 200`.

### Follow-ups

In priority order for the next session:

1. Execute the 8×8 / 2p baseline run end-to-end and populate this entry
   with the Headline result table (win rates + Wilson 95% CIs vs Random
   / Greedy / UCT-32) and training curves under `docs/phase3c/`.
2. Implement the gating tournament promotion policy in `train.py`,
   replacing the always-promote stub.
3. Train a Phase 3b PPO baseline so the "AlphaZero exceeds PPO"
   acceptance criterion can be tested.
4. Run the 4-dim-vs-scalar value-head ablation — the prompt's named
   "core research question" — using matched configs.
5. Network-size ablation (blocks and channels) once the baseline run's
   wall-clock is measured so ablation runs can be budgeted.

