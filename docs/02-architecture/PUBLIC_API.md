# Public API

> The public Python API (this library has no REST/HTTP API beyond the local demo viewers).
> Grouped by package; signatures are indicative. Status is **Implemented** unless noted.
> Last audited 2026-05-28.

## Package root — `territory_takeover`
Exports (via `__init__.py`, with lazy `__getattr__` for the Gym wrappers):

| Symbol | Kind | Purpose |
|---|---|---|
| `new_game(board_size, num_players, spawn_positions=None, seed=None)` | fn | Construct a fresh `GameState`. Default spawns are symmetric corner insets (2 or 4 players). `seed` permutes spawn assignment. |
| `reset(state)` | fn | In-place reset reusing the grid buffer (training loops). |
| `step(state, action, strict=False) -> StepResult` | fn | Apply a move; resolve enclosures; advance turn. Illegal → player dies (default) or `IllegalMoveError` (strict). |
| `detect_and_apply_enclosure(state, player_id, placed_cell) -> int` | fn | Claim-resolution primitive (see caller contract in ARCHITECTURE). |
| `compute_terminal_reward(state, scheme)` | fn | Per-player terminal rewards ("sparse"/"relative"). |
| `GameState`, `PlayerState`, `StepResult` | class | Core state/result types. |
| `IllegalMoveError` | exc | Raised by `step(..., strict=True)`. |
| `simulate_random_rollout(state, rng)` | fn | In-place uniform playout to terminal. |
| `TerritoryTakeoverEnv`, `MultiAgentEnv` | class | Gym wrappers (lazy import; need `gymnasium`). |
| constants | — | `EMPTY`, `PATH_CODES`, `CLAIMED_CODES`, `DIRECTIONS`, etc. |

## Actions — `territory_takeover.actions`
- `legal_action_mask(state, player_id) -> (4,) bool` — for policy masking.
- `legal_actions(state, player_id) -> list[int]`.
- `has_any_legal_action(state, player_id) -> bool` — allocation-free short-circuit.
- `action_to_coord(state, player_id, action) -> (row, col)`.

## Agents — `territory_takeover.search`
- `Agent` (Protocol): `name: str`, `select_action(state, player_id, time_budget_s, max_iterations) -> int`, `reset()`.
- Agents: `UniformRandomAgent`, `HeuristicGreedyAgent`, `MaxNAgent`, `ParanoidAgent`,
  `UCTAgent`, `RaveAgent` (and `AlphaZeroAgent` in `rl.alphazero`).
- Search fns: `uct_search`, `rave_search`, `maxn_search`, `paranoid_search`.
- Registry: `registry.REGISTRY`, `AgentSpec.build(rng)`.

## Harness — `territory_takeover.search.harness`
- `play_game(agents, board_size, num_players, seed, max_turns) -> GameState`.
- `tournament(agent_a, agent_b, num_games, board_size, seed) -> dict` (wins/ties).
- `run_match(agents, num_games, board_size, num_players, seed, num_workers)` — timed, optional MP.
- `round_robin(...)` — all-pairs; result types `Table`, `AgentStats`, `PairRow`, `MatchResult`, `GameLog`.

## Evaluation — `territory_takeover.eval`
- `LinearEvaluator(weights)`: `evaluate(state)`, `evaluate_for(state, player_id)`; `default_evaluator()`.
- `voronoi_partition(state)`, `reachable_area(state, player_id)`.
- Feature fns: `mobility`, `reachable_area_feature`, `head_opponent_distance`, `claimed_count`,
  `path_length`, `territory_total`, `enclosure_potential`, `choke_pressure`.

## Rollout policies — `territory_takeover.search.mcts.rollout`
- `uniform_rollout`, `informed_rollout(state, rng, epsilon, evaluator)`,
  `voronoi_guided_rollout(state, rng, epsilon, k)`, `make_rollout(kind) -> RolloutFn`.

## RL — `territory_takeover.rl`
- **Tabular:** `TabularQAgent` (`select_action`, `td_update`, `save`, `load`), `QConfig`,
  `encode_state`/`decode_state`, `train_loop`, `eval_against_baseline`.
- **PPO:** `ActorCritic`, `PpoNetConfig`, `RolloutBuffer`/`compute_gae`/`get_batch`,
  `ppo_update_step`, `VectorizedEnv`, `encode_observation`/`apply_action_mask`. **Partial** — no training CLI.
- **AlphaZero:** `AlphaZeroNet`, `AZNetConfig`, `NNEvaluator` (`evaluate`, virtual-loss
  apply/revert), `puct_search`, `AlphaZeroAgent`, `ReplayBuffer`/`Sample`, `train_loop`
  (**gating Stubbed**), `encode_az_observation`, `grid_channel_count`, `scalar_feature_dim`.
- **Curriculum:** `CurriculumSchedule`, `make_schedule`, `CurriculumTrainer`, `transfer_weights`.
- **Elo:** `compute_elo(match_results, initial_elo, k_factor)`, `save_elo_csv`.

## Visualization — `territory_takeover.viz*`
- `viz`: `render_ascii(state)`, `render_matplotlib(state, show_heads=True)`,
  `save_game_gif(states, filename, fps=4)`, `check_invariants(state)`.
- `viz_html`: `save_game_html(states, agents, filename, win_probs=None, elo_csv=None)`,
  `heuristic_win_probs`, `alphazero_win_probs`, `AgentCard`.
- `viz_live`: `LiveServer`, `LiveConfig`, `play_and_serve`.
- `viz_interactive`: `InteractiveServer`, `AGENT_PRESETS`.

## Stability notes
- Core engine + actions + state are the **stable, build-on-this** surface (Production).
- RL `Partial`/`Stubbed` items (PPO training, AZ gating) may change; see
  [`docs/04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md).

## Related docs
- Data structures: [`DATA_MODEL.md`](DATA_MODEL.md) · Encodings: [`STATE_AND_ENCODING.md`](STATE_AND_ENCODING.md)
- CLI surface: [`docs/03-implementation/ENTRYPOINTS_AND_SCRIPTS.md`](../03-implementation/ENTRYPOINTS_AND_SCRIPTS.md)
