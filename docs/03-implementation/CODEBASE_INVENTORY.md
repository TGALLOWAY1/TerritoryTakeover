# Codebase Inventory

> Module-by-module inventory of `src/territory_takeover/` (54 modules, ~13k LOC).
> This is the library equivalent of a component inventory. Status labels:
> **Implemented / Partial / Stubbed / Broken / Designed only / Deprecated / Unknown**.
> Evidence is the source path itself. Last audited 2026-05-28.

## Core engine

| Module | LOC | Status | Public surface |
|---|---:|---|---|
| `constants.py` | 40 | Implemented | `EMPTY`, `PATH_CODES`, `CLAIMED_CODES`, `DIRECTIONS` (N,S,W,E), per-player `PLAYER_n_PATH/CLAIMED`, default board/player sizes. All `Final`-typed. |
| `state.py` | 152 | Implemented | `PlayerState` (path/path_set/head/claimed_count/alive), `GameState` (grid + players + counters + scratch buffers); `GameState.empty()`, `.copy()` (cheap, tree-search-tuned), `.__repr__()` ASCII. |
| `actions.py` | 89 | Implemented | `legal_action_mask` → `(4,)` bool, `legal_actions` → list, `has_any_legal_action`, `action_to_coord`. Hot-path uses `grid.item(r,c)`. |
| `engine.py` | 517 | Implemented (1 deferred branch elsewhere) | `new_game`, `reset`, `step` → `StepResult`, `detect_and_apply_enclosure`, `compute_terminal_reward`, `IllegalMoveError`, `_default_spawns`, `_compute_winner`. Includes `_legacy_..._full_bfs` reference impl for tests. |
| `rollout.py` | 114 | Implemented | `simulate_random_rollout(state, rng)` — in-place play to terminal under uniform policy; returns half-move count. |

## Gym integration

| Module | LOC | Status | Public surface |
|---|---:|---|---|
| `gym_env.py` | 436 | Implemented | `TerritoryTakeoverEnv` (Gymnasium single-agent wrapper; auto-plays opponents; Dict obs `grid/current_player/heads/action_mask`; `Discrete(4)`; "ansi"/"rgb_array" render), `MultiAgentEnv` (PettingZoo-AEC-style, duck-typed). Lazy-imported via top-level `__getattr__`. |

## Evaluation (`eval/`)

| Module | LOC* | Status | Public surface |
|---|---:|---|---|
| `eval/features.py` | — | Implemented | Feature fns: `mobility`, `reachable_area_feature`, `head_opponent_distance`, `claimed_count`, `path_length`, `territory_total`, `enclosure_potential`, `choke_pressure`. |
| `eval/voronoi.py` | — | Implemented | `voronoi_partition(state)` multi-source BFS → owner grid; `reachable_area(state, pid)`. |
| `eval/heuristic.py` | — | Implemented | `LinearEvaluator` (weighted feature sum; `DEAD_SENTINEL=-1e6`), `default_evaluator()`, internal `_FeatureCache`. |
| `eval/tuning.py` | — | Implemented | `tune_weights(...)` evaluator-weight optimization. |

\* `eval/` totals ~1,056 LOC.

## Classical search (`search/`)

| Module | LOC* | Status | Public surface |
|---|---:|---|---|
| `search/agent.py` | 38 | Implemented | `Agent` Protocol: `name`, `select_action(state, pid, time_budget_s, max_iterations)`, `reset()`. |
| `search/random_agent.py` | ~115 | Implemented | `UniformRandomAgent`, `HeuristicGreedyAgent` (one-ply greedy). |
| `search/maxn.py` | ~354 | Implemented | `maxn_search`, `paranoid_search` (alpha-beta 2p reduction), `MaxNAgent`, `ParanoidAgent` (iterative deepening). |
| `search/registry.py` | 110 | Implemented | `REGISTRY` name→constructor, `STRATEGY_LABELS`, `AgentSpec.build(rng)`. |
| `search/harness.py` | ~623 | Implemented | `play_game`, `tournament`, `run_match` (timing + optional multiprocessing), `round_robin`; `GameLog`, `MatchResult`, `AgentStats`, `PairRow`, `Table` (Wilson 95% CI). |
| `search/mcts/node.py` | ~150 | Implemented | `MCTSNode` (per-player value vectors, progressive-widening reserve). |
| `search/mcts/uct.py` | ~556 | Implemented | `uct_search`, `UCTAgent` (tree reuse), `PWContext` (progressive widening), `RootSnapshot`, `reconstruct_actions`. |
| `search/mcts/rave.py` | ~531 | Implemented | `rave_search`, `RaveAgent` (AMAF/RAVE backup). |
| `search/mcts/rollout.py` | ~300 | Implemented | `uniform_rollout`, `informed_rollout`, `voronoi_guided_rollout`, `make_rollout(kind)`; `RolloutFn` type. |

\* `search/` totals ~2,866 LOC.

## Reinforcement learning (`rl/`)

| Module group | LOC* | Status | Notes |
|---|---:|---|---|
| `rl/tabular/` (config, state_encoder, q_agent, reward, eval, train) | ~1,241 | **Reference** (Implemented, validated baseline) | ε-greedy tabular Q with legal-action masking; pickling `_QState`; `encode_state`/`decode_state`. |
| `rl/ppo/` (spaces, network, ppo_core, vec_env) | ~1,028 | **Experimental** (Implemented primitives) | `ActorCritic`, `RolloutBuffer`/GAE, `ppo_update_step`, `VectorizedEnv`; `LOGIT_MASK_VALUE=-1e9`; obs `(2N+2,H,W)`. |
| `rl/alphazero/` (spaces, network, evaluator, mcts, replay, selfplay, train) | ~1,751 | **Experimental / Partial** | `AlphaZeroNet` (ResNet), `NNEvaluator` (cache+batch+virtual loss), `puct_search`/`AlphaZeroAgent`, `ReplayBuffer`, self-play, `train_loop`. **Gating tournament is a deferred TODO** (`train.py`) — latest snapshot always promoted. |
| `rl/curriculum/` (schedule, trainer, transfer) | ~732 | **Reference** (checkpoint shipped) | `CurriculumSchedule`, `CurriculumTrainer`, `transfer_weights`; reference checkpoint at `docs/phase3d/net_reference.pt`. |
| `rl/eval/` (elo) | ~150 | Implemented | `compute_elo`, `save_elo_csv`. |

\* `rl/` totals ~4,977 LOC.

## Visualization & demo servers

| Module | LOC | Status | Public surface |
|---|---:|---|---|
| `viz.py` | 312 | Implemented | `render_ascii`, `render_matplotlib`, `save_game_gif`, `check_invariants`. matplotlib/PIL lazily imported. |
| `viz_html.py` | 672 | Implemented | `save_game_html` (self-contained replay), `heuristic_win_probs`, `alphazero_win_probs`, `AgentCard`. |
| `viz_live.py` | 699 | Implemented | `LiveServer` (stdlib HTTP: `GET /state`, `POST /reset`), `LiveConfig`, `play_and_serve`. |
| `viz_interactive.py` | 1075 | Implemented | `InteractiveServer` (human-vs-agent + spectator; `GET /`, `/state`, `POST /start`, `/action`), `AGENT_PRESETS`. |

## Package entry points

| Module | Notes |
|---|---|
| `territory_takeover/__init__.py` | Public exports: constants, `GameState`, `PlayerState`, `StepResult`, engine fns, `IllegalMoveError`, `simulate_random_rollout`; lazy `__getattr__` for `TerritoryTakeoverEnv`/`MultiAgentEnv`. |
| `search/__init__.py` | Re-exports `Agent`, all agents, search fns, harness types. |
| `eval/__init__.py`, `rl/__init__.py`, submodule `__init__.py` | Per-package public surfaces. |

## Type-annotation coverage
- ~95–98% typed; `mypy --strict` over src/ and tests/.
- `Any` appears narrowly in `engine.py` (TYPE_CHECKING), `gym_env.py` (obs dict / policy
  callable), `search/harness.py` (multiprocessing work items), and the HTTP demo modules
  (`viz_live.py`, `viz_interactive.py` response payloads). Tracked in
  [`docs/04-quality/TECHNICAL_DEBT.md`](../04-quality/TECHNICAL_DEBT.md).

## Confirmed absent (library, not an app)
No web framework, REST API, database/ORM, authentication, async runtime, message queue,
or frontend framework. The only "web" code is optional stdlib `http.server` demo viewers
(`viz_live.py`, `viz_interactive.py`).

## Related docs
- Public API surface: [`docs/02-architecture/PUBLIC_API.md`](../02-architecture/PUBLIC_API.md)
- Architecture & dependency map: [`docs/02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md), [`SYSTEM_MAP.md`](../02-architecture/SYSTEM_MAP.md)
- Scripts/entry points: [`docs/03-implementation/ENTRYPOINTS_AND_SCRIPTS.md`](ENTRYPOINTS_AND_SCRIPTS.md)
