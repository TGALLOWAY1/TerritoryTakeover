# Data Model

> The in-memory data structures (this library has no database). Core types are numpy
> arrays and dataclasses. For tile codes and tensor encodings see
> [`STATE_AND_ENCODING.md`](STATE_AND_ENCODING.md). Last audited 2026-05-28.

## GameState (`state.py`)
- **Purpose:** Complete game state; the unit copied during tree search.
- **Fields:** `grid: np.int8[H,W]` (source of truth), `players: list[PlayerState]`,
  `current_player: int`, `turn_number: int`, `winner: int | None`, `done: bool`,
  `alive_count: int`, `empty_count: int`, plus scratch buffers `_scratch_reachable`,
  `_enclosure_stamp` (reused by enclosure BFS to avoid per-call allocation).
- **Created by:** `GameState.empty()`, `engine.new_game()`, `GameState.copy()`.
- **Read by:** everything (agents, eval, harness, viz).
- **Mutated by:** `engine.step`, `engine.reset`, `detect_and_apply_enclosure`.
- **Invariants:** grid is canonical; `players[i].path_set`, `claimed_count`, and global
  `empty_count`/`alive_count` must stay in lockstep with the grid. `copy()` does numpy `memcpy`
  on the grid, shallow-copies each path list/set (tuples are immutable so sharing is safe), and
  value-copies scalars; scratch buffers are **not** cloned.
- **Open questions:** none; consistency is enforced by tests + `viz.check_invariants`.

## PlayerState (`state.py`)
- **Purpose:** Per-player redundant view of that player's snake path.
- **Fields:** `player_id: int`, `path: list[tuple[int,int]]` (ordered, for head/predecessor),
  `path_set: set[tuple[int,int]]` (O(1) membership), `head: tuple[int,int]`,
  `claimed_count: int`, `alive: bool`.
- **Invariant:** `claimed_count` equals the grid count of that player's CLAIMED code.

## StepResult (`engine.py`)
- **Purpose:** Return value of `step`.
- **Fields:** `state: GameState`, `reward: float`, `done: bool`, `info: dict`.
- **Semantics:** `reward = 1.0 + claimed_this_turn` on a legal move, `0.0` on an illegal move.

## MCTSNode (`search/mcts/node.py`)
- **Purpose:** Search-tree node for UCT/RAVE.
- **Fields:** `children: dict`, `incoming_action`, `parent`, `player_to_move`, `state`,
  `untried_actions`, `visits: int`, `total_value` (**per-player vector**, not scalar — N-player
  aware), `terminal`, `terminal_value`, `pw_reserve` (progressive-widening deferred actions).
- **Read/updated by:** `uct_search`, `rave_search` (selection/expansion/backprop).

## AgentSpec / registry types (`search/registry.py`, `harness.py`)
- `AgentSpec(name, class_name, kwargs)` with `build(rng)`; `REGISTRY` maps class name →
  constructor; `STRATEGY_LABELS` maps to display labels.
- Harness result types: `GameLog`, `MatchResult`, `AgentStats`, `PairRow`, `Table` (Wilson CI).

## RL persistence types
- **Tabular:** `QConfig` (hyperparameters); `_QState` (pickled training state); `StateKey =
  tuple[...]` hashable state encoding (`rl/tabular/state_encoder.py`).
- **PPO:** `PpoNetConfig`; `RolloutBatch`/`RolloutBuffer` (ring buffer + GAE) (`rl/ppo/`).
- **AlphaZero:** `AZNetConfig`; `Sample(grid, scalars, mask, visits, final_scores,
  per_step_reward, step_index)`; `ReplayBuffer` (pre-allocated numpy, `.npz` save/load);
  `SelfPlayConfig`; `TrainConfig`/`TrainMetrics` (`rl/alphazero/`).
- **Curriculum:** `CurriculumSchedule` (stages of board_size/num_players/difficulty).

## On-disk artifacts (not a schema, but persisted)
| Artifact | Format | Producer | Consumer |
|---|---|---|---|
| Model checkpoint | `.pt` (torch) | AZ/curriculum trainers | `eval_*`, registry |
| Replay buffer | `.npz` | `ReplayBuffer.save` | `ReplayBuffer.load` |
| Tournament report | `.md` + `.csv` | `run_baseline_report.py` | reviewers, README |
| Elo table | `.csv` | `compute_elo.py` | viz win-prob/Elo overlays |
| Configs | `.yaml` | hand-authored (`configs/`) | training CLIs |

## Related docs
- Tile codes + tensor encodings: [`STATE_AND_ENCODING.md`](STATE_AND_ENCODING.md)
- Public constructors/functions: [`PUBLIC_API.md`](PUBLIC_API.md)
- Cheap-copy rationale: ADR-002 ([`docs/adr/ADR-002-state-split.md`](../adr/ADR-002-state-split.md))
