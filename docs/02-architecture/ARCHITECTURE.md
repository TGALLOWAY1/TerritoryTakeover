# Architecture

> How the system works internally. The repo is a layered Python library: a deterministic
> simulation core at the bottom, an algorithm stack on top, an evaluation/benchmark layer,
> and visualization surfaces. Last audited 2026-05-28.

## System overview

```
            ┌─────────────────────────────────────────────┐
   Layer 4  │  Visualization  (viz, viz_html, viz_live,     │
            │                  viz_interactive)             │
            └───────────────▲─────────────────────────────┘
                            │ reads state / game logs
            ┌───────────────┴─────────────────────────────┐
   Layer 3  │  Benchmark & Eval  (search/harness, eval/,    │
            │                     rl/eval/elo)              │
            └───────────────▲─────────────────────────────┘
                            │ runs agents
            ┌───────────────┴─────────────────────────────┐
   Layer 2  │  Agents:  classical search (search/)          │
            │           + RL (rl/tabular, ppo, alphazero,   │
            │             curriculum)                       │
            │  all conform to the search/agent.py Protocol  │
            └───────────────▲─────────────────────────────┘
                            │ step()/legal_actions()/copy()
            ┌───────────────┴─────────────────────────────┐
   Layer 1  │  Engine core:  state, actions, engine,        │
            │                rollout, constants             │
            └─────────────────────────────────────────────┘
```

The game environment is the contract every layer above shares. The game itself is a
testbed; the value is the layered, comparable algorithm stack.

## Major modules / responsibilities

- **`constants.py`** — tile encoding (EMPTY / PATH / CLAIMED codes), `DIRECTIONS` (N,S,W,E),
  board defaults. The 4-direction action space is fixed regardless of board size (keeps the
  MCTS/RL policy-head shape constant).
- **`state.py`** — `GameState` (int8 grid + per-player `PlayerState` + counters + scratch
  buffers) and the tree-search-tuned `copy()`.
- **`actions.py`** — legal-move queries and `(4,)` policy masks; hot-path `grid.item(r,c)`.
- **`engine.py`** — `new_game`/`reset`/`step`; `detect_and_apply_enclosure` (trigger check +
  boundary BFS flood fill); winner + terminal rewards.
- **`rollout.py`** — allocation-light random playout used by MCTS and self-play.
- **`search/`** — the `Agent` Protocol plus Random/Greedy, Max-N/Paranoid, UCT, RAVE, the
  rollout policies, the agent registry, and the tournament harness.
- **`eval/`** — heuristic `LinearEvaluator`, Voronoi reachability, feature functions, tuner.
- **`rl/`** — tabular Q, PPO primitives, AlphaZero (net/evaluator/MCTS/replay/self-play/train),
  curriculum, and Elo.
- **`viz*.py`** — ASCII/matplotlib/GIF, self-contained HTML replay, live + interactive HTTP viewers.

## Runtime flow (one move)

1. An `Agent.select_action(state, player_id, time_budget_s, max_iterations)` chooses a direction.
   - MCTS agents repeatedly `GameState.copy()` + `step` to build a search tree, optionally using
     `rollout` (classical) or an `NNEvaluator` (AlphaZero) at leaves.
2. `engine.step(state, action)` validates legality inline, writes the PATH code, advances the head,
   updates `path`/`path_set`/`empty_count`, then calls `detect_and_apply_enclosure`.
3. Enclosure: a ~4-lookup trigger check, then a boundary BFS over EMPTY cells (every non-empty
   tile is a wall). Enclosed cells are claimed by the triggering player.
4. `StepResult(state, reward, done, info)` returns; turn advances; game ends when
   `alive_count <= 1`, and `_compute_winner` fills `winner`.

## Data flow (a benchmark)

`run_match`/`round_robin` (harness) → derive per-game seeds + per-agent RNGs from one root via
`SeedSequence` → `play_game` drives agents through `engine.step` → terminal `GameState` →
`AgentStats`/`Table` with Wilson 95% CIs → markdown/CSV report (committed). Optional
multiprocessing produces **bit-identical** logs to the serial path (ADR-006).

## Auth / storage / external services

- **No authentication, no database, no external services.** State lives in memory; the int8
  grid is the source of truth.
- **Persistence** is file-based and optional: model checkpoints (`.pt` via `torch.save/load`),
  replay buffers (`.npz`), tournament reports (markdown/CSV/YAML), demo media (PNG/GIF/HTML).
- **Network** only via local stdlib `http.server` demo viewers (Layer 4).

## Important boundaries / invariants

- **Grid is canonical; caches must stay in lockstep.** `path_set` and `claimed_count` are caches
  every mutation must keep consistent with the grid (ADR-002). `copy()` knows exactly what to
  duplicate — extend `GameState`/`PlayerState` in place rather than adding parallel structures.
- **`detect_and_apply_enclosure` caller contract:** the placed cell must already be in
  `path`/`path_set`, set as `head`, and written to the grid before the call.
- **Fixed 4-action space** across board sizes (policy-head stability).
- **Agent Protocol** (`search/agent.py`) is the single interface the harness and viewers depend
  on — structural typing, no inheritance required.

## Known coupling / architectural risk

- **Redundant state caches** (grid vs. `path_set`/`claimed_count`/`empty_count`) trade safety for
  speed; correctness depends on every mutation path updating all of them. Mitigated by
  `viz.check_invariants` and the enclosure equivalence tests.
- **AlphaZero training is incomplete at the pipeline level** — gating is stubbed
  (`rl/alphazero/train.py`), so "self-play champion = latest snapshot". See KNOWN_ISSUES / ADR-005.
- **RL obs encoders are duplicated by design** (PPO rotates active player to channel 0; AlphaZero
  uses fixed seat order + turn one-hot). See [`STATE_AND_ENCODING.md`](STATE_AND_ENCODING.md).
- **Perf targets are not CI-enforced** — regressions are caught by the local benchmark harness,
  not automatically. See [`docs/04-quality/RISK_REGISTER.md`](../04-quality/RISK_REGISTER.md).

## Related docs
- Module dependency map: [`SYSTEM_MAP.md`](SYSTEM_MAP.md)
- Data structures: [`DATA_MODEL.md`](DATA_MODEL.md) · Encodings: [`STATE_AND_ENCODING.md`](STATE_AND_ENCODING.md)
- Public API: [`PUBLIC_API.md`](PUBLIC_API.md) · Dependencies: [`INTEGRATIONS.md`](INTEGRATIONS.md)
- Decision rationale: [`docs/adr/`](../adr/README.md), [`docs/06-history/DECISION_LOG.md`](../06-history/DECISION_LOG.md)
