# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install for development (requires Python 3.11+):

```
pip install -e ".[dev]"
```

Test, lint, typecheck:

```
pytest                              # full suite
pytest tests/test_enclosure.py -v   # single file
pytest -k test_simple_3x3_loop      # single test by name
ruff check .
mypy                                # strict mode, configured in pyproject.toml
```

`pytest` config (`testpaths = ["tests"]`, `addopts = "-ra"`) and `ruff`/`mypy` settings all live in `pyproject.toml`. `mypy` is run in strict mode — every new function must be fully typed.

## Architecture

TerritoryTakeover is a turn-based grid game engine designed for RL/MCTS. It is deliberately thin and numpy-first: the game state copies cheaply enough to be used as a tree-search node.

### Tile encoding (one int8 grid)

The grid is a single `np.int8` array of shape `(board_size, board_size)`. Values:

- `0` = EMPTY
- `1..4` = `PATH_CODES[player_id]` — a player's snake path
- `5..8` = `CLAIMED_CODES[player_id]` — a player's claimed territory

Constants (`src/territory_takeover/constants.py`) expose both scalar names (`PLAYER_1_PATH`, etc.) and the indexed tuples `PATH_CODES` / `CLAIMED_CODES` — prefer the tuples when a `player_id` is in scope so code stays symmetric across players. `DIRECTIONS` is `(N, S, W, E)` and the action space is fixed at 4 direction indices regardless of board size (keeps MCTS policy-head shape constant).

### State split: int8 grid + per-player structures

`GameState` (`src/territory_takeover/state.py`) holds the grid plus a list of `PlayerState`. Each `PlayerState` carries a redundant view of that player's path:

- `path: list[tuple[int, int]]` — ordered for head/predecessor access
- `path_set: set[tuple[int, int]]` — O(1) membership
- `head: tuple[int, int]` — current head
- `claimed_count: int` — must stay in lockstep with the grid's count of that player's CLAIMED cells

The grid is the source of truth for rendering; `path_set` and `claimed_count` are caches that every mutation must keep consistent. `GameState.copy()` is tuned for tree search: numpy `memcpy` on the grid, shallow copies of each path list/set (tuples of ints are immutable so sharing is safe), value copies for scalars.

### Engine entry points (`src/territory_takeover/engine.py`)

- `new_game(board_size, num_players, spawn_positions=None, seed=None)` — constructs a fresh `GameState`. Default spawns are symmetric corner insets (only defined for 2 and 4 players). `seed` permutes player-to-spawn assignment reproducibly when defaults are used.
- `reset(state)` — in-place reset reusing the existing grid buffer (important for training loops).
- `detect_and_apply_enclosure(state, player_id, placed_cell) -> int` — claim-resolution primitive. **Caller contract**: `placed_cell` must already be appended to the player's `path`/`path_set`, set as `head`, and written to the grid before this is called. The function returns the number of newly-claimed cells and mutates both `grid` and `claimed_count`. Algorithm: a cheap ~4-lookup trigger check (adjacency to a same-player path tile other than the predecessor), then a boundary BFS flood fill over EMPTY cells treating every non-empty tile as a wall. Attribution rule: enclosed cells always go to the triggering player, even if opponent path tiles form part of the pocket boundary.
- `step(state, action, strict=False) -> StepResult` — full move application. Runs an inline in-bounds/EMPTY legality check (no `legal_action_mask` allocation on the hot path), writes `PATH_CODES[player]` to the grid, appends to `path`/`path_set`, advances `head`, decrements `empty_count`, then delegates to `detect_and_apply_enclosure` for claim resolution. Illegal moves mark the moving player `alive=False` and advance the turn (default) or raise `IllegalMoveError` (`strict=True`) — the default is RL-friendly because agents emitting garbage mid-episode shouldn't crash training. Returns `StepResult(state, reward, done, info)` with `reward = 1.0 + claimed_this_turn` on legal moves and `0.0` on illegal. The game terminates when `state.alive_count <= 1`; `_compute_winner` then fills `state.winner`.

### Actions (`src/territory_takeover/actions.py`)

`legal_action_mask` returns a `(4,)` bool array (for policy masking); `legal_actions` returns a `list[int]`. A direction is legal iff the target cell is in bounds and EMPTY. Both functions hot-path `grid.item(r, c)` instead of `grid[r, c]` — keep that when editing; the perf target documented in the tests is < 1 µs per call.

### Performance targets (checked manually, not in CI)

- `GameState.copy()`: < 50 µs (test_state.py has a loose baseline).
- `legal_actions`: < 1 µs.
- `detect_and_apply_enclosure` after trigger: < 200 µs on 40×40. Trigger check itself ~4 hashtable lookups.

These numbers drive implementation choices — e.g. BFS uses an explicit `deque` with an `np.bool_` mask (no recursion: 40×40 would blow Python's 1000-frame limit), and `reset` reuses the grid buffer.

### Visualization & front end (`src/territory_takeover/viz*.py`)

All rendering is pure-Python, no JS build tooling. `viz.py` holds the canonical palettes (`TILE_COLORS`, `HEAD_EDGE_COLORS`) plus ASCII/matplotlib renderers; `viz_html.py` builds the JSON frame payload (`_frame_payload`) and static replays; `viz_live.py` streams a scripted spectator game.

The **Arena** front end is `viz_interactive.py` — a single self-contained HTML/CSS/JS page (inline strings) served by a stdlib `ThreadingHTTPServer`, launched via `scripts/play_interactive.py`. It drives the *real* engine, so the territory model is the engine's own: a cell is owned by a player once visited (its path) and stays owned, plus enclosed cells — a seat's territory shown in the UI is `len(path) + claimed_count`.

- **Endpoints**: `GET /` (page), `GET /state?episode&since` (incremental frame poll; includes `paused`/`speed`/`waiting_for_human`/`human_seat`), `GET /agents` (preset key/label/description), `POST /start` (config body), `POST /control` (`{"cmd": play|pause|step|reset|speed}`), `POST /action` (human move 0–3).
- **Pacing model**: the game runs in a background thread. In watch mode it blocks in `_wait_to_step()` (guarded by `_ctrl_cond`) until a control command lets it proceed; *step* requests advance exactly one engine move (one seat) and then re-pause; *speed* scales the pacing sleep (`1/(base_fps*speed)`). Play mode (a human in seat 0) keeps the existing arrow-key flow and ignores pause. `start_game` stores `_last_config` so `cmd=reset` can restart the same lineup.
- **Client-derived stats**: territory %, legal-move counts, and last-move direction are computed in the browser from the existing frame keys (`grid`/`heads`/`alive`/`path_len`/`claimed`) — do **not** change `_frame_payload`'s schema, which `tests/test_viz_html.py` pins. Agent presets live in `AGENT_PRESETS` (`viz_interactive.py`); `_build_agent` maps a preset key to a registry agent.
- **Inline assets caveat**: every physical line in the template strings still counts toward ruff `line-length = 100`, so keep embedded CSS/JS lines wrapped.

## Conventions

- Test style is pytest function-based (no classes, no fixtures beyond trivial helpers). Parametrized tests are currently avoided because mypy strict + missing `pytest` stubs flags `@pytest.mark.parametrize` as an untyped decorator — prefer an internal loop over `range(N)` with `f"trial={i}"` assertion messages.
- Ruff rule set: `E, F, I, B, UP, N, SIM, RUF, ANN, TID` with `line-length = 100`. `ANN` is disabled for `tests/*`.
- Prefer extending `PlayerState`/`GameState` in place over introducing parallel structures — the cheap-copy semantics depend on knowing exactly what to duplicate.
