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

Move application (writing `placed_cell` to the grid, appending to path, advancing `head`) is not yet implemented — `detect_and_apply_enclosure` is meant to be the last step of a future `apply_move`.

### Actions (`src/territory_takeover/actions.py`)

`legal_action_mask` returns a `(4,)` bool array (for policy masking); `legal_actions` returns a `list[int]`. A direction is legal iff the target cell is in bounds and EMPTY. Both functions hot-path `grid.item(r, c)` instead of `grid[r, c]` — keep that when editing; the perf target documented in the tests is < 1 µs per call.

### Performance targets (checked manually, not in CI)

- `GameState.copy()`: < 50 µs (test_state.py has a loose baseline).
- `legal_actions`: < 1 µs.
- `detect_and_apply_enclosure` after trigger: < 200 µs on 40×40. Trigger check itself ~4 hashtable lookups.

These numbers drive implementation choices — e.g. BFS uses an explicit `deque` with an `np.bool_` mask (no recursion: 40×40 would blow Python's 1000-frame limit), and `reset` reuses the grid buffer.

## Conventions

- Test style is pytest function-based (no classes, no fixtures beyond trivial helpers). Parametrized tests are currently avoided because mypy strict + missing `pytest` stubs flags `@pytest.mark.parametrize` as an untyped decorator — prefer an internal loop over `range(N)` with `f"trial={i}"` assertion messages.
- Ruff rule set: `E, F, I, B, UP, N, SIM, RUF, ANN, TID` with `line-length = 100`. `ANN` is disabled for `tests/*`.
- Prefer extending `PlayerState`/`GameState` in place over introducing parallel structures — the cheap-copy semantics depend on knowing exactly what to duplicate.
