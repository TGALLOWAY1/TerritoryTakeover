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
pytest tests/test_liveness.py -v    # single file
pytest -k test_claiming_move        # single test by name
ruff check .
mypy                                # strict mode, configured in pyproject.toml
```

`pytest` config (`testpaths = ["tests"]`, `addopts = "-ra"`) and `ruff`/`mypy` settings all live in `pyproject.toml`. `mypy` is run in strict mode — every new function must be fully typed.

## Game rules (corrected 2026-07)

The rules were corrected in July 2026 (the original implementation was a snake-path +
enclosure-flood-fill game that did not match the intended design). Current rules:

1. 2 or 4 players spawn at the exact board corners; the spawn cell is their first territory cell.
2. On your turn you move your head one cell N/S/W/E. The target must be in bounds and either
   EMPTY (you claim it — 1 point per cell) or **your own** cell (free traversal, reversing
   included). Other players' cells are permanent walls; territory never changes hands.
3. There is **no automatic enclosure capture** — walling off a region only reserves it (nobody
   else can get in); you still spend turns walking every cell to score it.
4. A player dies exactly when no EMPTY cell is reachable from their head through
   (own territory ∪ EMPTY) cells — walled out, or the board is full. Death keeps all territory.
5. The game ends when every player is dead; the winner is the player with the most territory
   (exact tie → no winner). Per-step reward is `1.0` for a claim, `0.0` for traversal.

## Architecture

TerritoryTakeover is a turn-based grid game engine designed for RL/MCTS. It is deliberately thin and numpy-first: the game state copies cheaply enough to be used as a tree-search node.

### Tile encoding (one int8 grid)

The grid is a single `np.int8` array of shape `(board_size, board_size)`. Values:

- `0` = EMPTY
- `1..4` = `OWNED_CODES[player_id]` — a player's territory (every cell they have visited)

Constants (`src/territory_takeover/constants.py`) expose both scalar names (`PLAYER_1_OWNED`, etc.) and the indexed tuple `OWNED_CODES` — prefer the tuple when a `player_id` is in scope so code stays symmetric across players. `DIRECTIONS` is `(N, S, W, E)` and the action space is fixed at 4 direction indices regardless of board size (keeps MCTS policy-head shape constant).

### State split: int8 grid + per-player scalars

`GameState` (`src/territory_takeover/state.py`) holds the grid plus a list of `PlayerState`:

- `head: tuple[int, int]` — current position (always on one of the player's own cells)
- `territory_count: int` — must stay in lockstep with the grid's count of that player's OWNED cells
- `alive: bool`, plus `alive_witness` — a cached EMPTY cell adjacent to the player's territory
  that proves liveness until that exact cell is claimed (O(1) amortized liveness checks)

The grid is the source of truth; `territory_count` and `empty_count` are caches every mutation must keep consistent. `GameState.copy()` is tuned for tree search: numpy `memcpy` on the grid plus value copies for scalars (~3 µs on 20×20) — there are no per-player containers to clone.

### Engine entry points (`src/territory_takeover/engine.py`)

- `new_game(board_size, num_players, spawn_positions=None, seed=None)` — constructs a fresh `GameState`. Default spawns are the exact corners (only defined for 2 and 4 players). `seed` permutes player-to-corner assignment reproducibly when defaults are used.
- `reset(state)` — in-place reset reusing the existing grid buffer (important for training loops).
- `has_reachable_empty(state, player_id) -> bool` — the liveness rule. Because a player's territory is connected and only grows, liveness ⟺ "some EMPTY cell is adjacent to own territory"; a successful check caches that cell as a witness on `PlayerState`, valid until claimed, making repeat checks O(1) amortized. Cache miss falls back to a BFS over own cells (stamped scratch buffer, no per-call allocation) that early-exits on the first EMPTY neighbor.
- `step(state, action, strict=False) -> StepResult` — full move application. Inline legality check (in-bounds and EMPTY-or-own; no `legal_action_mask` allocation on the hot path). A claim writes `OWNED_CODES[player]`, bumps `territory_count`, decrements `empty_count`, reward `1.0`; traversal just moves `head`, reward `0.0`. Illegal moves are a **no-op wasted turn** (default) or raise `IllegalMoveError` (`strict=True`) — death comes only from the liveness rule, never from a bad action, because RL agents emitting garbage mid-episode shouldn't crash training or be executed for it. `_advance_turn` skips dead seats and kills any alive seat whose head can no longer reach an EMPTY cell. The game terminates when `state.alive_count == 0`; `_compute_winner` then fills `state.winner` (argmax `territory_count`, tie → None).

### Actions (`src/territory_takeover/actions.py`)

`legal_action_mask` returns a `(4,)` bool array (for policy masking); `legal_actions` returns a `list[int]`; `claiming_actions` returns the EMPTY-target subset. A direction is legal iff the target cell is in bounds and EMPTY or the mover's own cell. All functions hot-path `grid.item(r, c)` instead of `grid[r, c]` — keep that when editing; the perf target documented in the tests is < 1 µs per call. Note `has_any_legal_action` is NOT the liveness test — a player can have traversal moves yet be dead.

### Rollouts (`src/territory_takeover/rollout.py`, `search/mcts/rollout.py`)

`simulate_random_rollout` plays to terminal under a **claim-biased** random policy (uniform over claiming moves when any exist, else uniform over traversal moves). The bias is what keeps playouts short — uniform-over-all-legal random-walks over already-owned cells and stretches games by ~10×. MCTS's `uniform_rollout` delegates to it; `informed_rollout` adds a claim bonus + frontier scoring.

### Performance targets (checked manually, not in CI)

- `GameState.copy()`: < 50 µs (test_state.py has a loose baseline; measured ~3 µs).
- `legal_actions`: < 1 µs.
- Liveness check (`has_reachable_empty`): O(1) amortized via the witness cache; full 20×20 claim-biased random game ≈ 6 ms.

These numbers drive implementation choices — e.g. the liveness BFS uses an explicit `deque` with an `np.int32` stamp scratch (no recursion, no per-call zeroing), and `reset` reuses the grid buffer.

### Visualization & front end (`src/territory_takeover/viz*.py`)

All rendering is pure-Python, no JS build tooling. `viz.py` holds the canonical palettes (`TILE_COLORS` — 5 entries: empty + 4 owned — and `HEAD_EDGE_COLORS`) plus ASCII/matplotlib renderers; `viz_html.py` builds the JSON frame payload (`_frame_payload`) and static replays; `viz_live.py` streams a scripted spectator game.

The **Arena** front end is `viz_interactive.py` — a single self-contained HTML/CSS/JS page (inline strings) served by a stdlib `ThreadingHTTPServer`, launched via `scripts/play_interactive.py`. It drives the *real* engine: a seat's territory shown in the UI is `territory_count`.

- **Endpoints**: `GET /` (page), `GET /state?episode&since` (incremental frame poll; includes `paused`/`speed`/`waiting_for_human`/`human_seat`), `GET /agents` (preset key/label/description), `POST /start` (config body), `POST /control` (`{"cmd": play|pause|step|reset|speed}`), `POST /action` (human move 0–3).
- **Pacing model**: the game runs in a background thread. In watch mode it blocks in `_wait_to_step()` (guarded by `_ctrl_cond`) until a control command lets it proceed; *step* requests advance exactly one engine move (one seat) and then re-pause; *speed* scales the pacing sleep (`1/(base_fps*speed)`). Play mode (a human in seat 0) keeps the existing arrow-key flow and ignores pause. `start_game` stores `_last_config` so `cmd=reset` can restart the same lineup.
- **Client-derived stats**: territory %, legal-move counts, and last-move direction are computed in the browser from the existing frame keys (`grid`/`heads`/`alive`/`territory`) — do **not** change `_frame_payload`'s schema, which `tests/test_viz_html.py` pins. Agent presets live in `AGENT_PRESETS` (`viz_interactive.py`); `_build_agent` maps a preset key to a registry agent.
- **Inline assets caveat**: every physical line in the template strings still counts toward ruff `line-length = 100`, so keep embedded CSS/JS lines wrapped.

## Conventions

- Test style is pytest function-based (no classes, no fixtures beyond trivial helpers). Parametrized tests are currently avoided because mypy strict + missing `pytest` stubs flags `@pytest.mark.parametrize` as an untyped decorator — prefer an internal loop over `range(N)` with `f"trial={i}"` assertion messages.
- Ruff rule set: `E, F, I, B, UP, N, SIM, RUF, ANN, TID` with `line-length = 100`. `ANN` is disabled for `tests/*`.
- Prefer extending `PlayerState`/`GameState` in place over introducing parallel structures — the cheap-copy semantics depend on knowing exactly what to duplicate.

## Historical note on committed benchmarks

The committed leaderboards under `docs/baseline_report*.md`, the `benchmarks/*_FINDINGS.md` notes, phase writeups (`KEY_FINDINGS.md`, `PHASE3_SUMMARY.md`), and the shipped `docs/phase3d/net_reference.pt` checkpoint all predate the 2026-07 rules correction and describe the OLD enclosure game. See `docs/experiments/corrected_rules_eval.md` for the first corrected-rules evaluation.
