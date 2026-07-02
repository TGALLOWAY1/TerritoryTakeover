"""Tests for the viz module: ASCII rendering, validation, and random-game sanity."""

from __future__ import annotations

import os
import random
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from territory_takeover.actions import legal_actions
from territory_takeover.constants import OWNED_CODES
from territory_takeover.engine import new_game, step
from territory_takeover.state import GameState, PlayerState
from territory_takeover.viz import HEAD_EDGE_COLORS, TILE_COLORS, render_ascii, validate_state


def _make_small_state() -> GameState:
    """4x4 board: player 0 owns (0,0)-(0,1) with head (0,1); player 1 owns (3,3)."""
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[0, 0] = OWNED_CODES[0]
    grid[0, 1] = OWNED_CODES[0]
    grid[3, 3] = OWNED_CODES[1]
    p0 = PlayerState(
        player_id=0,
        head=(0, 1),
        territory_count=2,
        alive=True,
    )
    p1 = PlayerState(
        player_id=1,
        head=(3, 3),
        territory_count=1,
        alive=True,
    )
    return GameState(grid=grid, players=[p0, p1])


def test_tile_colors_palette_shape() -> None:
    # One color per tile code: EMPTY + 4 owned codes.
    assert len(TILE_COLORS) == 5
    assert len(HEAD_EDGE_COLORS) == 4
    for color in (*TILE_COLORS, *HEAD_EDGE_COLORS):
        assert color.startswith("#") and len(color) == 7


def test_render_ascii_basic() -> None:
    state = _make_small_state()
    out = render_ascii(state)
    assert "turn=0" in out
    assert "[1]" in out
    assert "[2]" in out
    assert "." in out
    # Old claimed-tile letters are gone from the encoding.
    for letter in ("A", "B", "C", "D"):
        assert letter not in out.splitlines()[1]


def test_render_ascii_owned_non_head_cells_use_digits() -> None:
    state = _make_small_state()
    out = render_ascii(state)
    # (0, 0) is owned but not the head, (0, 1) is the head: "1[1].." row.
    first_row = out.splitlines()[1]
    assert first_row == "1[1].."


def test_render_ascii_empty_state() -> None:
    state = GameState.empty(height=3, width=3, num_players=4)
    out = render_ascii(state)
    assert "[" not in out
    assert out.count(".") == 9


def test_render_ascii_marks_every_spawned_head() -> None:
    state = new_game(6, 4, seed=None)
    out = render_ascii(state)
    for digit in ("1", "2", "3", "4"):
        assert f"[{digit}]" in out


def test_validate_state_valid_new_game() -> None:
    state = new_game(10, 4, seed=0)
    assert validate_state(state) == []
    assert validate_state(state, deep=True) == []


def test_validate_state_detects_grid_desync_at_head() -> None:
    state = new_game(10, 4, seed=0)
    head = state.players[0].head
    state.grid[head] = 0
    state.empty_count += 1
    violations = validate_state(state)
    assert violations
    assert any("expected OWNED code" in v for v in violations)


def test_validate_state_detects_head_out_of_bounds() -> None:
    state = new_game(10, 4, seed=0)
    state.players[0].head = (99, 99)
    violations = validate_state(state)
    assert any("out of bounds" in v for v in violations)


def test_validate_state_detects_head_off_own_territory() -> None:
    state = new_game(10, 2, seed=None)
    # Move the head bookkeeping onto an EMPTY cell without touching the grid.
    state.players[0].head = (5, 5)
    violations = validate_state(state)
    assert any("expected OWNED code" in v for v in violations)


def test_validate_state_detects_territory_count_desync() -> None:
    state = new_game(10, 4, seed=0)
    state.players[0].territory_count = 99
    violations = validate_state(state)
    assert any("territory_count" in v for v in violations)


def test_validate_state_detects_empty_count_desync() -> None:
    state = new_game(10, 4, seed=0)
    state.empty_count -= 5
    violations = validate_state(state)
    assert any("empty_count" in v for v in violations)


def test_validate_state_detects_alive_count_desync() -> None:
    state = new_game(10, 4, seed=0)
    state.players[0].alive = False  # without decrementing alive_count
    violations = validate_state(state)
    assert any("alive_count" in v for v in violations)


def test_validate_state_deep_detects_disconnected_territory() -> None:
    state = new_game(10, 2, seed=None)
    # Paint an owned cell far from player 0's spawn, keeping the cheap
    # count checks consistent so only the connectivity check can fire.
    state.grid[5, 5] = OWNED_CODES[0]
    state.players[0].territory_count += 1
    state.empty_count -= 1
    assert validate_state(state, deep=False) == []
    deep_violations = validate_state(state, deep=True)
    assert any("disconnected" in v for v in deep_violations)


def test_validate_state_deep_accepts_grown_territory() -> None:
    # Territory grown through real engine moves is 4-connected by construction.
    state = new_game(6, 2, seed=None)
    for _ in range(8):
        if state.done:
            break
        acts = legal_actions(state, state.current_player)
        assert acts, "fresh corner spawns always have a legal move"
        step(state, acts[0])
    assert validate_state(state, deep=True) == []


def test_random_games_stay_valid() -> None:
    """CI check: play random games on a 10x10 board, validate after every step."""
    num_games = 25
    board_size = 10
    rng = random.Random(20260418)
    for game_idx in range(num_games):
        state = new_game(board_size, 4, seed=game_idx)
        turns = 0
        while not state.done and turns < 2000:
            pid = state.current_player
            acts = legal_actions(state, pid)
            action = rng.choice(acts) if acts else 0
            step(state, action)
            turns += 1
            violations = validate_state(state)
            assert not violations, (
                f"game {game_idx} turn {turns}: {violations[:3]}"
            )
        deep_violations = validate_state(state, deep=True)
        assert not deep_violations, (
            f"game {game_idx} terminal deep check: {deep_violations[:3]}"
        )


def test_render_matplotlib_smoke() -> None:
    pytest.importorskip("matplotlib")
    from territory_takeover.viz import render_matplotlib

    state = new_game(10, 4, seed=0)
    ax = render_matplotlib(state)
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.close("all")


def test_save_game_gif_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from territory_takeover.viz import save_game_gif

    state = new_game(10, 4, seed=0)
    trajectory = [state.copy()]
    for _ in range(3):
        acts = legal_actions(state, state.current_player) or [0]
        step(state, acts[0])
        trajectory.append(state.copy())

    out_path = tmp_path / "game.gif"
    save_game_gif(trajectory, str(out_path), fps=4)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_game_gif_empty_raises() -> None:
    from territory_takeover.viz import save_game_gif

    with pytest.raises(ValueError):
        save_game_gif([], "/tmp/unused.gif")
