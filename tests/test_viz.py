"""Tests for the viz module: ASCII rendering, validation, and random-game sanity."""

from __future__ import annotations

import os
import random
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from territory_takeover.actions import legal_actions
from territory_takeover.constants import (
    PLAYER_1_CLAIMED,
    PLAYER_1_PATH,
    PLAYER_2_PATH,
)
from territory_takeover.engine import new_game, step
from territory_takeover.state import GameState, PlayerState
from territory_takeover.viz import render_ascii, validate_state


def _make_small_state() -> GameState:
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[0, 0] = PLAYER_1_PATH
    grid[0, 1] = PLAYER_1_PATH
    grid[3, 3] = PLAYER_2_PATH
    p0 = PlayerState(
        player_id=0,
        path=[(0, 0), (0, 1)],
        path_set={(0, 0), (0, 1)},
        head=(0, 1),
        claimed_count=0,
        alive=True,
    )
    p1 = PlayerState(
        player_id=1,
        path=[(3, 3)],
        path_set={(3, 3)},
        head=(3, 3),
        claimed_count=0,
        alive=True,
    )
    return GameState(grid=grid, players=[p0, p1])


def test_render_ascii_basic() -> None:
    state = _make_small_state()
    out = render_ascii(state)
    assert "turn=0" in out
    assert "[1]" in out
    assert "[2]" in out
    assert "." in out
    for letter in ("A", "B", "C", "D"):
        assert letter not in out


def test_render_ascii_includes_claimed_letters() -> None:
    state = _make_small_state()
    state.grid[2, 2] = PLAYER_1_CLAIMED
    state.players[0].claimed_count = 1
    out = render_ascii(state)
    assert "A" in out


def test_render_ascii_empty_state() -> None:
    state = GameState.empty(height=3, width=3, num_players=4)
    out = render_ascii(state)
    assert "[" not in out
    assert out.count(".") == 9


def test_validate_state_valid_new_game() -> None:
    state = new_game(10, 4, seed=0)
    assert validate_state(state) == []
    assert validate_state(state, deep=True) == []


def test_validate_state_detects_grid_desync() -> None:
    state = new_game(10, 4, seed=0)
    head = state.players[0].head
    state.grid[head] = 0
    violations = validate_state(state)
    assert violations
    assert any("PATH code" in v or "expected" in v for v in violations)


def test_validate_state_detects_head_mismatch() -> None:
    state = new_game(10, 4, seed=0)
    state.players[0].head = (9, 9)
    violations = validate_state(state)
    assert any("head" in v and "path[-1]" in v for v in violations)


def test_validate_state_detects_path_set_desync() -> None:
    state = new_game(10, 4, seed=0)
    state.players[0].path_set.add((7, 7))
    violations = validate_state(state)
    assert any("path_set" in v for v in violations)


def test_validate_state_detects_claimed_count_desync() -> None:
    state = new_game(10, 4, seed=0)
    state.players[0].claimed_count = 99
    violations = validate_state(state)
    assert any("claimed_count" in v for v in violations)


def test_validate_state_detects_shared_cell() -> None:
    state = _make_small_state()
    shared = (0, 1)
    state.players[1].path.append(shared)
    state.players[1].path_set.add(shared)
    state.players[1].head = shared
    violations = validate_state(state)
    assert any("both player" in v for v in violations)


def test_validate_state_deep_catches_unfenced_claimed() -> None:
    state = new_game(10, 4, seed=0)
    state.grid[0, 5] = PLAYER_1_CLAIMED
    state.players[0].claimed_count += 1
    assert validate_state(state, deep=False) == []
    deep_violations = validate_state(state, deep=True)
    assert any("reachable from boundary" in v for v in deep_violations)


def test_validate_state_deep_accepts_properly_fenced_claimed() -> None:
    """A claimed cell surrounded by path tiles should pass the deep check."""
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1, 1] = PLAYER_1_PATH
    grid[1, 2] = PLAYER_1_PATH
    grid[1, 3] = PLAYER_1_PATH
    grid[2, 1] = PLAYER_1_PATH
    grid[2, 3] = PLAYER_1_PATH
    grid[3, 1] = PLAYER_1_PATH
    grid[3, 2] = PLAYER_1_PATH
    grid[3, 3] = PLAYER_1_PATH
    grid[2, 2] = PLAYER_1_CLAIMED
    path_cells = [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 3),
        (3, 1), (3, 2), (3, 3),
    ]
    p0 = PlayerState(
        player_id=0,
        path=path_cells,
        path_set=set(path_cells),
        head=(3, 3),
        claimed_count=1,
        alive=True,
    )
    p1 = PlayerState(
        player_id=1,
        path=[],
        path_set=set(),
        head=(-1, -1),
        claimed_count=0,
        alive=True,
    )
    state = GameState(grid=grid, players=[p0, p1])
    assert validate_state(state, deep=True) == []


def test_random_games_stay_valid() -> None:
    """CI check: play 50 random games on a 10x10 board, validate after every step."""
    num_games = 50
    board_size = 10
    rng = random.Random(20260418)
    for game_idx in range(num_games):
        state = new_game(board_size, 4, seed=game_idx)
        turns = 0
        while not state.done and turns < 400:
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
