"""Tests for GameState / PlayerState data structures and copy semantics."""

from __future__ import annotations

import time

import numpy as np

from territory_takeover.constants import (
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    EMPTY,
    PLAYER_1_CLAIMED,
    PLAYER_1_PATH,
    PLAYER_2_CLAIMED,
    PLAYER_2_PATH,
    PLAYER_3_CLAIMED,
    PLAYER_3_PATH,
    PLAYER_4_CLAIMED,
    PLAYER_4_PATH,
)
from territory_takeover.state import GameState, PlayerState


def _make_small_state() -> GameState:
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[0, 0] = PLAYER_1_PATH
    grid[0, 1] = PLAYER_1_PATH
    grid[5, 5] = PLAYER_2_PATH
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
        path=[(5, 5)],
        path_set={(5, 5)},
        head=(5, 5),
        claimed_count=0,
        alive=True,
    )
    return GameState(
        grid=grid,
        players=[p0, p1],
        current_player=0,
        turn_number=0,
        winner=None,
        done=False,
    )


def test_game_state_copy_independence() -> None:
    original = _make_small_state()
    snapshot_grid = original.grid.copy()
    snapshot_p0_path = list(original.players[0].path)
    snapshot_p0_path_set = set(original.players[0].path_set)
    snapshot_p0_head = original.players[0].head
    snapshot_p0_claimed = original.players[0].claimed_count
    snapshot_p0_alive = original.players[0].alive
    snapshot_current = original.current_player
    snapshot_turn = original.turn_number
    snapshot_winner = original.winner
    snapshot_done = original.done

    clone = original.copy()

    # Grid mutation.
    clone.grid[2, 2] = PLAYER_3_PATH
    # Player 0 mutations.
    clone.players[0].path.append((0, 2))
    clone.players[0].path_set.add((0, 2))
    clone.players[0].head = (0, 2)
    clone.players[0].claimed_count += 5
    clone.players[0].alive = False
    # GameState scalars.
    clone.current_player = 1
    clone.turn_number = 42
    clone.winner = 1
    clone.done = True

    # Original unchanged in every dimension.
    assert np.array_equal(original.grid, snapshot_grid)
    assert original.players[0].path == snapshot_p0_path
    assert original.players[0].path_set == snapshot_p0_path_set
    assert original.players[0].head == snapshot_p0_head
    assert original.players[0].claimed_count == snapshot_p0_claimed
    assert original.players[0].alive == snapshot_p0_alive
    assert original.current_player == snapshot_current
    assert original.turn_number == snapshot_turn
    assert original.winner == snapshot_winner
    assert original.done == snapshot_done

    # Clone reflects mutations.
    assert clone.grid[2, 2] == PLAYER_3_PATH
    assert (0, 2) in clone.players[0].path_set
    assert clone.players[0].head == (0, 2)
    assert clone.done is True


def test_copy_does_not_share_mutable_containers() -> None:
    original = _make_small_state()
    clone = original.copy()

    assert clone.grid is not original.grid
    assert clone.players is not original.players
    for orig_p, clone_p in zip(original.players, clone.players, strict=True):
        assert orig_p is not clone_p
        assert orig_p.path is not clone_p.path
        assert orig_p.path_set is not clone_p.path_set


def test_repr_contains_legend_chars() -> None:
    grid = np.zeros((2, 8), dtype=np.int8)
    grid[0, 0] = EMPTY
    grid[0, 1] = PLAYER_1_PATH
    grid[0, 2] = PLAYER_2_PATH
    grid[0, 3] = PLAYER_3_PATH
    grid[0, 4] = PLAYER_4_PATH
    grid[1, 0] = PLAYER_1_CLAIMED
    grid[1, 1] = PLAYER_2_CLAIMED
    grid[1, 2] = PLAYER_3_CLAIMED
    grid[1, 3] = PLAYER_4_CLAIMED
    state = GameState(grid=grid, players=[])
    text = repr(state)

    for ch in (".", "1", "2", "3", "4", "A", "B", "C", "D"):
        assert ch in text
    assert "turn=0" in text
    assert "current_player=0" in text


def test_copy_performance_40x40(capsys: object) -> None:
    # Build a default-size state with realistic per-player path sizes.
    state = GameState.empty(
        height=DEFAULT_BOARD_HEIGHT,
        width=DEFAULT_BOARD_WIDTH,
        num_players=4,
    )
    for p in state.players:
        cells = [(p.player_id, c) for c in range(20)]
        p.path = list(cells)
        p.path_set = set(cells)
        p.head = cells[-1]

    iterations = 1000
    # Warmup.
    for _ in range(50):
        state.copy()

    start = time.perf_counter()
    for _ in range(iterations):
        state.copy()
    elapsed = time.perf_counter() - start
    mean_us = (elapsed / iterations) * 1e6

    # The engineering target is < 50 µs; assert a loose CI-friendly upper bound
    # that still catches catastrophic regressions (e.g. accidental deepcopy).
    print(f"GameState.copy() mean: {mean_us:.2f} µs over {iterations} iters")
    assert mean_us < 200.0, f"copy too slow: {mean_us:.2f} µs mean"
