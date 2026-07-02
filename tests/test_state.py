"""Tests for GameState / PlayerState data structures and copy semantics."""

from __future__ import annotations

import time

import numpy as np

from territory_takeover.constants import (
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    EMPTY,
    PLAYER_1_OWNED,
    PLAYER_2_OWNED,
    PLAYER_3_OWNED,
    PLAYER_4_OWNED,
)
from territory_takeover.state import GameState, PlayerState


def _make_small_state() -> GameState:
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[0, 0] = PLAYER_1_OWNED
    grid[0, 1] = PLAYER_1_OWNED
    grid[5, 5] = PLAYER_2_OWNED
    p0 = PlayerState(
        player_id=0,
        head=(0, 1),
        territory_count=2,
        alive=True,
    )
    p1 = PlayerState(
        player_id=1,
        head=(5, 5),
        territory_count=1,
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
    original.players[0].alive_witness = (0, 2)
    snapshot_grid = original.grid.copy()
    snapshot_p0_head = original.players[0].head
    snapshot_p0_territory = original.players[0].territory_count
    snapshot_p0_alive = original.players[0].alive
    snapshot_current = original.current_player
    snapshot_turn = original.turn_number
    snapshot_winner = original.winner
    snapshot_done = original.done

    clone = original.copy()

    # Witness cache is propagated (it is valid across copies).
    assert clone.players[0].alive_witness == (0, 2)

    # Grid mutation.
    clone.grid[2, 2] = PLAYER_3_OWNED
    # Player 0 mutations.
    clone.players[0].head = (0, 2)
    clone.players[0].territory_count += 5
    clone.players[0].alive = False
    clone.players[0].alive_witness = None
    # GameState scalars.
    clone.current_player = 1
    clone.turn_number = 42
    clone.winner = 1
    clone.done = True

    # Original unchanged in every dimension.
    assert np.array_equal(original.grid, snapshot_grid)
    assert original.players[0].head == snapshot_p0_head
    assert original.players[0].territory_count == snapshot_p0_territory
    assert original.players[0].alive == snapshot_p0_alive
    assert original.players[0].alive_witness == (0, 2)
    assert original.current_player == snapshot_current
    assert original.turn_number == snapshot_turn
    assert original.winner == snapshot_winner
    assert original.done == snapshot_done

    # Clone reflects mutations.
    assert clone.grid[2, 2] == PLAYER_3_OWNED
    assert clone.players[0].head == (0, 2)
    assert clone.done is True


def test_copy_does_not_share_mutable_containers() -> None:
    original = _make_small_state()
    clone = original.copy()

    assert clone.grid is not original.grid
    assert clone.players is not original.players
    for orig_p, clone_p in zip(original.players, clone.players, strict=True):
        assert orig_p is not clone_p


def test_copy_does_not_share_scratch_buffer() -> None:
    original = _make_small_state()
    original._scratch_reachable = np.zeros((6, 6), dtype=np.int32)
    original._reach_stamp = 17
    clone = original.copy()
    assert clone._scratch_reachable is None
    assert clone._reach_stamp == 0


def test_repr_contains_legend_chars() -> None:
    grid = np.zeros((2, 8), dtype=np.int8)
    grid[0, 0] = EMPTY
    grid[0, 1] = PLAYER_1_OWNED
    grid[0, 2] = PLAYER_2_OWNED
    grid[0, 3] = PLAYER_3_OWNED
    grid[0, 4] = PLAYER_4_OWNED
    state = GameState(grid=grid, players=[])
    text = repr(state)

    for ch in (".", "1", "2", "3", "4"):
        assert ch in text
    assert "turn=0" in text
    assert "current_player=0" in text


def test_empty_factory_seeds_counts() -> None:
    state = GameState.empty(height=5, width=7, num_players=3)
    assert state.grid.shape == (5, 7)
    assert state.empty_count == 35
    assert state.alive_count == 3
    for i, p in enumerate(state.players):
        assert p.player_id == i
        assert p.head == (-1, -1)
        assert p.territory_count == 0
        assert p.alive


def test_copy_performance_40x40(capsys: object) -> None:
    state = GameState.empty(
        height=DEFAULT_BOARD_HEIGHT,
        width=DEFAULT_BOARD_WIDTH,
        num_players=4,
    )
    for p in state.players:
        p.head = (p.player_id, 0)
        p.territory_count = 20

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
