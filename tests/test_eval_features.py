"""Tests for the territory_takeover.eval subpackage."""

from __future__ import annotations

import math

import numpy as np

from territory_takeover.constants import (
    PLAYER_1_PATH,
    PLAYER_2_PATH,
    PLAYER_3_PATH,
)
from territory_takeover.eval import (
    head_opponent_distance,
    mobility,
    reachable_area,
    voronoi_partition,
)
from territory_takeover.state import GameState, PlayerState


def _make_state(
    grid: np.ndarray, heads: list[tuple[int, int]]
) -> GameState:
    players = [
        PlayerState(
            player_id=i,
            path=[h],
            path_set={h},
            head=h,
            claimed_count=0,
            alive=True,
        )
        for i, h in enumerate(heads)
    ]
    return GameState(grid=grid, players=players)


def test_voronoi_diagonal_split_5x5() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[0, 0] = PLAYER_1_PATH
    grid[4, 4] = PLAYER_2_PATH
    state = _make_state(grid, [(0, 0), (4, 4)])

    partition = voronoi_partition(state)

    for r in range(5):
        for c in range(5):
            msg = f"cell=({r},{c})"
            if r + c < 4:
                assert partition[r, c] == 0, msg
            elif r + c > 4:
                assert partition[r, c] == 1, msg
            else:
                assert partition[r, c] == -1, msg


def test_voronoi_walled_off_regions() -> None:
    # Row 2 is a solid wall separating P0 (top) from P1 (bottom).
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, :] = PLAYER_3_PATH
    grid[0, 0] = PLAYER_1_PATH
    grid[4, 4] = PLAYER_2_PATH
    state = _make_state(grid, [(0, 0), (4, 4)])

    partition = voronoi_partition(state)

    # Top half (rows 0, 1) belongs entirely to P0.
    for r in (0, 1):
        for c in range(5):
            assert partition[r, c] == 0, f"top cell=({r},{c})"
    # Wall row is unreachable.
    for c in range(5):
        assert partition[2, c] == -1, f"wall cell=(2,{c})"
    # Bottom half (rows 3, 4) belongs entirely to P1.
    for r in (3, 4):
        for c in range(5):
            assert partition[r, c] == 1, f"bottom cell=({r},{c})"

    assert reachable_area(state, 0) == 10
    assert reachable_area(state, 1) == 10


def test_reachable_area_conservation() -> None:
    # Mixed setup: some walls and claims, two living players.
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[1, 1] = PLAYER_1_PATH  # P0 head
    grid[4, 5] = PLAYER_2_PATH  # P1 head
    grid[2, 2] = PLAYER_3_PATH  # stray wall
    grid[3, 3] = PLAYER_3_PATH  # stray wall
    state = _make_state(grid, [(1, 1), (4, 5)])

    partition = voronoi_partition(state)
    num_alive = sum(1 for p in state.players if p.alive)
    empty_total = int(np.count_nonzero(grid == 0))
    contested_empty = int(np.count_nonzero((partition == -1) & (grid == 0)))
    won_sum = sum(reachable_area(state, p.player_id) for p in state.players if p.alive)

    assert won_sum + contested_empty == empty_total + num_alive


def test_mobility_zero_when_surrounded() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = PLAYER_1_PATH
    grid[1, 2] = PLAYER_2_PATH
    grid[3, 2] = PLAYER_2_PATH
    grid[2, 1] = PLAYER_2_PATH
    grid[2, 3] = PLAYER_2_PATH
    state = _make_state(grid, [(2, 2), (1, 2)])

    assert mobility(state, 0) == 0


def test_head_opponent_distance_symmetric_and_inf() -> None:
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[1, 1] = PLAYER_1_PATH
    grid[3, 4] = PLAYER_2_PATH
    state = _make_state(grid, [(1, 1), (3, 4)])

    assert head_opponent_distance(state, 0) == 5.0
    assert head_opponent_distance(state, 1) == 5.0

    state.players[1].alive = False
    assert head_opponent_distance(state, 0) == math.inf
