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
    choke_pressure,
    enclosure_potential,
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


def _make_state_with_path(
    grid: np.ndarray, paths: list[list[tuple[int, int]]]
) -> GameState:
    """Build a state where each player's path is the given ordered cell list.

    Grid must already have path codes painted on the corresponding cells.
    """
    players = [
        PlayerState(
            player_id=i,
            path=list(path),
            path_set=set(path),
            head=path[-1],
            claimed_count=0,
            alive=True,
        )
        for i, path in enumerate(paths)
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


def test_enclosure_potential_u_shape_one_step_close() -> None:
    # Player 1 has drawn a U-shape around a single interior cell (2,2).
    # From the head (3,1), one step up to (2,1) lands adjacent to (1,1),
    # closing a loop that encloses exactly (2,2).
    #
    #    0 1 2 3 4 5
    # 0  . . . . . .
    # 1  . 1 1 1 . .
    # 2  . 1 . 1 . .     <-- (2,2) will be the enclosed cell
    # 3  . H 1 1 . .     <-- H = head
    # 4  . . . . . .
    # 5  . . . . . .
    grid = np.zeros((6, 6), dtype=np.int8)
    path = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1)]
    for r, c in path:
        grid[r, c] = PLAYER_1_PATH
    state = _make_state_with_path(grid, [path])

    result = enclosure_potential(state, 0)
    assert result == (1, 1), f"got {result}"


def test_enclosure_potential_straight_line_returns_none() -> None:
    # A straight-line path in open space: the shortest "closing" route goes
    # around the outside of the line, but the resulting loop is a flat
    # rectangle that encloses zero cells. Must return None.
    grid = np.zeros((5, 5), dtype=np.int8)
    path = [(2, 0), (2, 1), (2, 2)]
    for r, c in path:
        grid[r, c] = PLAYER_1_PATH
    state = _make_state_with_path(grid, [path])

    assert enclosure_potential(state, 0) is None


def test_enclosure_potential_short_path_returns_none() -> None:
    # Path of length 2 (head + predecessor) cannot close any loop.
    grid = np.zeros((5, 5), dtype=np.int8)
    path = [(2, 2), (2, 3)]
    for r, c in path:
        grid[r, c] = PLAYER_1_PATH
    state = _make_state_with_path(grid, [path])

    assert enclosure_potential(state, 0) is None


def test_enclosure_potential_head_fully_walled_returns_none() -> None:
    # Head has no empty neighbours at all — BFS drains immediately.
    #
    #    0 1 2 3 4
    # 0  . . . . .
    # 1  . . 2 . .
    # 2  1 1 H 2 .     <-- H at (2,2); (2,1) is predecessor, rest are P2 walls
    # 3  . . 2 . .
    # 4  . . . . .
    grid = np.zeros((5, 5), dtype=np.int8)
    p1_path = [(2, 0), (2, 1), (2, 2)]
    for r, c in p1_path:
        grid[r, c] = PLAYER_1_PATH
    grid[1, 2] = PLAYER_2_PATH
    grid[3, 2] = PLAYER_2_PATH
    grid[2, 3] = PLAYER_2_PATH
    state = _make_state_with_path(grid, [p1_path, [(1, 2)]])

    assert enclosure_potential(state, 0) is None


def test_choke_pressure_single_player_empty_board_is_zero() -> None:
    # With no opponents, the player's Voronoi reach equals the solo BFS
    # reach (mod the head off-by-one that the feature adjusts for), so
    # choke pressure is exactly 0.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = PLAYER_1_PATH
    state = _make_state(grid, [(2, 2)])

    assert choke_pressure(state, 0) == 0.0


def test_choke_pressure_equidistant_opponent_is_one() -> None:
    # A 1x3 board with P0 at (0,0) and P1 at (0,2). The lone empty cell
    # (0,1) is equidistant from both heads, so it's contested (owner = -1).
    # P0's contested reach = 0 empty cells; solo reach = 1 empty cell.
    # choke_pressure = 1 - 0/1 = 1.0.
    grid = np.zeros((1, 3), dtype=np.int8)
    grid[0, 0] = PLAYER_1_PATH
    grid[0, 2] = PLAYER_2_PATH
    state = _make_state(grid, [(0, 0), (0, 2)])

    assert choke_pressure(state, 0) == 1.0


def test_choke_pressure_trapped_player_returns_one() -> None:
    # P0's head has zero empty neighbours — solo_bfs_area = 0 → return 1.0.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = PLAYER_1_PATH
    grid[1, 2] = PLAYER_2_PATH
    grid[3, 2] = PLAYER_2_PATH
    grid[2, 1] = PLAYER_2_PATH
    grid[2, 3] = PLAYER_2_PATH
    state = _make_state(grid, [(2, 2), (1, 2)])

    assert choke_pressure(state, 0) == 1.0


def test_choke_pressure_dead_player_returns_one() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = PLAYER_1_PATH
    grid[0, 0] = PLAYER_2_PATH
    state = _make_state(grid, [(2, 2), (0, 0)])
    state.players[0].alive = False

    assert choke_pressure(state, 0) == 1.0
