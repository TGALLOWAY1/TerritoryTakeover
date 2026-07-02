"""Tests for the territory_takeover.eval subpackage (features + Voronoi)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from territory_takeover.constants import EMPTY, OWNED_CODES
from territory_takeover.engine import new_game, step
from territory_takeover.eval import (
    choke_pressure,
    claiming_mobility,
    head_opponent_distance,
    mobility,
    reachable_area,
    reachable_area_feature,
    territory_total,
    voronoi_partition,
)
from territory_takeover.state import GameState, PlayerState


def _make_state(
    grid: NDArray[np.int8], heads: list[tuple[int, int]]
) -> GameState:
    """Build a GameState from a painted grid plus per-player head positions.

    ``territory_count`` is derived from the grid so the cache invariant
    holds; ``empty_count``/``alive_count`` are seeded consistently.
    """
    players = [
        PlayerState(
            player_id=i,
            head=h,
            territory_count=int(np.count_nonzero(grid == OWNED_CODES[i])),
            alive=True,
        )
        for i, h in enumerate(heads)
    ]
    state = GameState(grid=grid, players=players)
    state.alive_count = len(players)
    state.empty_count = int(np.count_nonzero(grid == EMPTY))
    return state


def test_voronoi_diagonal_split_5x5() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[0, 0] = OWNED_CODES[0]
    grid[4, 4] = OWNED_CODES[1]
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


def test_voronoi_seeds_entire_territory() -> None:
    # Under traversal rules any owned cell is a departure point, so the whole
    # territory is seeded at distance 0. P0's arm reaches across the top row
    # even though P0's head sits back at the spawn corner: (1, 3) is distance
    # 1 from P0's territory but distance 4 from P1's, so P0 wins it.
    grid = np.zeros((5, 5), dtype=np.int8)
    for c in range(4):
        grid[0, c] = OWNED_CODES[0]
    grid[4, 4] = OWNED_CODES[1]
    state = _make_state(grid, [(0, 0), (4, 4)])

    partition = voronoi_partition(state)

    assert partition[1, 3] == 0, "cell adjacent to P0's arm must belong to P0"
    # Owned cells carry their owner in the partition (seeded at distance 0).
    for c in range(4):
        assert partition[0, c] == 0, f"own cell (0,{c})"
    assert partition[4, 4] == 1


def test_voronoi_dead_player_territory_is_a_wall() -> None:
    # Row 2 belongs to a dead P2 — not seeded, but still a wall. It splits
    # the board into a P0 half (top) and a P1 half (bottom).
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, :] = OWNED_CODES[2]
    grid[0, 0] = OWNED_CODES[0]
    grid[4, 4] = OWNED_CODES[1]
    state = _make_state(grid, [(0, 0), (4, 4), (2, 0)])
    state.players[2].alive = False
    state.alive_count = 2

    partition = voronoi_partition(state)

    for r in (0, 1):
        for c in range(5):
            assert partition[r, c] == 0, f"top cell=({r},{c})"
    # Dead player's wall row is unowned in the partition.
    for c in range(5):
        assert partition[2, c] == -1, f"wall cell=(2,{c})"
    for r in (3, 4):
        for c in range(5):
            assert partition[r, c] == 1, f"bottom cell=({r},{c})"

    # reachable_area includes the player's own territory (the seed cells).
    assert reachable_area(state, 0) == 10
    assert reachable_area(state, 1) == 10
    assert reachable_area(state, 2) == 0


def test_voronoi_sealed_pocket_attributed_to_sealer() -> None:
    # An EMPTY pocket enclosed by one player's wall is reachable only by that
    # player, so the partition hands it to them automatically.
    grid = np.zeros((5, 5), dtype=np.int8)
    pocket_wall = [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    for r, c in pocket_wall:
        grid[r, c] = OWNED_CODES[0]
    grid[4, 4] = OWNED_CODES[1]
    state = _make_state(grid, [(0, 2), (4, 4)])

    partition = voronoi_partition(state)

    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert partition[r, c] == 0, f"pocket cell=({r},{c})"


def test_reachable_area_conservation() -> None:
    # Every cell is labelled with an alive player or -1, so summing the
    # per-player reachable areas plus the -1 cells recovers the whole board's
    # empty cells and alive territory.
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[1, 1] = OWNED_CODES[0]
    grid[1, 2] = OWNED_CODES[0]
    grid[4, 5] = OWNED_CODES[1]
    grid[2, 2] = OWNED_CODES[2]  # dead player's wall
    grid[3, 3] = OWNED_CODES[2]  # dead player's wall
    state = _make_state(grid, [(1, 1), (4, 5), (2, 2)])
    state.players[2].alive = False
    state.alive_count = 2

    partition = voronoi_partition(state)
    empty_total = int(np.count_nonzero(grid == EMPTY))
    unowned_empty = int(np.count_nonzero((partition == -1) & (grid == EMPTY)))
    alive_territory = sum(p.territory_count for p in state.players if p.alive)
    won_sum = sum(reachable_area(state, p.player_id) for p in state.players if p.alive)

    assert won_sum + unowned_empty == empty_total + alive_territory


def test_mobility_counts_traversal_and_claims() -> None:
    # From (2,2): N is an opponent wall, W is P0's own cell (traversal is
    # legal), S and E are empty claims. mobility = 3, claiming_mobility = 2.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = OWNED_CODES[0]
    grid[2, 1] = OWNED_CODES[0]
    grid[1, 2] = OWNED_CODES[1]
    state = _make_state(grid, [(2, 2), (1, 2)])

    assert mobility(state, 0) == 3
    assert claiming_mobility(state, 0) == 2


def test_mobility_zero_when_surrounded_by_opponents() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = OWNED_CODES[0]
    grid[1, 2] = OWNED_CODES[1]
    grid[3, 2] = OWNED_CODES[1]
    grid[2, 1] = OWNED_CODES[1]
    grid[2, 3] = OWNED_CODES[1]
    state = _make_state(grid, [(2, 2), (1, 2)])

    assert mobility(state, 0) == 0
    assert claiming_mobility(state, 0) == 0


def test_head_opponent_distance_symmetric_and_inf() -> None:
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[1, 1] = OWNED_CODES[0]
    grid[3, 4] = OWNED_CODES[1]
    state = _make_state(grid, [(1, 1), (3, 4)])

    assert head_opponent_distance(state, 0) == 5.0
    assert head_opponent_distance(state, 1) == 5.0

    state.players[1].alive = False
    assert head_opponent_distance(state, 0) == math.inf


def test_territory_total_tracks_claims_and_traversal() -> None:
    # Spawn counts as territory 1; a claim adds 1; a traversal move back over
    # an owned cell adds nothing.
    state = new_game(board_size=6, num_players=2, spawn_positions=[(2, 2), (5, 5)])
    assert territory_total(state, 0) == 1

    step(state, 3, strict=True)  # P0 east: claims (2, 3)
    assert territory_total(state, 0) == 2

    step(state, 0, strict=True)  # P1 north: claims (4, 5)
    step(state, 2, strict=True)  # P0 west: traversal back onto (2, 2)
    assert territory_total(state, 0) == 2


def test_reachable_area_feature_matches_voronoi_helper() -> None:
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[1, 1] = OWNED_CODES[0]
    grid[4, 4] = OWNED_CODES[1]
    state = _make_state(grid, [(1, 1), (4, 4)])

    for pid in range(2):
        assert reachable_area_feature(state, pid) == reachable_area(state, pid), f"pid={pid}"


def test_choke_pressure_single_player_empty_board_is_zero() -> None:
    # No opponents: the Voronoi reach equals the solo reach, so pressure is 0.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = OWNED_CODES[0]
    state = _make_state(grid, [(2, 2)])

    assert choke_pressure(state, 0) == 0.0


def test_choke_pressure_equidistant_opponent_is_one() -> None:
    # 1x3 board, P0 at (0,0), P1 at (0,2). The lone empty cell (0,1) is
    # equidistant from both territories, so it's contested (owner = -1).
    # P0's contested reach = 0 empty cells; solo reach = 1 empty cell.
    # choke_pressure = 1 - 0/1 = 1.0.
    grid = np.zeros((1, 3), dtype=np.int8)
    grid[0, 0] = OWNED_CODES[0]
    grid[0, 2] = OWNED_CODES[1]
    state = _make_state(grid, [(0, 0), (0, 2)])

    assert choke_pressure(state, 0) == 1.0


def test_choke_pressure_walled_out_player_returns_one() -> None:
    # P0's territory has zero empty neighbours: solo reach = 0 -> 1.0.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = OWNED_CODES[0]
    grid[1, 2] = OWNED_CODES[1]
    grid[3, 2] = OWNED_CODES[1]
    grid[2, 1] = OWNED_CODES[1]
    grid[2, 3] = OWNED_CODES[1]
    state = _make_state(grid, [(2, 2), (1, 2)])

    assert choke_pressure(state, 0) == 1.0


def test_choke_pressure_dead_player_returns_one() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = OWNED_CODES[0]
    grid[0, 0] = OWNED_CODES[1]
    state = _make_state(grid, [(2, 2), (0, 0)])
    state.players[0].alive = False
    state.alive_count = 1

    assert choke_pressure(state, 0) == 1.0


def test_choke_pressure_unspawned_player_returns_one() -> None:
    # A player whose head is still the (-1, -1) sentinel has no territory to
    # measure from; the feature must return the saturated value, not crash.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[0, 0] = OWNED_CODES[1]
    players = [
        PlayerState(player_id=0, head=(-1, -1), territory_count=0, alive=True),
        PlayerState(player_id=1, head=(0, 0), territory_count=1, alive=True),
    ]
    state = GameState(grid=grid, players=players)
    state.alive_count = 2
    state.empty_count = 24

    assert choke_pressure(state, 0) == 1.0
