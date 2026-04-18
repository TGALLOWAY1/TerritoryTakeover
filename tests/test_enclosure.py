"""Tests for engine.detect_and_apply_enclosure."""

from __future__ import annotations

import numpy as np

from territory_takeover.constants import CLAIMED_CODES, PATH_CODES
from territory_takeover.engine import detect_and_apply_enclosure
from territory_takeover.state import GameState


def _lay_path(state: GameState, player_id: int, cells: list[tuple[int, int]]) -> None:
    """Install `cells` (in order) as player_id's path and write their path code to grid."""
    p = state.players[player_id]
    p.path = list(cells)
    p.path_set = set(cells)
    p.head = cells[-1]
    for r, c in cells:
        state.grid[r, c] = PATH_CODES[player_id]


def test_simple_3x3_loop_encloses_one_cell() -> None:
    state = GameState.empty(5, 5, num_players=1)
    path = [
        (1, 1), (1, 2), (1, 3),
        (2, 3),
        (3, 3), (3, 2), (3, 1),
        (2, 1),
    ]
    _lay_path(state, 0, path)

    count = detect_and_apply_enclosure(state, 0, placed_cell=(2, 1))

    assert count == 1
    assert state.grid[2, 2] == CLAIMED_CODES[0]
    assert state.players[0].claimed_count == 1


def test_loop_using_board_edge() -> None:
    # 5x5 board. Path uses row 0 and col 0 (the board edge) as part of the
    # boundary of the pocket. Interior = (1,1) and (1,2).
    state = GameState.empty(5, 5, num_players=1)
    path = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3),
        (2, 3), (2, 2), (2, 1), (2, 0),
        (1, 0),
    ]
    _lay_path(state, 0, path)

    count = detect_and_apply_enclosure(state, 0, placed_cell=(1, 0))

    assert count == 2
    assert state.grid[1, 1] == CLAIMED_CODES[0]
    assert state.grid[1, 2] == CLAIMED_CODES[0]
    assert state.players[0].claimed_count == 2


def test_straight_line_extension_no_enclosure() -> None:
    state = GameState.empty(5, 5, num_players=1)
    path = [(0, 0), (0, 1), (0, 2)]
    _lay_path(state, 0, path)
    grid_before = state.grid.copy()

    count = detect_and_apply_enclosure(state, 0, placed_cell=(0, 2))

    assert count == 0
    assert state.players[0].claimed_count == 0
    assert np.array_equal(state.grid, grid_before)


def test_single_tile_path_no_enclosure() -> None:
    state = GameState.empty(5, 5, num_players=1)
    _lay_path(state, 0, [(2, 2)])
    grid_before = state.grid.copy()

    count = detect_and_apply_enclosure(state, 0, placed_cell=(2, 2))

    assert count == 0
    assert state.players[0].claimed_count == 0
    assert np.array_equal(state.grid, grid_before)


def test_loop_encloses_opponent_path_tile() -> None:
    # 5x5 board. Player 0 lays a perimeter that walls off interior cells,
    # one of which holds a player-1 path tile. That tile must stay as
    # player-1's path (grid unchanged), and only surrounding empties are
    # claimed by player 0.
    state = GameState.empty(5, 5, num_players=2)

    # Opponent path tile at (2,2).
    _lay_path(state, 1, [(2, 2)])

    p0_path = [
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 4),
        (3, 4), (3, 3), (3, 2), (3, 1), (3, 0),
        (2, 0),
    ]
    _lay_path(state, 0, p0_path)

    count = detect_and_apply_enclosure(state, 0, placed_cell=(2, 0))

    # Interior empties are (2,1) and (2,3). (2,2) is opponent path.
    assert count == 2
    assert state.grid[2, 1] == CLAIMED_CODES[0]
    assert state.grid[2, 3] == CLAIMED_CODES[0]
    assert state.grid[2, 2] == PATH_CODES[1]  # opponent tile untouched
    assert state.players[0].claimed_count == 2
    assert state.players[1].path_set == {(2, 2)}
    assert state.players[1].claimed_count == 0


def test_nested_loops_no_double_claim() -> None:
    # 7x7 board. A pre-existing claimed tile at (3,3) represents territory
    # already captured by an earlier (smaller) loop. Now player 0 closes a
    # larger loop around it. Only the empty cells between the outer path
    # and the pre-claimed cell should be newly claimed; the claimed cell
    # itself stays claimed (and is NOT re-counted).
    state = GameState.empty(7, 7, num_players=1)

    # Pre-existing claimed cell at (3,3), claimed_count tracks it.
    state.grid[3, 3] = CLAIMED_CODES[0]
    state.players[0].claimed_count = 1

    outer = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 5),
        (3, 5), (4, 5), (5, 5),
        (5, 4), (5, 3), (5, 2), (5, 1),
        (4, 1), (3, 1),
        (2, 1),
    ]
    _lay_path(state, 0, outer)

    count = detect_and_apply_enclosure(state, 0, placed_cell=(2, 1))

    # Interior (2,2)-(2,4), (3,2)-(3,4), (4,2)-(4,4) = 9 cells, 1 already claimed.
    assert count == 8
    assert state.grid[3, 3] == CLAIMED_CODES[0]  # pre-existing stays
    for r in (2, 3, 4):
        for c in (2, 3, 4):
            assert state.grid[r, c] == CLAIMED_CODES[0]
    assert state.players[0].claimed_count == 9


def test_idempotent_second_call_claims_zero() -> None:
    # Calling the function again after the loop is already resolved should
    # still trigger (geometry unchanged) but claim 0 additional cells.
    state = GameState.empty(5, 5, num_players=1)
    path = [
        (1, 1), (1, 2), (1, 3),
        (2, 3),
        (3, 3), (3, 2), (3, 1),
        (2, 1),
    ]
    _lay_path(state, 0, path)

    first = detect_and_apply_enclosure(state, 0, placed_cell=(2, 1))
    second = detect_and_apply_enclosure(state, 0, placed_cell=(2, 1))

    assert first == 1
    assert second == 0
    assert state.players[0].claimed_count == 1


def _rectangular_perimeter_path(
    r1: int, c1: int, r2: int, c2: int
) -> list[tuple[int, int]]:
    """Return the cells of the perimeter of the rectangle [r1..r2] x [c1..c2]
    as a single path that closes on itself. The last cell is adjacent to the
    first, so placing it triggers enclosure detection."""
    cells: list[tuple[int, int]] = []
    # Top row left-to-right
    for c in range(c1, c2 + 1):
        cells.append((r1, c))
    # Right col top+1 to bottom
    for r in range(r1 + 1, r2 + 1):
        cells.append((r, c2))
    # Bottom row right-1 to left
    for c in range(c2 - 1, c1 - 1, -1):
        cells.append((r2, c))
    # Left col bottom-1 to top+1 (stop before closing onto (r1, c1))
    for r in range(r2 - 1, r1, -1):
        cells.append((r, c1))
    return cells


def test_random_consistency() -> None:
    board_size = 10
    for trial in range(10):
        rng = np.random.default_rng(trial)
        r1 = int(rng.integers(0, board_size - 3))
        c1 = int(rng.integers(0, board_size - 3))
        r2 = int(rng.integers(r1 + 2, board_size))
        c2 = int(rng.integers(c1 + 2, board_size))

        state = GameState.empty(board_size, board_size, num_players=1)
        path = _rectangular_perimeter_path(r1, c1, r2, c2)
        _lay_path(state, 0, path)

        before = int((state.grid == CLAIMED_CODES[0]).sum())
        count = detect_and_apply_enclosure(state, 0, placed_cell=path[-1])
        after = int((state.grid == CLAIMED_CODES[0]).sum())

        expected_interior = (r2 - r1 - 1) * (c2 - c1 - 1)
        assert count == expected_interior, f"trial={trial}"
        assert count == after - before, f"trial={trial}"
        assert state.players[0].claimed_count == after, f"trial={trial}"
