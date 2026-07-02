"""Liveness rule end-to-end: sealed regions, blockouts, and endgame filling.

Under the corrected rules a player dies exactly when no EMPTY cell is
reachable from their head through (own territory ∪ EMPTY) cells. These
tests exercise the rule through real games rather than hand-poked grids.
"""

from __future__ import annotations

import numpy as np

from territory_takeover.actions import claiming_actions, legal_actions
from territory_takeover.constants import EMPTY
from territory_takeover.engine import has_reachable_empty, new_game, step


def test_lone_player_fills_entire_board_then_dies() -> None:
    state = new_game(board_size=4, num_players=1, spawn_positions=[(0, 0)])
    rng = np.random.default_rng(0)
    guard = 0
    while not state.done:
        pid = state.current_player
        claims = claiming_actions(state, pid)
        acts = claims if claims else legal_actions(state, pid)
        step(state, int(rng.choice(acts)))
        guard += 1
        assert guard < 2000
    assert state.players[0].territory_count == 16, "lone player should fill the board"
    assert state.empty_count == 0
    assert state.winner == 0


def test_sealing_a_region_reserves_it_for_the_sealer() -> None:
    # 5x5. P0 walls off the left two columns by owning all of column 2;
    # P1 lives on the right. P1 must never claim any cell in columns 0-1.
    state = new_game(board_size=5, num_players=2, spawn_positions=[(0, 2), (0, 4)])
    grid = state.grid
    for r in range(1, 5):
        grid[r, 2] = 1
    state.players[0].territory_count = 5
    state.players[0].head = (4, 2)
    state.empty_count = int((grid == EMPTY).sum())

    rng = np.random.default_rng(3)
    guard = 0
    while not state.done:
        pid = state.current_player
        claims = claiming_actions(state, pid)
        acts = claims if claims else legal_actions(state, pid)
        step(state, int(rng.choice(acts)))
        guard += 1
        assert guard < 2000

    # Columns 0-1 (10 cells) were reachable only by P0 — and since P0 only
    # dies when no empty is reachable, it must have filled them completely.
    left = state.grid[:, :2]
    assert not np.any(left == 2), "P1 must never enter the sealed region"
    assert np.all(left == 1), "P0 must end up owning every sealed cell"
    assert state.players[0].territory_count >= 15, "P0 owns wall + sealed half at least"


def test_blocked_player_dies_while_board_still_has_empties() -> None:
    # 4x4: P1 confined to the corner while P0 owns the separating wall.
    state = new_game(board_size=4, num_players=2, spawn_positions=[(0, 1), (0, 0)])
    grid = state.grid
    grid[1, 0] = 1
    grid[1, 1] = 1
    state.players[0].territory_count = 3
    state.players[0].head = (1, 1)
    state.empty_count = int((grid == EMPTY).sum())

    assert not has_reachable_empty(state, 1), "P1 is sealed at spawn"

    rng = np.random.default_rng(0)
    p1_died_with_empties = False
    guard = 0
    while not state.done:
        pid = state.current_player
        claims = claiming_actions(state, pid)
        acts = claims if claims else legal_actions(state, pid)
        step(state, int(rng.choice(acts)))
        if not state.players[1].alive and state.empty_count > 0:
            p1_died_with_empties = True
        guard += 1
        assert guard < 2000

    assert p1_died_with_empties, "P1 must be marked dead before the board fills"
    assert state.players[1].territory_count == 1, "P1 keeps only its spawn cell"
    assert state.winner == 0


def test_traversal_back_through_own_cells_reaches_far_frontier() -> None:
    # P0 claims a corridor east along the top row, then walks back west
    # through its own cells (traversal, zero reward) to claim downward from
    # the spawn column — legal the whole way.
    state = new_game(board_size=3, num_players=1, spawn_positions=[(0, 0)])
    assert step(state, 3).reward == 1.0  # (0,1) claim
    assert step(state, 3).reward == 1.0  # (0,2) claim
    assert step(state, 2).reward == 0.0  # back to (0,1): traversal
    assert step(state, 2).reward == 0.0  # back to (0,0): traversal
    assert step(state, 1).reward == 1.0  # (1,0) claim
    assert state.players[0].territory_count == 4
    assert state.players[0].alive
    assert not state.done
