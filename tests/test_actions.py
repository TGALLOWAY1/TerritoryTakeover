"""Legality checks under the corrected rules: claims + own-territory traversal."""

from __future__ import annotations

import numpy as np

from territory_takeover.actions import (
    action_to_coord,
    claiming_actions,
    has_any_legal_action,
    legal_action_mask,
    legal_actions,
)
from territory_takeover.engine import new_game, step


def test_spawn_corner_legality_2p() -> None:
    state = new_game(board_size=5, num_players=2)
    # P0 at (0,0): N and W are out of bounds; S and E are empty.
    assert legal_actions(state, 0) == [1, 3]
    # P1 at (4,4): S and E are out of bounds; N and W are empty.
    assert legal_actions(state, 1) == [0, 2]


def test_mask_agrees_with_legal_actions() -> None:
    state = new_game(board_size=6, num_players=4)
    rng = np.random.default_rng(0)
    for trial in range(200):
        if state.done:
            break
        pid = state.current_player
        mask = legal_action_mask(state, pid)
        acts = legal_actions(state, pid)
        assert [i for i in range(4) if mask[i]] == acts, f"trial={trial}"
        assert has_any_legal_action(state, pid) == bool(acts), f"trial={trial}"
        if acts:
            step(state, int(rng.choice(acts)))


def test_own_cells_are_traversable_and_reversal_is_legal() -> None:
    state = new_game(board_size=5, num_players=2)
    step(state, 1)  # P0 S -> (1,0), claims
    step(state, 0)  # P1 N -> (3,4), claims
    # P0 head at (1,0); N is P0's own spawn cell -> traversal legal (reversal).
    acts = legal_actions(state, 0)
    assert 0 in acts, "reversing onto own cell must be legal"
    assert 1 in acts and 3 in acts


def test_opponent_cells_are_walls() -> None:
    state = new_game(board_size=3, num_players=2, spawn_positions=[(0, 0), (0, 1)])
    # P0 at (0,0): E neighbor (0,1) is P1's cell -> illegal; S is empty.
    acts = legal_actions(state, 0)
    assert 3 not in acts
    assert acts == [1]


def test_claiming_actions_is_the_empty_target_subset() -> None:
    state = new_game(board_size=5, num_players=2)
    step(state, 1)  # P0 -> (1,0)
    step(state, 0)  # P1 -> (3,4)
    legal = legal_actions(state, 0)
    claims = claiming_actions(state, 0)
    assert set(claims) <= set(legal)
    # N is traversal (own spawn), so it is legal but not claiming.
    assert 0 in legal and 0 not in claims
    for a in claims:
        r, c = action_to_coord(state, 0, a)
        assert state.grid.item(r, c) == 0, f"action {a} target not empty"


def test_action_to_coord_matches_directions() -> None:
    state = new_game(board_size=5, num_players=2)
    # Head at (0,0).
    assert action_to_coord(state, 0, 0) == (-1, 0)
    assert action_to_coord(state, 0, 1) == (1, 0)
    assert action_to_coord(state, 0, 2) == (0, -1)
    assert action_to_coord(state, 0, 3) == (0, 1)


def test_fully_walled_player_has_no_legal_actions() -> None:
    # P1 alone in the corner (0,0); P0 owns (0,1) and (1,0), boxing P1 in.
    state = new_game(board_size=3, num_players=2, spawn_positions=[(0, 1), (0, 0)])
    state.grid[1, 0] = 1
    state.empty_count -= 1
    state.players[0].territory_count += 1
    assert legal_actions(state, 1) == []
    assert not has_any_legal_action(state, 1)
