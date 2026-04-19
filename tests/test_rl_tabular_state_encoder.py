"""Tests for rl.tabular.state_encoder.

Invariants covered:

1. Semantically identical states (same head + 4-neighborhood classes + phase)
   produce identical keys.
2. Moving the head one cell changes the key.
3. An opponent tile in a neighbor slot encodes as NBR_OPP_PATH regardless of
   which opponent owns it (2p and 4p both collapse to one "opp" class).
4. Out-of-bounds neighbors at corners produce NBR_OOB.
5. The phase bucket transitions at the documented thresholds.
"""

from __future__ import annotations

import numpy as np

from territory_takeover.constants import (
    CLAIMED_CODES,
    EMPTY,
    PATH_CODES,
)
from territory_takeover.rl.tabular.state_encoder import (
    NBR_EMPTY,
    NBR_OOB,
    NBR_OPP_CLAIM,
    NBR_OPP_PATH,
    NBR_OWN_PATH,
    PHASE_EARLY,
    PHASE_END,
    PHASE_LATE,
    PHASE_MID,
    encode_state,
)
from territory_takeover.state import GameState, PlayerState


def _make_state(
    grid: np.ndarray,
    heads: list[tuple[int, int]],
) -> GameState:
    players = []
    for i, head in enumerate(heads):
        players.append(
            PlayerState(
                player_id=i,
                path=[head],
                path_set={head},
                head=head,
                claimed_count=0,
                alive=True,
            )
        )
    return GameState(grid=grid, players=players)


def test_encoder_identical_states_produce_same_key() -> None:
    # Build two independent states with the same head, same neighborhood, same
    # phase and confirm they encode to the same key.
    for i in range(10):
        grid_a = np.zeros((8, 8), dtype=np.int8)
        grid_b = np.zeros((8, 8), dtype=np.int8)
        grid_a[3, 3] = PATH_CODES[0]
        grid_b[3, 3] = PATH_CODES[0]
        # Place an opponent path cell NORTH for both.
        grid_a[2, 3] = PATH_CODES[1]
        grid_b[2, 3] = PATH_CODES[1]
        state_a = _make_state(grid_a, [(3, 3), (0, 0)])
        state_b = _make_state(grid_b, [(3, 3), (0, 0)])
        key_a = encode_state(state_a, 0)
        key_b = encode_state(state_b, 0)
        assert key_a == key_b, f"trial={i}: {key_a} != {key_b}"


def test_encoder_head_move_changes_key() -> None:
    # Same grid content, head moved one cell — key must differ.
    for i in range(5):
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[3, 3] = PATH_CODES[0]
        grid[3, 4] = PATH_CODES[0]
        state_before = _make_state(grid, [(3, 3), (0, 0)])
        state_after = _make_state(grid, [(3, 4), (0, 0)])
        key_before = encode_state(state_before, 0)
        key_after = encode_state(state_after, 0)
        assert key_before != key_after, f"trial={i}: keys coincide"


def test_encoder_oob_at_corner_head() -> None:
    # Head at (0,0): N and W neighbors are out of bounds.
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    state = _make_state(grid, [(0, 0), (7, 7)])
    key = encode_state(state, 0)
    # Tuple layout: (r, c, N, S, W, E, phase)
    assert key[0] == 0
    assert key[1] == 0
    assert key[2] == NBR_OOB, f"N should be OOB, got {key[2]}"
    assert key[3] == NBR_EMPTY, f"S should be empty, got {key[3]}"
    assert key[4] == NBR_OOB, f"W should be OOB, got {key[4]}"
    assert key[5] == NBR_EMPTY, f"E should be empty, got {key[5]}"


def test_encoder_opponent_collapse_across_player_ids() -> None:
    # Place p1 tile north; separately place p2 and p3 tiles north, with the
    # agent always being player 0. All three should encode to NBR_OPP_PATH.
    for opp_id in range(1, 4):
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[3, 3] = PATH_CODES[0]
        grid[2, 3] = PATH_CODES[opp_id]
        heads: list[tuple[int, int]] = [(3, 3), (2, 3)]
        # For 3+ players, fill in extra player heads away from the head cell.
        if opp_id >= 2:
            heads = [*heads, (7, 0)]
        if opp_id >= 3:
            heads = [*heads, (0, 7)]
        state = _make_state(grid, heads)
        key = encode_state(state, 0)
        assert key[2] == NBR_OPP_PATH, f"opp_id={opp_id}: N neighbor class {key[2]}"
    # Also check that a CLAIMED code on the neighbor collapses to OPP_CLAIM.
    grid2 = np.zeros((8, 8), dtype=np.int8)
    grid2[3, 3] = PATH_CODES[0]
    grid2[2, 3] = CLAIMED_CODES[2]
    heads2: list[tuple[int, int]] = [(3, 3), (7, 0), (0, 7)]
    state2 = _make_state(grid2, heads2)
    key2 = encode_state(state2, 0)
    assert key2[2] == NBR_OPP_CLAIM, f"claimed neighbor class {key2[2]}"


def test_encoder_own_path_vs_claim_distinguished() -> None:
    # A own-path tile adjacent vs an own-claim tile adjacent must map to
    # different neighbor classes.
    grid_path = np.zeros((8, 8), dtype=np.int8)
    grid_path[3, 3] = PATH_CODES[0]
    grid_path[2, 3] = PATH_CODES[0]
    grid_claim = np.zeros((8, 8), dtype=np.int8)
    grid_claim[3, 3] = PATH_CODES[0]
    grid_claim[2, 3] = CLAIMED_CODES[0]
    state_path = _make_state(grid_path, [(3, 3), (7, 7)])
    state_claim = _make_state(grid_claim, [(3, 3), (7, 7)])
    key_path = encode_state(state_path, 0)
    key_claim = encode_state(state_claim, 0)
    assert key_path[2] == NBR_OWN_PATH, f"own-path class {key_path[2]}"
    assert key_claim[2] != key_path[2], f"{key_claim[2]} == {key_path[2]}"


def test_encoder_phase_transitions_at_thresholds() -> None:
    # Construct grids with controlled empty fractions: 1.0, 0.7, 0.4, 0.1.
    # Thresholds are 0.80, 0.55, 0.25 (strictly greater for upper bucket).
    test_cases = [
        (0.95, PHASE_EARLY),
        (0.70, PHASE_MID),
        (0.40, PHASE_LATE),
        (0.10, PHASE_END),
    ]
    for i, (empty_fraction, expected_phase) in enumerate(test_cases):
        size = 10
        total = size * size
        num_filled = round(total * (1.0 - empty_fraction))
        grid = np.zeros((size, size), dtype=np.int8)
        # Fill the first `num_filled` cells with an opponent path so neither
        # head nor nbrs change.
        flat_indices = np.arange(num_filled)
        rows = flat_indices // size
        cols = flat_indices % size
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            grid[r, c] = PATH_CODES[1]
        # Put own head at a cell guaranteed to still be empty.
        head = (size - 1, size - 1)
        grid[head] = PATH_CODES[0]
        state = _make_state(grid, [head, (0, 0)])
        key = encode_state(state, 0)
        assert key[6] == expected_phase, (
            f"trial={i}: empty={empty_fraction} expected phase={expected_phase} "
            f"got {key[6]} (raw empty={(grid == EMPTY).mean():.3f})"
        )
