"""Tests for rl.tabular.state_encoder.

Invariants covered:

1. Semantically identical states (same head + 4-neighborhood classes + phase)
   produce identical keys.
2. Moving the head one cell changes the key.
3. An opponent tile in a neighbor slot encodes as NBR_OPP regardless of
   which opponent owns it (2p and 4p both collapse to one "opp" class).
4. Out-of-bounds neighbors at corners produce NBR_OOB.
5. Own territory vs opponent territory map to distinct neighbor classes.
6. The phase bucket transitions at the documented thresholds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from territory_takeover.constants import EMPTY, OWNED_CODES
from territory_takeover.rl.tabular.state_encoder import (
    NBR_EMPTY,
    NBR_OOB,
    NBR_OPP,
    NBR_OWN,
    PHASE_EARLY,
    PHASE_END,
    PHASE_LATE,
    PHASE_MID,
    encode_state,
)
from territory_takeover.state import GameState, PlayerState


def _make_state(
    grid: NDArray[np.int8],
    heads: list[tuple[int, int]],
) -> GameState:
    players = []
    for i, head in enumerate(heads):
        players.append(
            PlayerState(
                player_id=i,
                head=head,
                territory_count=1,
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
        grid_a[3, 3] = OWNED_CODES[0]
        grid_b[3, 3] = OWNED_CODES[0]
        # Place an opponent-owned cell NORTH for both.
        grid_a[2, 3] = OWNED_CODES[1]
        grid_b[2, 3] = OWNED_CODES[1]
        state_a = _make_state(grid_a, [(3, 3), (0, 0)])
        state_b = _make_state(grid_b, [(3, 3), (0, 0)])
        key_a = encode_state(state_a, 0)
        key_b = encode_state(state_b, 0)
        assert key_a == key_b, f"trial={i}: {key_a} != {key_b}"


def test_encoder_head_move_changes_key() -> None:
    # Same grid content, head moved one cell — key must differ.
    for i in range(5):
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[3, 3] = OWNED_CODES[0]
        grid[3, 4] = OWNED_CODES[0]
        state_before = _make_state(grid, [(3, 3), (0, 0)])
        state_after = _make_state(grid, [(3, 4), (0, 0)])
        key_before = encode_state(state_before, 0)
        key_after = encode_state(state_after, 0)
        assert key_before != key_after, f"trial={i}: keys coincide"


def test_encoder_oob_at_corner_head() -> None:
    # Head at (0,0): N and W neighbors are out of bounds.
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[0, 0] = OWNED_CODES[0]
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
    # Place p1 territory north; separately place p2 and p3 territory north,
    # with the agent always being player 0. All three should encode to
    # NBR_OPP.
    for opp_id in range(1, 4):
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[3, 3] = OWNED_CODES[0]
        grid[2, 3] = OWNED_CODES[opp_id]
        heads: list[tuple[int, int]] = [(3, 3), (2, 3)]
        # For 3+ players, fill in extra player heads away from the head cell.
        if opp_id >= 2:
            heads = [*heads, (7, 0)]
        if opp_id >= 3:
            heads = [*heads, (0, 7)]
        state = _make_state(grid, heads)
        key = encode_state(state, 0)
        assert key[2] == NBR_OPP, f"opp_id={opp_id}: N neighbor class {key[2]}"


def test_encoder_own_vs_opponent_distinguished() -> None:
    # An own-territory tile adjacent vs an opponent tile adjacent must map to
    # different neighbor classes.
    grid_own = np.zeros((8, 8), dtype=np.int8)
    grid_own[3, 3] = OWNED_CODES[0]
    grid_own[2, 3] = OWNED_CODES[0]
    grid_opp = np.zeros((8, 8), dtype=np.int8)
    grid_opp[3, 3] = OWNED_CODES[0]
    grid_opp[2, 3] = OWNED_CODES[1]
    state_own = _make_state(grid_own, [(3, 3), (7, 7)])
    state_opp = _make_state(grid_opp, [(3, 3), (7, 7)])
    key_own = encode_state(state_own, 0)
    key_opp = encode_state(state_opp, 0)
    assert key_own[2] == NBR_OWN, f"own-territory class {key_own[2]}"
    assert key_opp[2] == NBR_OPP, f"opponent class {key_opp[2]}"
    assert key_own[2] != key_opp[2], f"{key_own[2]} == {key_opp[2]}"


def test_encoder_phase_transitions_at_thresholds() -> None:
    # Construct grids with controlled empty fractions: ~0.95, 0.70, 0.40, 0.10.
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
        # Fill the first `num_filled` cells with opponent territory so neither
        # head nor nbrs change.
        flat_indices = np.arange(num_filled)
        rows = flat_indices // size
        cols = flat_indices % size
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            grid[r, c] = OWNED_CODES[1]
        # Put own head at a cell guaranteed to still be empty.
        head = (size - 1, size - 1)
        grid[head] = OWNED_CODES[0]
        state = _make_state(grid, [head, (0, 0)])
        key = encode_state(state, 0)
        assert key[6] == expected_phase, (
            f"trial={i}: empty={empty_fraction} expected phase={expected_phase} "
            f"got {key[6]} (raw empty={(grid == EMPTY).mean():.3f})"
        )
