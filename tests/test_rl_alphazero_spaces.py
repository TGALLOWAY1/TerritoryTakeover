"""Tests for the Phase 3c AlphaZero observation encoder.

The AZ encoder differs from Phase 3b in two ways: fixed seat ordering (no
rotation) and an explicit per-seat turn one-hot. Both are non-negotiable
for the 4-dim value head to have stable seat semantics.

Plane layout (2N + 2 channels): [0..N-1] per-seat OWNED cells, [N] EMPTY,
[N+1] active-player head one-hot, [N+2..2N+1] per-seat turn one-hot.
"""

from __future__ import annotations

import numpy as np

from territory_takeover.constants import OWNED_CODES
from territory_takeover.engine import new_game
from territory_takeover.rl.alphazero.spaces import (
    encode_az_observation,
    grid_channel_count,
    scalar_feature_dim,
)
from territory_takeover.state import GameState, PlayerState


def _build_4x4_2p_state(current_player: int = 0) -> GameState:
    """4x4 / 2p state: P0 owns (0,0)+(1,0) with head (0,0); P1 owns (3,3)."""
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[0, 0] = OWNED_CODES[0]
    grid[1, 0] = OWNED_CODES[0]
    grid[3, 3] = OWNED_CODES[1]

    p1 = PlayerState(player_id=0, head=(0, 0), territory_count=2, alive=True)
    p2 = PlayerState(player_id=1, head=(3, 3), territory_count=1, alive=True)
    return GameState(
        grid=grid,
        players=[p1, p2],
        current_player=current_player,
        turn_number=2,
    )


def test_channel_count_helpers() -> None:
    assert grid_channel_count(2) == 2 * 2 + 2
    assert grid_channel_count(4) == 2 * 4 + 2
    assert scalar_feature_dim(2) == 3 + 2
    assert scalar_feature_dim(4) == 3 + 4


def test_fixed_seat_ordering_no_rotation_2p() -> None:
    """P0's owned cells are on channel 0 regardless of whose turn it is."""
    state_p0 = _build_4x4_2p_state(current_player=0)
    state_p1 = _build_4x4_2p_state(current_player=1)

    planes_when_p0_moves, _ = encode_az_observation(state_p0, active_player=0)
    planes_when_p1_moves, _ = encode_az_observation(state_p1, active_player=1)

    # P0's owned cells (0,0) and (1,0) live on channel 0 in BOTH encodings —
    # not rotated onto the active player's slot.
    assert planes_when_p0_moves[0, 0, 0] == 1.0
    assert planes_when_p0_moves[0, 1, 0] == 1.0
    assert planes_when_p1_moves[0, 0, 0] == 1.0
    assert planes_when_p1_moves[0, 1, 0] == 1.0
    assert planes_when_p0_moves[0].sum() == 2.0
    assert planes_when_p1_moves[0].sum() == 2.0

    # P1's owned cell (3, 3) lives on channel 1 in BOTH encodings.
    assert planes_when_p0_moves[1, 3, 3] == 1.0
    assert planes_when_p1_moves[1, 3, 3] == 1.0
    assert planes_when_p0_moves[1].sum() == 1.0
    assert planes_when_p1_moves[1].sum() == 1.0

    # EMPTY plane at channel N == 2: 16 cells - 3 owned = 13.
    assert planes_when_p0_moves[2].sum() == 13.0
    assert planes_when_p1_moves[2].sum() == 13.0


def test_turn_one_hot_exactly_one_plane_hot_2p() -> None:
    state = _build_4x4_2p_state(current_player=0)

    planes_p0, _ = encode_az_observation(state, active_player=0)
    planes_p1, _ = encode_az_observation(state, active_player=1)

    num_players = 2
    turn_block_start = num_players + 2  # == 4

    # Active player 0 -> plane 4 all ones, plane 5 all zeros.
    assert planes_p0[turn_block_start].sum() == 16.0
    assert planes_p0[turn_block_start + 1].sum() == 0.0

    # Active player 1 -> plane 5 all ones, plane 4 all zeros.
    assert planes_p1[turn_block_start + 1].sum() == 16.0
    assert planes_p1[turn_block_start].sum() == 0.0


def test_head_one_hot_follows_active_player() -> None:
    state = _build_4x4_2p_state(current_player=0)

    planes_p0, _ = encode_az_observation(state, active_player=0)
    planes_p1, _ = encode_az_observation(state, active_player=1)

    num_players = 2
    head_plane = num_players + 1  # == 3

    # Active 0: head at (0, 0).
    assert planes_p0[head_plane, 0, 0] == 1.0
    assert planes_p0[head_plane].sum() == 1.0

    # Active 1: head at (3, 3).
    assert planes_p1[head_plane, 3, 3] == 1.0
    assert planes_p1[head_plane].sum() == 1.0


def test_dead_active_player_head_plane_is_zero() -> None:
    state = _build_4x4_2p_state(current_player=0)
    state.players[0].alive = False
    state.players[0].head = (-1, -1)

    planes, _ = encode_az_observation(state, active_player=0)
    assert planes[2 + 1].sum() == 0.0


def test_scalars_are_seat_ordered_not_rotated() -> None:
    state = _build_4x4_2p_state(current_player=0)

    _, scalars_p0 = encode_az_observation(state, active_player=0)
    _, scalars_p1 = encode_az_observation(state, active_player=1)

    # Seat 0 owns 2/16 cells, seat 1 owns 1/16 => indices 1, 2 carry seat-0,
    # seat-1 territory fractions. They DO NOT swap when the active changes.
    assert scalars_p0[1] == scalars_p1[1] == 2.0 / 16.0
    assert scalars_p0[2] == scalars_p1[2] == 1.0 / 16.0

    # Trailing scalar duplicates the ACTIVE player's territory fraction.
    assert scalars_p0[2 + 2] == 2.0 / 16.0
    assert scalars_p1[2 + 2] == 1.0 / 16.0

    # Fill ratio at index N + 1: 3 owned cells / 16.
    assert scalars_p0[2 + 1] == np.float32(3.0 / 16.0)


def test_encoder_works_for_4p_on_fresh_game() -> None:
    state = new_game(board_size=6, num_players=4)
    planes, scalars = encode_az_observation(state, active_player=state.current_player)

    assert planes.shape == (grid_channel_count(4), 6, 6)
    assert planes.dtype == np.float32
    assert scalars.shape == (scalar_feature_dim(4),)
    assert scalars.dtype == np.float32

    # Each seat's spawn corner is on its own plane.
    for pid in range(4):
        assert planes[pid].sum() == 1.0, f"trial={pid}"

    # Exactly one turn plane is fully on.
    turn_block_start = 4 + 2
    turn_planes = planes[turn_block_start : turn_block_start + 4]
    per_plane_sum = turn_planes.reshape(4, -1).sum(axis=1)
    assert int((per_plane_sum == 36.0).sum()) == 1
    assert int((per_plane_sum == 0.0).sum()) == 3


def test_encoder_does_not_mutate_state() -> None:
    state = _build_4x4_2p_state(current_player=0)
    grid_before = state.grid.copy()
    encode_az_observation(state, active_player=0)
    np.testing.assert_array_equal(state.grid, grid_before)
