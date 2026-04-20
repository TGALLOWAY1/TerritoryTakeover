"""Tests for the PPO observation encoder."""

from __future__ import annotations

import numpy as np

from territory_takeover.constants import CLAIMED_CODES, PATH_CODES
from territory_takeover.engine import new_game
from territory_takeover.rl.ppo.spaces import encode_observation
from territory_takeover.state import GameState, PlayerState


def _hand_built_4x4_2p_state() -> GameState:
    """Build a deterministic 4x4 / 2p state for channel-by-channel checks.

    Layout (P1 at 0,0 with claimed at 1,0; P2 at 3,3 with claimed at 3,2):
        1 . . .
        5 . . .
        . . . .
        . . B 2
    where 1=P1 path, 5=P1 claimed, 2=P2 path, B=P2 claimed.
    """
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[1, 0] = CLAIMED_CODES[0]
    grid[3, 3] = PATH_CODES[1]
    grid[3, 2] = CLAIMED_CODES[1]

    p1 = PlayerState(
        player_id=0,
        path=[(0, 0)],
        path_set={(0, 0)},
        head=(0, 0),
        claimed_count=1,
        alive=True,
    )
    p2 = PlayerState(
        player_id=1,
        path=[(3, 3)],
        path_set={(3, 3)},
        head=(3, 3),
        claimed_count=1,
        alive=True,
    )
    return GameState(
        grid=grid,
        players=[p1, p2],
        current_player=0,
        turn_number=2,
    )


def test_encode_observation_channel_layout_2p() -> None:
    state = _hand_built_4x4_2p_state()
    planes, scalars = encode_observation(state, player_id=0)

    assert planes.shape == (2 * 2 + 2, 4, 4)
    assert planes.dtype == np.float32

    # Channel 0 = active player's path (P1 at (0,0)).
    assert planes[0, 0, 0] == 1.0
    assert planes[0].sum() == 1.0

    # Channel 1 = opponent's path (P2 at (3,3)).
    assert planes[1, 3, 3] == 1.0
    assert planes[1].sum() == 1.0

    # Channel 2 = active player's claimed (P1 at (1,0)).
    assert planes[2, 1, 0] == 1.0
    assert planes[2].sum() == 1.0

    # Channel 3 = opponent's claimed (P2 at (3,2)).
    assert planes[3, 3, 2] == 1.0
    assert planes[3].sum() == 1.0

    # Channel 4 = EMPTY: 16 cells - 4 filled = 12.
    assert planes[4].sum() == 12.0

    # Channel 5 = active player's head, one-hot at (0, 0).
    assert planes[5, 0, 0] == 1.0
    assert planes[5].sum() == 1.0

    # Scalars: (turn_norm, p1_claim_norm, p2_claim_norm, fill_ratio, active_occ).
    # turn_number=2, total_cells=16 => 0.125
    # claim_norm = 1/16 = 0.0625 for both
    # fill_ratio = 4/16 = 0.25
    # active_occ = (1 path + 1 claim) / 16 = 0.125
    assert scalars.shape == (3 + 2,)
    np.testing.assert_allclose(
        scalars, np.array([0.125, 0.0625, 0.0625, 0.25, 0.125], dtype=np.float32)
    )


def test_encode_observation_rotation_invariance_2p() -> None:
    """Active player is always channel 0 of the path block."""
    state = _hand_built_4x4_2p_state()

    planes_p0, _ = encode_observation(state, player_id=0)
    planes_p1, _ = encode_observation(state, player_id=1)

    # P0 view: channel 0 is P1's path (at 0,0); P1 view: channel 0 is P2's.
    assert planes_p0[0, 0, 0] == 1.0
    assert planes_p0[0, 3, 3] == 0.0

    assert planes_p1[0, 0, 0] == 0.0
    assert planes_p1[0, 3, 3] == 1.0

    # The "opponent" channel for P0 (channel 1) equals the "self" channel for
    # P1 (channel 0).
    np.testing.assert_array_equal(planes_p0[1], planes_p1[0])
    np.testing.assert_array_equal(planes_p1[1], planes_p0[0])
    # Same swap for the claimed block.
    np.testing.assert_array_equal(planes_p0[3], planes_p1[2])
    np.testing.assert_array_equal(planes_p1[3], planes_p0[2])

    # EMPTY channel is view-independent.
    np.testing.assert_array_equal(planes_p0[4], planes_p1[4])

    # Head plane should mark each player's own head.
    assert planes_p0[5, 0, 0] == 1.0
    assert planes_p1[5, 3, 3] == 1.0


def test_encode_observation_scalar_ranges_on_real_game() -> None:
    """Scalars must lie in [0, 1] across a random 8x8 4p rollout."""
    rng = np.random.default_rng(42)
    state = new_game(board_size=8, num_players=4, seed=0)
    for _ in range(20):
        for pid in range(4):
            _, scalars = encode_observation(state, player_id=pid)
            assert np.all(scalars >= 0.0), scalars
            assert np.all(scalars <= 1.0), scalars

        from territory_takeover.actions import legal_actions
        from territory_takeover.engine import step

        legal = legal_actions(state, state.current_player)
        if not legal or state.done:
            break
        step(state, int(rng.choice(legal)), strict=False)


def test_encode_observation_4p_shapes() -> None:
    state = new_game(board_size=10, num_players=4, seed=1)
    planes, scalars = encode_observation(state, player_id=2)
    assert planes.shape == (2 * 4 + 2, 10, 10)
    assert scalars.shape == (3 + 4,)
    # Each player's path/claimed planes should be mutually exclusive.
    total_occupancy = planes[: 2 * 4].sum(axis=0)
    assert total_occupancy.max() <= 1.0


def test_encode_observation_handles_dead_player_head() -> None:
    """If the active player is dead and has sentinel head, head plane is zero."""
    state = _hand_built_4x4_2p_state()
    state.players[0].alive = False
    state.players[0].head = (-1, -1)

    planes, _ = encode_observation(state, player_id=0)
    # Head channel is index 2N + 1.
    assert planes[2 * 2 + 1].sum() == 0.0
