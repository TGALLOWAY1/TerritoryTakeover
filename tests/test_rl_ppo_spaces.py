"""Tests for the PPO observation encoder and action-mask helper.

Plane layout (N + 2 channels): [0..N-1] per-player OWNED cells rotated so
the active player is channel 0, [N] EMPTY, [N+1] active-player head one-hot.
"""

from __future__ import annotations

import numpy as np
import torch

from territory_takeover.constants import OWNED_CODES
from territory_takeover.engine import new_game
from territory_takeover.rl.ppo.spaces import (
    LOGIT_MASK_VALUE,
    apply_action_mask,
    encode_observation,
    legal_mask_array,
    legal_mask_tensor,
)
from territory_takeover.state import GameState, PlayerState


def _hand_built_4x4_2p_state() -> GameState:
    """Build a deterministic 4x4 / 2p state for channel-by-channel checks.

    Layout (P1 owns (0,0)+(1,0) with head at (0,0); P2 owns (3,2)+(3,3) with
    head at (3,3)):
        1 . . .
        1 . . .
        . . . .
        . . 2 2
    """
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[0, 0] = OWNED_CODES[0]
    grid[1, 0] = OWNED_CODES[0]
    grid[3, 3] = OWNED_CODES[1]
    grid[3, 2] = OWNED_CODES[1]

    p1 = PlayerState(player_id=0, head=(0, 0), territory_count=2, alive=True)
    p2 = PlayerState(player_id=1, head=(3, 3), territory_count=2, alive=True)
    return GameState(
        grid=grid,
        players=[p1, p2],
        current_player=0,
        turn_number=2,
    )


def test_encode_observation_channel_layout_2p() -> None:
    state = _hand_built_4x4_2p_state()
    planes, scalars = encode_observation(state, player_id=0)

    assert planes.shape == (2 + 2, 4, 4)
    assert planes.dtype == np.float32

    # Channel 0 = active player's owned cells (P1 at (0,0) and (1,0)).
    assert planes[0, 0, 0] == 1.0
    assert planes[0, 1, 0] == 1.0
    assert planes[0].sum() == 2.0

    # Channel 1 = opponent's owned cells (P2 at (3,2) and (3,3)).
    assert planes[1, 3, 2] == 1.0
    assert planes[1, 3, 3] == 1.0
    assert planes[1].sum() == 2.0

    # Channel 2 = EMPTY: 16 cells - 4 owned = 12.
    assert planes[2].sum() == 12.0

    # Channel 3 = active player's head, one-hot at (0, 0).
    assert planes[3, 0, 0] == 1.0
    assert planes[3].sum() == 1.0

    # Scalars: (turn_norm, self_terr, opp_terr, fill_ratio, active_terr).
    # turn_number=2, total_cells=16 => 0.125
    # territory fraction = 2/16 = 0.125 for both seats
    # fill_ratio = 4/16 = 0.25
    # trailing scalar duplicates the active player's territory fraction.
    assert scalars.shape == (3 + 2,)
    np.testing.assert_allclose(
        scalars, np.array([0.125, 0.125, 0.125, 0.25, 0.125], dtype=np.float32)
    )


def test_encode_observation_rotation_invariance_2p() -> None:
    """Active player is always channel 0 of the owned block."""
    state = _hand_built_4x4_2p_state()

    planes_p0, _ = encode_observation(state, player_id=0)
    planes_p1, _ = encode_observation(state, player_id=1)

    # P0 view: channel 0 holds P1's cells (at 0,0); P1 view: channel 0 holds
    # P2's cells (at 3,3).
    assert planes_p0[0, 0, 0] == 1.0
    assert planes_p0[0, 3, 3] == 0.0

    assert planes_p1[0, 0, 0] == 0.0
    assert planes_p1[0, 3, 3] == 1.0

    # The "opponent" channel for P0 (channel 1) equals the "self" channel for
    # P1 (channel 0), and vice versa.
    np.testing.assert_array_equal(planes_p0[1], planes_p1[0])
    np.testing.assert_array_equal(planes_p1[1], planes_p0[0])

    # EMPTY channel is view-independent.
    np.testing.assert_array_equal(planes_p0[2], planes_p1[2])

    # Head plane should mark each player's own head.
    assert planes_p0[3, 0, 0] == 1.0
    assert planes_p1[3, 3, 3] == 1.0


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
    assert planes.shape == (4 + 2, 10, 10)
    assert scalars.shape == (3 + 4,)
    # The per-player owned planes should be mutually exclusive.
    total_occupancy = planes[:4].sum(axis=0)
    assert total_occupancy.max() <= 1.0


def test_encode_observation_handles_dead_player_head() -> None:
    """If the active player is dead and has sentinel head, head plane is zero."""
    state = _hand_built_4x4_2p_state()
    state.players[0].alive = False
    state.players[0].head = (-1, -1)

    planes, _ = encode_observation(state, player_id=0)
    # Head channel is index N + 1.
    assert planes[2 + 1].sum() == 0.0


def test_legal_mask_array_matches_engine_mask() -> None:
    state = new_game(board_size=6, num_players=2, seed=3)
    from territory_takeover.actions import legal_action_mask as engine_mask

    for pid in range(2):
        np.testing.assert_array_equal(
            legal_mask_array(state, pid), engine_mask(state, pid)
        )


def test_legal_mask_tensor_dtype_and_shape() -> None:
    state = new_game(board_size=6, num_players=2, seed=3)
    t = legal_mask_tensor(state, 0)
    assert t.dtype == torch.bool
    assert t.shape == (4,)


def test_apply_action_mask_replaces_illegal_logits() -> None:
    logits = torch.tensor([0.5, -1.0, 2.0, 0.1])
    mask = torch.tensor([True, False, True, False])
    out = apply_action_mask(logits, mask)
    assert out[0].item() == 0.5
    assert out[2].item() == 2.0
    assert out[1].item() == LOGIT_MASK_VALUE
    assert out[3].item() == LOGIT_MASK_VALUE


def test_apply_action_mask_softmax_is_zero_on_illegal() -> None:
    """Masking BEFORE softmax must push illegal probabilities to exactly 0."""
    logits = torch.tensor([[0.1, 5.0, -3.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
    masked = apply_action_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)

    # Illegal positions must be exactly zero in fp32 under LOGIT_MASK_VALUE=-1e9.
    assert probs[0, 1].item() == 0.0
    assert probs[0, 3].item() == 0.0
    assert probs[1, 0].item() == 0.0

    # Legal probabilities sum to 1.
    torch.testing.assert_close(
        probs.sum(dim=-1), torch.ones(2), rtol=1e-6, atol=1e-6
    )


def test_apply_action_mask_gradient_flow() -> None:
    """Gradients must flow through legal logits only."""
    logits = torch.tensor([0.5, -1.0, 2.0, 0.1], requires_grad=True)
    mask = torch.tensor([True, False, True, False])
    masked = apply_action_mask(logits, mask)
    logprobs = torch.log_softmax(masked, dim=-1)
    # Use the log-prob of a legal action.
    loss = -logprobs[0]
    loss.backward()  # type: ignore[no-untyped-call]

    assert logits.grad is not None
    # Legal positions receive non-zero gradient.
    assert logits.grad[0].abs().item() > 0.0
    assert logits.grad[2].abs().item() > 0.0
    # Illegal positions receive ~zero gradient (up to fp32 noise from softmax).
    assert abs(logits.grad[1].item()) < 1e-6
    assert abs(logits.grad[3].item()) < 1e-6


def test_apply_action_mask_shape_mismatch_raises() -> None:
    logits = torch.zeros(4)
    mask = torch.zeros(3, dtype=torch.bool)
    try:
        apply_action_mask(logits, mask)
    except ValueError as exc:
        assert "last dim" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched last-dim")
