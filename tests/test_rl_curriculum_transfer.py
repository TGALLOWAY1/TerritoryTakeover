"""Tests for cross-stage weight transfer."""

from __future__ import annotations

import torch

from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig
from territory_takeover.rl.curriculum import transfer_weights


def _conv_cfg(board_size: int, num_players: int = 2) -> AZNetConfig:
    return AZNetConfig(
        board_size=board_size,
        num_players=num_players,
        num_res_blocks=1,
        channels=16,
        value_hidden=16,
        head_type="conv",
    )


def test_transfer_conv_trunk_matches_bitwise_across_board_sizes() -> None:
    torch.manual_seed(0)
    src = AlphaZeroNet(_conv_cfg(board_size=10))
    dst = AlphaZeroNet(_conv_cfg(board_size=20))

    report = transfer_weights(src.state_dict(), dst)

    # All parameters must match between source and destination under conv
    # heads when player count is unchanged — nothing is board-size-locked.
    assert report.shape_mismatched_keys == ()
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert set(report.matched_keys) == set(src.state_dict().keys())

    # Bitwise trunk equality.
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    for key in src_state:
        assert torch.equal(dst_state[key], src_state[key]), f"{key} did not copy bitwise"


def test_transfer_preserves_forward_on_larger_board() -> None:
    torch.manual_seed(1)
    src = AlphaZeroNet(_conv_cfg(board_size=10))
    dst = AlphaZeroNet(_conv_cfg(board_size=15))
    transfer_weights(src.state_dict(), dst)

    dst.eval()
    grid = torch.zeros(1, dst.cfg.grid_in_channels, 15, 15)
    scalars = torch.zeros(1, dst.cfg.scalar_feature_dim)
    mask = torch.ones(1, 4, dtype=torch.bool)

    logits, value = dst(grid, scalars, mask)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert logits.shape == (1, 4)
    assert value.shape == (1, 2)


def test_transfer_drops_fc_heads_going_to_conv() -> None:
    torch.manual_seed(2)
    fc_cfg = AZNetConfig(
        board_size=10,
        num_players=2,
        num_res_blocks=1,
        channels=16,
        value_hidden=16,
        head_type="fc",
    )
    src = AlphaZeroNet(fc_cfg)
    dst = AlphaZeroNet(_conv_cfg(board_size=15))

    report = transfer_weights(src.state_dict(), dst)

    # Conv trunk keys (stem, residual blocks, 1x1 head convs) exist in both.
    trunk_keys = [k for k in report.matched_keys if k.startswith(("stem_", "res_", "policy_conv", "policy_bn", "value_conv", "value_bn"))]
    assert len(trunk_keys) > 0

    # FC policy_fc weight shape (4, 200) != conv policy_fc shape (4, 2),
    # so it lands in shape_mismatched_keys.
    assert "policy_fc.weight" in report.shape_mismatched_keys
    assert "value_fc1.weight" in report.shape_mismatched_keys


def test_transfer_handles_differing_num_players_gracefully() -> None:
    torch.manual_seed(3)
    src = AlphaZeroNet(_conv_cfg(board_size=10, num_players=2))
    dst = AlphaZeroNet(_conv_cfg(board_size=10, num_players=4))

    report = transfer_weights(src.state_dict(), dst)

    # Stem conv has different in_channels (3*2+2=8 vs 3*4+2=14) and
    # value_fc2 has different out_features (2 vs 4). Both land in
    # shape_mismatched.
    assert "stem_conv.weight" in report.shape_mismatched_keys
    assert "value_fc2.weight" in report.shape_mismatched_keys

    # Residual blocks and head convs have no num_players dependence,
    # so they should transfer cleanly.
    assert any(k.startswith("res_blocks.0.conv1") for k in report.matched_keys)

    # Destination still does a finite forward after partial transfer.
    dst.eval()
    grid = torch.zeros(1, dst.cfg.grid_in_channels, 10, 10)
    scalars = torch.zeros(1, dst.cfg.scalar_feature_dim)
    mask = torch.ones(1, 4, dtype=torch.bool)
    logits, value = dst(grid, scalars, mask)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
