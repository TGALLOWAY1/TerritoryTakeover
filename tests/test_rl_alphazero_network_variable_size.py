"""Variable-board-size forward pass tests for Phase 3d curriculum.

Ensures the ``head_type="conv"`` variant of :class:`AlphaZeroNet` accepts
arbitrary ``H x W`` inputs without re-creation and that ``head_type="fc"``
remains strictly locked to its configured ``board_size``.
"""

from __future__ import annotations

import pytest
import torch

from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig


def _conv_cfg(num_players: int, board_size: int = 10) -> AZNetConfig:
    return AZNetConfig(
        board_size=board_size,
        num_players=num_players,
        num_res_blocks=1,
        channels=16,
        value_hidden=16,
        head_type="conv",
    )


def test_conv_head_accepts_multiple_board_sizes() -> None:
    cfg = _conv_cfg(num_players=2, board_size=10)
    net = AlphaZeroNet(cfg)
    net.eval()

    for h in (10, 15, 20, 40):
        grid = torch.zeros(1, cfg.grid_in_channels, h, h)
        scalars = torch.zeros(1, cfg.scalar_feature_dim)
        mask = torch.ones(1, 4, dtype=torch.bool)

        logits, value = net(grid, scalars, mask)
        assert logits.shape == (1, 4), f"H={h}: logits shape {logits.shape}"
        assert value.shape == (1, 2), f"H={h}: value shape {value.shape}"
        assert torch.isfinite(logits).all(), f"H={h}: non-finite logits"
        assert torch.isfinite(value).all(), f"H={h}: non-finite value"


def test_conv_head_4p_variable_size() -> None:
    cfg = _conv_cfg(num_players=4, board_size=10)
    net = AlphaZeroNet(cfg)
    net.eval()

    for h in (10, 15, 25, 40):
        grid = torch.zeros(2, cfg.grid_in_channels, h, h)
        scalars = torch.zeros(2, cfg.scalar_feature_dim)
        mask = torch.ones(2, 4, dtype=torch.bool)

        logits, value = net(grid, scalars, mask)
        assert logits.shape == (2, 4)
        assert value.shape == (2, 4)


def test_fc_head_rejects_different_board_size() -> None:
    cfg = AZNetConfig(
        board_size=8,
        num_players=2,
        num_res_blocks=1,
        channels=16,
        value_hidden=16,
        head_type="fc",
    )
    net = AlphaZeroNet(cfg)
    net.eval()

    wrong_h = 12
    grid = torch.zeros(1, cfg.grid_in_channels, wrong_h, wrong_h)
    scalars = torch.zeros(1, cfg.scalar_feature_dim)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(RuntimeError):
        net(grid, scalars, mask)


def test_invalid_head_type_raises() -> None:
    with pytest.raises(ValueError, match="head_type"):
        AZNetConfig(
            board_size=8,
            num_players=2,
            num_res_blocks=1,
            channels=16,
            value_hidden=16,
            head_type="bogus",  # type: ignore[arg-type]
        )


def test_conv_head_non_square_board() -> None:
    cfg = _conv_cfg(num_players=2, board_size=10)
    net = AlphaZeroNet(cfg)
    net.eval()

    grid = torch.zeros(1, cfg.grid_in_channels, 12, 18)
    scalars = torch.zeros(1, cfg.scalar_feature_dim)
    mask = torch.ones(1, 4, dtype=torch.bool)

    logits, value = net(grid, scalars, mask)
    assert logits.shape == (1, 4)
    assert value.shape == (1, 2)
