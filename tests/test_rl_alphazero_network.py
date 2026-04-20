"""Tests for the Phase 3c AlphaZero network.

Checks the contract the MCTS driver depends on:
- Forward-pass shapes for 2p and 4p configs.
- Value outputs in [-1, 1] (tanh + normalization assumption).
- Illegal logits are set to LOGIT_MASK_VALUE.
- Scalar vs 4-dim value head both wire up and produce right shapes.
- End-to-end path from a GameState through encode_az_observation into the
  network returns something sensible.
"""

from __future__ import annotations

import numpy as np
import torch

from territory_takeover.engine import new_game
from territory_takeover.rl.alphazero.network import (
    AlphaZeroNet,
    AZNetConfig,
    ResidualBlock,
)
from territory_takeover.rl.alphazero.spaces import (
    encode_az_observation,
    grid_channel_count,
    scalar_feature_dim,
)
from territory_takeover.rl.ppo.spaces import LOGIT_MASK_VALUE


def _tiny_cfg(num_players: int, board_size: int = 6) -> AZNetConfig:
    return AZNetConfig(
        board_size=board_size,
        num_players=num_players,
        num_res_blocks=1,
        channels=16,
        value_hidden=16,
    )


def test_forward_shapes_2p() -> None:
    cfg = _tiny_cfg(2)
    net = AlphaZeroNet(cfg)
    net.eval()
    batch = 3
    grid = torch.zeros(batch, cfg.grid_in_channels, cfg.board_size, cfg.board_size)
    scalars = torch.zeros(batch, cfg.scalar_feature_dim)
    mask = torch.ones(batch, 4, dtype=torch.bool)

    policy_logits, value = net(grid, scalars, mask)

    assert policy_logits.shape == (batch, 4)
    assert value.shape == (batch, 2)


def test_forward_shapes_4p() -> None:
    cfg = _tiny_cfg(4, board_size=5)
    net = AlphaZeroNet(cfg)
    net.eval()
    batch = 2
    grid = torch.zeros(batch, cfg.grid_in_channels, cfg.board_size, cfg.board_size)
    scalars = torch.zeros(batch, cfg.scalar_feature_dim)
    mask = torch.ones(batch, 4, dtype=torch.bool)

    policy_logits, value = net(grid, scalars, mask)

    assert policy_logits.shape == (batch, 4)
    assert value.shape == (batch, 4)


def test_value_in_tanh_range() -> None:
    cfg = _tiny_cfg(4, board_size=5)
    net = AlphaZeroNet(cfg)
    net.eval()
    grid = torch.randn(8, cfg.grid_in_channels, cfg.board_size, cfg.board_size)
    scalars = torch.randn(8, cfg.scalar_feature_dim)
    mask = torch.ones(8, 4, dtype=torch.bool)

    _, value = net(grid, scalars, mask)

    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_illegal_logits_are_masked() -> None:
    cfg = _tiny_cfg(2)
    net = AlphaZeroNet(cfg)
    net.eval()
    grid = torch.zeros(1, cfg.grid_in_channels, cfg.board_size, cfg.board_size)
    scalars = torch.zeros(1, cfg.scalar_feature_dim)
    mask = torch.tensor([[True, False, True, False]])

    policy_logits, _ = net(grid, scalars, mask)

    assert policy_logits[0, 1].item() == LOGIT_MASK_VALUE
    assert policy_logits[0, 3].item() == LOGIT_MASK_VALUE
    # Legal positions pass through unchanged (they are NOT LOGIT_MASK_VALUE).
    assert policy_logits[0, 0].item() > LOGIT_MASK_VALUE / 2
    assert policy_logits[0, 2].item() > LOGIT_MASK_VALUE / 2


def test_scalar_value_head_emits_single_output() -> None:
    cfg = AZNetConfig(
        board_size=6,
        num_players=4,
        num_res_blocks=1,
        channels=16,
        value_hidden=16,
        scalar_value_head=True,
    )
    net = AlphaZeroNet(cfg)
    net.eval()
    grid = torch.zeros(2, cfg.grid_in_channels, cfg.board_size, cfg.board_size)
    scalars = torch.zeros(2, cfg.scalar_feature_dim)
    mask = torch.ones(2, 4, dtype=torch.bool)

    _, value = net(grid, scalars, mask)
    assert value.shape == (2, 1)


def test_end_to_end_from_gamestate() -> None:
    state = new_game(board_size=6, num_players=4)
    cfg = _tiny_cfg(4, board_size=6)
    net = AlphaZeroNet(cfg)
    net.eval()

    planes, scalars = encode_az_observation(state, active_player=state.current_player)
    assert planes.shape == (grid_channel_count(4), 6, 6)
    assert scalars.shape == (scalar_feature_dim(4),)

    grid_t = torch.from_numpy(planes).unsqueeze(0)
    scalars_t = torch.from_numpy(scalars).unsqueeze(0)
    mask_t = torch.ones(1, 4, dtype=torch.bool)

    logits, value = net(grid_t, scalars_t, mask_t)
    assert logits.shape == (1, 4)
    assert value.shape == (1, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock(16)
    block.eval()
    x = torch.randn(2, 16, 5, 5)
    y = block(x)
    assert y.shape == x.shape


def test_determinism_under_manual_seed() -> None:
    cfg = _tiny_cfg(2)
    torch.manual_seed(0)
    net_a = AlphaZeroNet(cfg)
    net_a.eval()

    torch.manual_seed(0)
    net_b = AlphaZeroNet(cfg)
    net_b.eval()

    grid = torch.randn(4, cfg.grid_in_channels, cfg.board_size, cfg.board_size)
    scalars = torch.randn(4, cfg.scalar_feature_dim)
    mask = torch.ones(4, 4, dtype=torch.bool)

    with torch.no_grad():
        la, va = net_a(grid, scalars, mask)
        lb, vb = net_b(grid, scalars, mask)

    torch.testing.assert_close(la, lb)
    torch.testing.assert_close(va, vb)
    # Sanity: values are not trivially zero (would hide non-determinism).
    assert not np.allclose(va.numpy(), 0.0)
