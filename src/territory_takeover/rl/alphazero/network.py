"""ResNet policy/value network for Phase 3c AlphaZero.

Layout:

1. Stem: Conv2d(C_in -> channels, 3x3) + BN + ReLU.
2. Residual trunk: ``num_res_blocks`` blocks, each Conv-BN-ReLU-Conv-BN +
   identity skip + ReLU.
3. Policy head: 1x1 Conv(channels -> 2) + BN + ReLU + flatten +
   Linear(2*H*W -> 4). Logits are masked via
   :func:`territory_takeover.rl.ppo.spaces.apply_action_mask`.
4. Value head: 1x1 Conv(channels -> 1) + BN + ReLU + flatten +
   Linear(H*W -> value_hidden) + ReLU + Linear(value_hidden -> num_players)
   + tanh. Output is in ``[-1, 1]`` per seat, matching the AlphaGo Zero
   scheme but generalized to ``N`` seats as in Petosa & Balch (2019).

Scalar features are injected into the value head MLP (concatenated after
the flattened conv output) rather than the conv trunk itself: the scalars
are positional-invariant quantities (turn, per-seat claim fractions) and
tile-replicating them into every (H, W) cell inflates the parameter count
without adding information.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from territory_takeover.rl.alphazero.spaces import (
    grid_channel_count,
    scalar_feature_dim,
)
from territory_takeover.rl.ppo.spaces import apply_action_mask


@dataclass(frozen=True, slots=True)
class AZNetConfig:
    """Architectural config for the AlphaZero net.

    Depth (``num_res_blocks``) and width (``channels``) are meant to be
    swept; defaults match the phase plan (10 blocks, 128 channels). Small
    configs (1-2 blocks, 32-64 channels) are used by the smoke tests.
    """

    board_size: int
    num_players: int
    num_res_blocks: int = 10
    channels: int = 128
    value_hidden: int = 128
    scalar_value_head: bool = False
    """If True, emit a single scalar "active player" value instead of a
    per-seat vector. Used exclusively for the 4-dim vs scalar ablation;
    keep False for the main Phase 3c runs.
    """

    @property
    def grid_in_channels(self) -> int:
        return grid_channel_count(self.num_players)

    @property
    def scalar_feature_dim(self) -> int:
        return scalar_feature_dim(self.num_players)

    @property
    def value_out_dim(self) -> int:
        return 1 if self.scalar_value_head else self.num_players


class ResidualBlock(nn.Module):  # type: ignore[misc]
    """Standard pre-activation-free residual block.

    Two 3x3 convs with BatchNorm, identity skip, ReLU after add. Kept
    deliberately plain — no bottleneck, no squeeze-excite — to match the
    AlphaGo Zero baseline.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return torch.relu(out)


class AlphaZeroNet(nn.Module):  # type: ignore[misc]
    """ResNet trunk + masked policy head + per-seat value head.

    ``forward(grid, scalars, mask)`` returns ``(policy_logits, value)``:

    - ``policy_logits``: ``(B, 4)`` float, already masked — legal positions
      keep their logit, illegal positions are set to ``LOGIT_MASK_VALUE``.
      Callers wrap into ``torch.distributions.Categorical(logits=...)`` or
      softmax directly for MCTS priors.
    - ``value``: ``(B, N)`` float in ``[-1, 1]`` under the default
      per-seat head, or ``(B, 1)`` under ``scalar_value_head=True``.
    """

    def __init__(self, cfg: AZNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.stem_conv = nn.Conv2d(
            cfg.grid_in_channels, cfg.channels, kernel_size=3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(cfg.channels)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(cfg.channels) for _ in range(cfg.num_res_blocks)]
        )

        # Policy head.
        self.policy_conv = nn.Conv2d(cfg.channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * cfg.board_size * cfg.board_size, 4)

        # Value head.
        self.value_conv = nn.Conv2d(cfg.channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(
            cfg.board_size * cfg.board_size + cfg.scalar_feature_dim,
            cfg.value_hidden,
        )
        self.value_fc2 = nn.Linear(cfg.value_hidden, cfg.value_out_dim)

    def forward(
        self,
        grid_planes: torch.Tensor,
        scalar_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if grid_planes.dim() != 4:
            raise ValueError(
                f"grid_planes must be 4D (B, C, H, W); got {tuple(grid_planes.shape)}"
            )
        if scalar_features.dim() != 2:
            raise ValueError(
                f"scalar_features must be 2D (B, F); got {tuple(scalar_features.shape)}"
            )

        x = self.stem_conv(grid_planes)
        x = self.stem_bn(x)
        x = torch.relu(x)
        for block in self.res_blocks:
            x = block(x)

        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = torch.relu(p)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)
        policy_logits = apply_action_mask(policy_logits, action_mask)

        v = self.value_conv(x)
        v = self.value_bn(v)
        v = torch.relu(v)
        v = v.flatten(start_dim=1)
        v = torch.cat([v, scalar_features], dim=1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


__all__ = ["AZNetConfig", "AlphaZeroNet", "ResidualBlock"]
