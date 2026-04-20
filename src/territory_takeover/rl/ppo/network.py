"""Actor-critic CNN for Phase 3b PPO.

The network is intentionally small — the residual trunk lands in Phase 3c.
Layout:

1. Four Conv2d(3x3, padding=1, 64 channels) + ReLU on the ``(2N + 2, H, W)``
   grid tensor.
2. Global average pool to a ``(64,)`` feature vector.
3. Concatenate scalar features ``(3 + N,)``.
4. Shared MLP to a 256-dim embedding.
5. Two linear heads:
   - Policy: ``Linear(256, 4)`` -> logits masked via
     :func:`territory_takeover.rl.ppo.spaces.apply_action_mask` before the
     :class:`torch.distributions.Categorical` is constructed.
   - Value: ``Linear(256, 1)`` -> scalar expected return for the current
     player. A 4-dim value head is a Phase 3c concern.

``forward`` returns ``(dist, value)`` where ``dist`` is a masked
Categorical. Callers can then ``sample()`` / ``log_prob`` / ``entropy()`` and
will never draw illegal actions. Determinism given a fixed
``torch.manual_seed`` is covered by the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .spaces import apply_action_mask


@dataclass(frozen=True, slots=True)
class PpoNetConfig:
    """Architectural config for the actor-critic network.

    Kept small / explicit so it can round-trip through YAML without pulling
    torch into config loading. The scalar-feature dim is derived from
    :func:`spaces.encode_observation` — ``3 + num_players``.
    """

    board_size: int
    num_players: int
    num_conv_layers: int = 4
    channels: int = 64
    hidden_dim: int = 256

    @property
    def grid_in_channels(self) -> int:
        return 2 * self.num_players + 2

    @property
    def scalar_feature_dim(self) -> int:
        return 3 + self.num_players


class ActorCritic(nn.Module):
    """Shared conv trunk + policy and value heads.

    Example usage::

        cfg = PpoNetConfig(board_size=15, num_players=4)
        net = ActorCritic(cfg)
        grid = torch.zeros(B, cfg.grid_in_channels, cfg.board_size, cfg.board_size)
        scalars = torch.zeros(B, cfg.scalar_feature_dim)
        mask = torch.ones(B, 4, dtype=torch.bool)
        dist, value = net(grid, scalars, mask)
        action = dist.sample()
        logprob = dist.log_prob(action)
    """

    cfg: PpoNetConfig
    conv_trunk: nn.Sequential
    trunk_mlp: nn.Sequential
    policy_head: nn.Linear
    value_head: nn.Linear

    def __init__(self, cfg: PpoNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        conv_layers: list[nn.Module] = []
        in_ch = cfg.grid_in_channels
        for _ in range(cfg.num_conv_layers):
            conv_layers.append(
                nn.Conv2d(in_ch, cfg.channels, kernel_size=3, padding=1)
            )
            conv_layers.append(nn.ReLU(inplace=True))
            in_ch = cfg.channels
        self.conv_trunk = nn.Sequential(*conv_layers)

        mlp_in = cfg.channels + cfg.scalar_feature_dim
        self.trunk_mlp = nn.Sequential(
            nn.Linear(mlp_in, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(cfg.hidden_dim, 4)
        self.value_head = nn.Linear(cfg.hidden_dim, 1)

    def trunk(
        self, grid_planes: torch.Tensor, scalar_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute the 256-d shared embedding (batch dim required)."""
        if grid_planes.dim() != 4:
            raise ValueError(
                f"grid_planes must be 4D (B, C, H, W), got shape {tuple(grid_planes.shape)}"
            )
        if scalar_features.dim() != 2:
            raise ValueError(
                f"scalar_features must be 2D (B, F), got shape "
                f"{tuple(scalar_features.shape)}"
            )
        feat_map = self.conv_trunk(grid_planes)
        # Global average pool over (H, W) -> (B, C).
        pooled = feat_map.mean(dim=(2, 3))
        joined = torch.cat([pooled, scalar_features], dim=1)
        embedding: torch.Tensor = self.trunk_mlp(joined)
        return embedding

    def forward(
        self,
        grid_planes: torch.Tensor,
        scalar_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Return ``(masked Categorical, value)`` for a batch of observations."""
        emb = self.trunk(grid_planes, scalar_features)
        raw_logits = self.policy_head(emb)
        masked_logits = apply_action_mask(raw_logits, action_mask)
        value = self.value_head(emb).squeeze(-1)
        # Categorical uses logits (unnormalized log-probs) directly — no need
        # to softmax ourselves; the distribution handles masked-to-zero
        # probabilities correctly because LOGIT_MASK_VALUE is strongly negative.
        return Categorical(logits=masked_logits), value


__all__ = ["ActorCritic", "PpoNetConfig"]
