"""PPO primitives: rollout buffer, GAE, and the clipped update step.

Adapted from CleanRL's reference implementation
(https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py). The math
and the minibatch loop structure follow that file directly; deviations:

- Observations are two tensors (grid, scalars) plus an action mask instead of
  a single flat vector, because the network in :mod:`.network` reads both.
- The update step enforces logit-level action masking via
  :func:`.spaces.apply_action_mask` (applied inside the network forward) so
  stale logits from the rollout are never trusted — masks are re-applied
  from stored legal actions each minibatch.

The training loop itself (env stepping, opponent pool, logging) lives in
``train.py`` which is a later-session deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 — standard torch alias

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .network import ActorCritic


@dataclass(slots=True)
class RolloutBatch:
    """A flat minibatch drawn from a :class:`RolloutBuffer`.

    All tensors share leading dim ``M`` (minibatch size). ``advantages`` are
    pre-normalized (zero mean, unit std) by the PPO update step, not by the
    buffer, so callers can inspect raw values if they wish.
    """

    grid: torch.Tensor
    scalars: torch.Tensor
    mask: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """Fixed-capacity rollout storage for ``T`` timesteps across ``N`` envs.

    Layout is ``(T, N, ...)``; consumers flatten to ``(T * N, ...)`` at
    training time. All tensors are preallocated on construction so the hot
    loop only writes into contiguous memory.

    The ``dones`` flag marks terminal transitions *at the time of the reward*
    — i.e. ``dones[t] == 1`` means ``rewards[t]`` includes the terminal
    reward and ``values[t + 1]`` should be zeroed for GAE.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        grid_shape: tuple[int, int, int],
        scalar_dim: int,
        num_actions: int = 4,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = torch.device(device)

        t, n = num_steps, num_envs
        c, h, w = grid_shape
        self.grid = torch.zeros(t, n, c, h, w, dtype=torch.float32, device=device)
        self.scalars = torch.zeros(t, n, scalar_dim, dtype=torch.float32, device=device)
        self.mask = torch.zeros(t, n, num_actions, dtype=torch.bool, device=device)
        self.actions = torch.zeros(t, n, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(t, n, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(t, n, dtype=torch.float32, device=device)
        self.dones = torch.zeros(t, n, dtype=torch.float32, device=device)
        self.values = torch.zeros(t, n, dtype=torch.float32, device=device)

        self._ptr = 0

    def reset(self) -> None:
        """Rewind the write pointer. Does not zero memory."""
        self._ptr = 0

    def full(self) -> bool:
        return self._ptr >= self.num_steps

    def add(
        self,
        grid: torch.Tensor,
        scalars: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write one timestep (all envs) into the buffer."""
        if self._ptr >= self.num_steps:
            raise RuntimeError(
                f"RolloutBuffer is full ({self._ptr}/{self.num_steps}); call reset()"
            )
        t = self._ptr
        self.grid[t] = grid
        self.scalars[t] = scalars
        self.mask[t] = mask
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self._ptr += 1

    def flat_view(
        self, advantages: torch.Tensor, returns: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return a dict of tensors flattened to shape ``(T * N, ...)``.

        ``advantages`` and ``returns`` are passed in from :func:`compute_gae`
        so the buffer itself stays agnostic to the advantage estimator used.
        """
        t, n = self.num_steps, self.num_envs
        return {
            "grid": self.grid.reshape(t * n, *self.grid.shape[2:]),
            "scalars": self.scalars.reshape(t * n, self.scalars.shape[2]),
            "mask": self.mask.reshape(t * n, self.mask.shape[2]),
            "actions": self.actions.reshape(t * n),
            "logprobs": self.logprobs.reshape(t * n),
            "values": self.values.reshape(t * n),
            "advantages": advantages.reshape(t * n),
            "returns": returns.reshape(t * n),
        }

    def iter_minibatches(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        minibatch_size: int,
        rng: np.random.Generator,
    ) -> list[RolloutBatch]:
        """Shuffle the flat view and split into minibatches of ``minibatch_size``."""
        flat = self.flat_view(advantages, returns)
        total = self.num_steps * self.num_envs
        if total % minibatch_size != 0:
            raise ValueError(
                f"Total rollout transitions {total} must be divisible by "
                f"minibatch_size {minibatch_size}"
            )
        indices: NDArray[np.int64] = rng.permutation(total)
        idx_tensor = torch.from_numpy(indices).to(self.device)

        batches: list[RolloutBatch] = []
        for start in range(0, total, minibatch_size):
            sel = idx_tensor[start : start + minibatch_size]
            batches.append(
                RolloutBatch(
                    grid=flat["grid"][sel],
                    scalars=flat["scalars"][sel],
                    mask=flat["mask"][sel],
                    actions=flat["actions"][sel],
                    logprobs=flat["logprobs"][sel],
                    advantages=flat["advantages"][sel],
                    returns=flat["returns"][sel],
                    values=flat["values"][sel],
                )
            )
        return batches


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation, adapted from CleanRL's ppo.py.

    Reference:
    https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py (approx.
    lines 281-298 of the commit referenced at plan time).

    Arguments
    ---------
    rewards : ``(T, N)`` tensor of per-step rewards.
    values  : ``(T, N)`` tensor of value-function estimates at each step.
    dones   : ``(T, N)`` tensor of 0/1 terminal flags aligned with rewards.
    bootstrap_value : ``(N,)`` tensor — the value of the state *after* the
        last recorded step, used to bootstrap the advantage at the tail. Set
        to zero for envs whose last transition had ``done == 1``.
    gamma   : discount factor.
    lam     : GAE lambda.

    Returns
    -------
    advantages : ``(T, N)``.
    returns    : ``(T, N)`` — ``advantages + values``, the value target used
        by the critic.
    """
    t, n = rewards.shape
    if values.shape != (t, n):
        raise ValueError(f"values shape {values.shape} != rewards {rewards.shape}")
    if dones.shape != (t, n):
        raise ValueError(f"dones shape {dones.shape} != rewards {rewards.shape}")
    if bootstrap_value.shape != (n,):
        raise ValueError(
            f"bootstrap_value shape {bootstrap_value.shape} != ({n},)"
        )

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(n, dtype=rewards.dtype, device=rewards.device)

    for step in reversed(range(t)):
        if step == t - 1:
            next_non_terminal = 1.0 - dones[step]
            next_value = bootstrap_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_value = values[step + 1]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[step] = last_gae

    returns = advantages + values
    return advantages, returns


@dataclass(slots=True)
class PpoConfig:
    """Hyperparameters for :func:`ppo_update`.

    Defaults track the task prompt: clip=0.2, 4 epochs per rollout (looped by
    caller), minibatch 256. Learning rate is owned by the optimizer.
    """

    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    clip_value_loss: bool = True


def ppo_update(
    network: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    cfg: PpoConfig,
) -> dict[str, float]:
    """One SGD step of the clipped PPO objective on ``batch``.

    Returns a dict of scalar loss diagnostics (policy/value/entropy/kl/
    clip_frac) suitable for tensorboard. The function calls ``backward`` on
    the combined loss and ``optimizer.step``; callers own epoch / minibatch
    looping and LR scheduling.
    """
    dist, new_values = network(batch.grid, batch.scalars, batch.mask)
    new_logprobs = dist.log_prob(batch.actions)
    entropy = dist.entropy().mean()

    advantages = batch.advantages
    if cfg.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Clipped policy loss (surrogate objective), per CleanRL.
    log_ratio = new_logprobs - batch.logprobs
    ratio = log_ratio.exp()
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss with optional CleanRL-style clipping around the old value.
    if cfg.clip_value_loss:
        v_clipped = batch.values + torch.clamp(
            new_values - batch.values, -cfg.clip_eps, cfg.clip_eps
        )
        v_loss_unclipped = (new_values - batch.returns).pow(2)
        v_loss_clipped = (v_clipped - batch.returns).pow(2)
        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        value_loss = 0.5 * F.mse_loss(new_values, batch.returns)

    loss = policy_loss - cfg.entropy_coef * entropy + cfg.value_coef * value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), cfg.max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_frac = ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()

    return {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "approx_kl": float(approx_kl),
        "clip_frac": float(clip_frac),
    }


__all__ = [
    "PpoConfig",
    "RolloutBatch",
    "RolloutBuffer",
    "compute_gae",
    "ppo_update",
]
