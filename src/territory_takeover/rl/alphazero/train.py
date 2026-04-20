"""AlphaZero training primitives: loss, single SGD step, training loop.

``train_step`` is a pure function of ``(net, optimizer, batch)`` that
computes the masked cross-entropy policy loss, MSE value loss, and L2
regularization, runs one backward pass + optimizer step, and returns the
per-loss scalars for logging. It is trivially unit-testable: run a few
steps on a hand-built batch and assert total loss decreases.

``train_loop`` is the orchestrator: for ``N`` iterations it generates
self-play games, appends to the replay buffer, runs ``train_steps`` SGD
updates sampled from the buffer, and periodically snapshots the network
to disk. The gating tournament that would decide whether a new snapshot
replaces the self-play champion is left as a stub — under the reduced
Phase 3c scope the latest snapshot always drives self-play. A TODO
marker is left in-code so the follow-up is easy to find.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from territory_takeover.rl.alphazero.evaluator import NNEvaluator
from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig
from territory_takeover.rl.alphazero.replay import ReplayBuffer
from territory_takeover.rl.alphazero.selfplay import SelfPlayConfig, play_game
from territory_takeover.rl.alphazero.spaces import grid_channel_count, scalar_feature_dim
from territory_takeover.rl.ppo.spaces import LOGIT_MASK_VALUE

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class TrainConfig:
    num_iterations: int = 50
    games_per_iteration: int = 10
    train_steps_per_iteration: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    l2_weight: float = 1e-4
    value_loss_coef: float = 1.0
    buffer_capacity: int = 50_000
    snapshot_every: int = 10


@dataclass(slots=True)
class TrainMetrics:
    iterations: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    l2_losses: list[float] = field(default_factory=list)
    total_losses: list[float] = field(default_factory=list)
    buffer_sizes: list[int] = field(default_factory=list)
    self_play_half_moves: list[float] = field(default_factory=list)


def _masked_softmax_cross_entropy(
    logits: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy against ``targets`` (visit *distribution*) with illegal
    logits set to :data:`LOGIT_MASK_VALUE` beforehand.

    ``targets`` is expected to be non-negative and to sum to 1 along the
    last axis on legal positions (zero on illegal). Returns a scalar loss
    averaged over the batch.
    """
    masked_logits = torch.where(
        mask, logits, torch.full_like(logits, LOGIT_MASK_VALUE)
    )
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    # -sum(target * log_prob). Illegal positions have target=0 so they
    # zero out of the sum — the log_prob there is ~LOGIT_MASK_VALUE but
    # multiplied by 0 is 0.
    per_sample = -(targets * log_probs).sum(dim=-1)
    return per_sample.mean()


def _l2_regularization(net: nn.Module) -> torch.Tensor:
    total = torch.zeros(1, device=next(net.parameters()).device)
    for p in net.parameters():
        total = total + p.pow(2).sum()
    return total.squeeze(0)


def train_step(
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    value_loss_coef: float = 1.0,
    l2_weight: float = 1e-4,
) -> dict[str, float]:
    """One SGD step; returns per-loss scalar floats."""
    grids, scalars, masks, visits, final_scores = batch
    visit_totals = visits.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    targets = visits / visit_totals

    net.train()
    optimizer.zero_grad()
    logits, values = net(grids, scalars, masks)

    policy_loss = _masked_softmax_cross_entropy(logits, masks, targets)
    # Scalar-head ablation path uses the mean of final_scores as a stand-in
    # target; the per-seat head gets the full vector. The ablation is
    # deliberately weak and flagged in the Phase 3c writeup.
    value_target = (
        final_scores.mean(dim=-1, keepdim=True)
        if values.shape[1] == 1
        else final_scores
    )
    value_loss = nn.functional.mse_loss(values, value_target)
    l2 = _l2_regularization(net)
    total = policy_loss + value_loss_coef * value_loss + l2_weight * l2

    total.backward()
    optimizer.step()
    net.eval()

    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "l2_loss": float(l2.detach().cpu().item()),
        "total_loss": float(total.detach().cpu().item()),
    }


def _build_buffer(net_cfg: AZNetConfig, train_cfg: TrainConfig) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=train_cfg.buffer_capacity,
        grid_shape=(
            grid_channel_count(net_cfg.num_players),
            net_cfg.board_size,
            net_cfg.board_size,
        ),
        scalar_dim=scalar_feature_dim(net_cfg.num_players),
        num_players=net_cfg.num_players,
    )


def train_loop(
    net_cfg: AZNetConfig,
    train_cfg: TrainConfig,
    self_play_cfg: SelfPlayConfig,
    *,
    out_dir: Path,
    seed: int = 0,
    device: str = "cpu",
) -> TrainMetrics:
    """Run the AlphaZero training loop and persist artifacts under ``out_dir``.

    Side effects:
    - ``out_dir`` is created.
    - ``net_final.pt`` plus ``net_{iter}.pt`` snapshots are written.
    - ``iteration_log.csv`` holds one row per iteration.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    net = AlphaZeroNet(net_cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=train_cfg.learning_rate)
    buffer = _build_buffer(net_cfg, train_cfg)
    evaluator = NNEvaluator(net, device=device)

    metrics = TrainMetrics()
    log_path = out_dir / "iteration_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "policy_loss",
                "value_loss",
                "l2_loss",
                "total_loss",
                "buffer_size",
                "avg_half_moves",
            ]
        )

        for iteration in range(train_cfg.num_iterations):
            total_half_moves = 0
            for _ in range(train_cfg.games_per_iteration):
                evaluator.reset()
                samples = play_game(evaluator, self_play_cfg, rng=rng)
                buffer.extend(samples)
                total_half_moves += len(samples)
            avg_half_moves = (
                total_half_moves / max(train_cfg.games_per_iteration, 1)
            )

            # TODO: gating tournament — compare latest snapshot to the
            # current champion before promoting. Out of scope for the
            # reduced Phase 3c cadence.

            losses = {"policy_loss": 0.0, "value_loss": 0.0, "l2_loss": 0.0, "total_loss": 0.0}
            for _ in range(train_cfg.train_steps_per_iteration):
                g, s, m, v, fs = buffer.sample(train_cfg.batch_size, rng)
                batch = (
                    torch.from_numpy(g).to(device),
                    torch.from_numpy(s).to(device),
                    torch.from_numpy(m).to(device),
                    torch.from_numpy(v).to(device),
                    torch.from_numpy(fs).to(device),
                )
                step_losses = train_step(
                    net,
                    optimizer,
                    batch,
                    value_loss_coef=train_cfg.value_loss_coef,
                    l2_weight=train_cfg.l2_weight,
                )
                for k, v_ in step_losses.items():
                    losses[k] += v_
            for k in losses:
                losses[k] /= max(train_cfg.train_steps_per_iteration, 1)

            writer.writerow(
                [
                    iteration,
                    f"{losses['policy_loss']:.6f}",
                    f"{losses['value_loss']:.6f}",
                    f"{losses['l2_loss']:.6f}",
                    f"{losses['total_loss']:.6f}",
                    len(buffer),
                    f"{avg_half_moves:.2f}",
                ]
            )
            f.flush()

            metrics.iterations.append(iteration)
            metrics.policy_losses.append(losses["policy_loss"])
            metrics.value_losses.append(losses["value_loss"])
            metrics.l2_losses.append(losses["l2_loss"])
            metrics.total_losses.append(losses["total_loss"])
            metrics.buffer_sizes.append(len(buffer))
            metrics.self_play_half_moves.append(avg_half_moves)

            if (iteration + 1) % train_cfg.snapshot_every == 0:
                torch.save(net.state_dict(), out_dir / f"net_{iteration + 1}.pt")

    torch.save(net.state_dict(), out_dir / "net_final.pt")
    return metrics


__all__ = [
    "TrainConfig",
    "TrainMetrics",
    "train_loop",
    "train_step",
]
