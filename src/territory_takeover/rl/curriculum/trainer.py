"""Per-stage curriculum trainer on top of the AlphaZero primitives.

This module owns the stage-transition state machine. Within a stage it
reuses :func:`train_step`, :func:`play_game`, :class:`ReplayBuffer`, and
:class:`NNEvaluator` from :mod:`territory_takeover.rl.alphazero` — the
only new logic is stage setup, weight transfer, periodic evaluation, and
promotion decisions.

**Evaluation signal.** Within a stage we use a cheap "win rate vs
:class:`UniformRandomAgent`" proxy, not a full Elo round-robin. This
avoids embedding Elo computation in the hot path (one Elo recomputation
per stage-eval would dominate wall-clock). The rolling plateau check in
:class:`PromotionState` works on any scalar, so we feed it
``win_rate * 100`` and configure ``elo_gain_threshold`` accordingly in
the YAML. Full BT-MLE Elo is computed once at the end of a run via
``scripts/compute_elo.py``.

**First-enclosure instrumentation.** Self-play detects first enclosure
in-trainer (not in ``selfplay.play_game``) by observing the delta in the
``claimed_count`` accumulator across every game. The first game in which
any player's ``claimed_count > 0`` is recorded as the stage's
first-enclosure step count.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from territory_takeover.engine import new_game, step
from territory_takeover.rl.alphazero.evaluator import NNEvaluator
from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig
from territory_takeover.rl.alphazero.replay import ReplayBuffer
from territory_takeover.rl.alphazero.selfplay import SelfPlayConfig, play_game
from territory_takeover.rl.alphazero.spaces import grid_channel_count, scalar_feature_dim
from territory_takeover.rl.alphazero.train import TrainConfig, train_step
from territory_takeover.rl.curriculum.schedule import (
    PromotionState,
    Schedule,
    Stage,
)
from territory_takeover.rl.curriculum.transfer import transfer_weights
from territory_takeover.search.random_agent import UniformRandomAgent


@dataclass(frozen=True, slots=True)
class CurriculumTrainConfig:
    """Trainer-level config (orthogonal to the :class:`Schedule`)."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    l2_weight: float = 1e-4
    value_loss_coef: float = 1.0
    buffer_capacity: int = 50_000
    eval_games_per_check: int = 16
    iterations_per_eval: int = 1
    """How many outer training iterations (self-play + SGD) between
    evaluations."""


@dataclass(slots=True)
class StageResult:
    name: str
    board_size: int
    num_players: int
    self_play_steps: int
    num_iterations: int
    final_win_rate_vs_random: float
    first_enclosure_step: int | None
    promoted: bool


def _evaluate_vs_random(
    net: AlphaZeroNet,
    stage: Stage,
    num_games: int,
    rng: np.random.Generator,
    device: str,
) -> float:
    """Play ``num_games`` games; AlphaZero takes seat 0, Random fills others.

    Returns the AlphaZero agent's score: wins + 0.5 * ties, divided by games.
    """
    from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent

    az_agent = AlphaZeroAgent(
        net,
        iterations=stage.puct_iterations,
        c_puct=stage.c_puct,
        device=device,
        temperature_eval=0.0,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    random_agent = UniformRandomAgent(rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))))

    az_score = 0.0
    for _ in range(num_games):
        state = new_game(
            board_size=stage.board_size,
            num_players=stage.num_players,
            spawn_positions=stage.spawn_positions,
        )
        az_agent.reset()
        while not state.done:
            player = state.current_player
            agent = az_agent if player == 0 else random_agent
            action = agent.select_action(state, player)
            step(state, action, strict=False)

        final = np.asarray([p.claimed_count for p in state.players], dtype=np.int64)
        az_claim = int(final[0])
        best = int(final.max())
        if az_claim == best:
            ties = int((final == best).sum())
            az_score += 1.0 / ties
        # else: loss, 0 points
    return az_score / max(num_games, 1)


def _build_net(stage: Stage, template: AZNetConfig, device: str) -> AlphaZeroNet:
    """Construct a net for ``stage`` using ``template`` for arch hyperparams."""
    cfg = AZNetConfig(
        board_size=stage.board_size,
        num_players=stage.num_players,
        num_res_blocks=template.num_res_blocks,
        channels=template.channels,
        value_hidden=template.value_hidden,
        scalar_value_head=template.scalar_value_head,
        head_type="conv",  # curriculum mandates variable-size heads
    )
    return AlphaZeroNet(cfg).to(device)


def _build_buffer(stage: Stage, capacity: int) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=capacity,
        grid_shape=(
            grid_channel_count(stage.num_players),
            stage.board_size,
            stage.board_size,
        ),
        scalar_dim=scalar_feature_dim(stage.num_players),
        num_players=stage.num_players,
    )


def _self_play_config(stage: Stage) -> SelfPlayConfig:
    return SelfPlayConfig(
        board_size=stage.board_size,
        num_players=stage.num_players,
        puct_iterations=stage.puct_iterations,
        c_puct=stage.c_puct,
        dirichlet_alpha=stage.dirichlet_alpha,
        dirichlet_eps=stage.dirichlet_eps,
        temperature_moves=stage.temperature_moves,
        max_half_moves=stage.max_half_moves,
    )


def _persist_progress(out_dir: Path, payload: dict[str, Any]) -> None:
    path = out_dir / "curriculum_progress.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True))


@dataclass(slots=True)
class CurriculumRunState:
    """Top-level run state for resumption."""

    current_stage_index: int = 0
    stage_results: list[StageResult] = field(default_factory=list)
    promotion_state: PromotionState | None = None


def train_curriculum(
    schedule: Schedule,
    train_cfg: CurriculumTrainConfig,
    template: AZNetConfig,
    *,
    out_dir: Path,
    seed: int = 0,
    device: str = "cpu",
    log_path: Path | None = None,
) -> list[StageResult]:
    """Drive a curriculum end-to-end and persist artifacts under ``out_dir``.

    Side effects:
    - ``out_dir/stage_{name}/`` per stage with ``net_final.pt``, ``log.csv``.
    - ``out_dir/net_final.pt`` — final stage's checkpoint.
    - ``out_dir/curriculum_progress.yaml`` updated on every promotion.
    - ``out_dir/first_enclosure.csv`` — per-stage first-enclosure step.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    prev_state_dict: dict[str, torch.Tensor] | None = None
    cumulative_steps = 0
    results: list[StageResult] = []

    first_enc_log = out_dir / "first_enclosure.csv"
    with first_enc_log.open("w", newline="") as feh:
        fe_writer = csv.writer(feh)
        fe_writer.writerow(["stage_name", "cumulative_step", "first_enclosure_step"])

        for stage_idx, stage in enumerate(schedule.stages):
            stage_dir = out_dir / f"stage_{stage_idx:02d}_{stage.name}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            net = _build_net(stage, template, device)
            if prev_state_dict is not None:
                report = transfer_weights(prev_state_dict, net)
                (stage_dir / "transfer_report.json").write_text(
                    json.dumps(
                        {
                            "matched": list(report.matched_keys),
                            "missing": list(report.missing_keys),
                            "unexpected": list(report.unexpected_keys),
                            "shape_mismatched": list(report.shape_mismatched_keys),
                        },
                        indent=2,
                    )
                )

            optimizer = torch.optim.Adam(net.parameters(), lr=train_cfg.learning_rate)
            buffer = _build_buffer(stage, train_cfg.buffer_capacity)
            evaluator = NNEvaluator(net, device=device)
            sp_cfg = _self_play_config(stage)
            promotion = PromotionState(criterion=stage.promotion)

            stage_log_path = stage_dir / "log.csv"
            first_enclosure_step: int | None = None

            with stage_log_path.open("w", newline="") as slf:
                stage_writer = csv.writer(slf)
                stage_writer.writerow(
                    [
                        "iteration",
                        "stage_steps",
                        "buffer_size",
                        "policy_loss",
                        "value_loss",
                        "total_loss",
                        "avg_half_moves",
                        "win_rate_vs_random",
                    ]
                )

                iteration = 0
                while not promotion.should_promote():
                    total_half_moves = 0
                    for _ in range(max(1, train_cfg.iterations_per_eval)):
                        for _ in range(stage.games_per_iteration):
                            evaluator.reset()
                            samples = play_game(evaluator, sp_cfg, rng=rng)
                            buffer.extend(samples)
                            total_half_moves += len(samples)
                            promotion.record_steps(len(samples))
                            cumulative_steps += len(samples)

                            if first_enclosure_step is None and any(
                                s.final_scores[p] > _NEUTRAL_SCORE
                                for s in samples
                                for p in range(stage.num_players)
                            ):
                                first_enclosure_step = promotion.stage_steps

                        losses_acc = {
                            "policy_loss": 0.0,
                            "value_loss": 0.0,
                            "l2_loss": 0.0,
                            "total_loss": 0.0,
                        }
                        for _ in range(stage.train_steps_per_iteration):
                            if len(buffer) == 0:
                                break
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
                                losses_acc[k] += v_
                        denom = max(stage.train_steps_per_iteration, 1)
                        for k in losses_acc:
                            losses_acc[k] /= denom

                        iteration += 1

                    win_rate = _evaluate_vs_random(
                        net,
                        stage,
                        num_games=train_cfg.eval_games_per_check,
                        rng=rng,
                        device=device,
                    )
                    promotion.record_evaluation(win_rate * 100.0)

                    stage_writer.writerow(
                        [
                            iteration,
                            promotion.stage_steps,
                            len(buffer),
                            f"{losses_acc['policy_loss']:.6f}",
                            f"{losses_acc['value_loss']:.6f}",
                            f"{losses_acc['total_loss']:.6f}",
                            f"{total_half_moves / max(stage.games_per_iteration, 1):.2f}",
                            f"{win_rate:.4f}",
                        ]
                    )
                    slf.flush()

                    _persist_progress(
                        out_dir,
                        {
                            "seed": seed,
                            "current_stage_index": stage_idx,
                            "stage_name": stage.name,
                            "cumulative_steps": cumulative_steps,
                            "promotion": promotion.to_dict(),
                        },
                    )

            torch.save(net.state_dict(), stage_dir / "net_final.pt")
            result = StageResult(
                name=stage.name,
                board_size=stage.board_size,
                num_players=stage.num_players,
                self_play_steps=promotion.stage_steps,
                num_iterations=iteration,
                final_win_rate_vs_random=(promotion.elo_window[-1] / 100.0)
                if promotion.elo_window
                else 0.0,
                first_enclosure_step=first_enclosure_step,
                promoted=True,
            )
            results.append(result)
            fe_writer.writerow(
                [
                    stage.name,
                    cumulative_steps,
                    first_enclosure_step if first_enclosure_step is not None else -1,
                ]
            )
            feh.flush()

            prev_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if prev_state_dict is not None:
        torch.save(prev_state_dict, out_dir / "net_final.pt")

    (out_dir / "stage_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    return results


# Sentinel: any positive per-seat score in ``final_scores`` (normalized
# to [-1, 1]) means at least one player claimed territory by enclosure.
# ``_terminal_value_normalized`` maps claim share linearly into that
# range, so a strictly-positive entry implies a non-trivial share.
_NEUTRAL_SCORE = -0.9


__all__ = ["CurriculumTrainConfig", "StageResult", "train_curriculum"]
