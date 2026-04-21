"""Tests for the AlphaZero n-step bootstrapped value target.

Two bars to clear:

1. **Pure-function math.** ``compute_nstep_value_targets`` must implement
   the exact n-step return formula the docstring promises. Verified on
   a hand-built 5-step trajectory with hand-computed expected outputs
   — independent of the net, evaluator, or engine.
2. **End-to-end plumbing.** A 2-iteration ``train_loop`` with
   ``value_target_mode='nstep'`` must run without error, populate the
   replay buffer with per-step reward / step-index columns, and
   produce value-head targets in ``[-1, 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from territory_takeover.rl.alphazero.network import AZNetConfig
from territory_takeover.rl.alphazero.selfplay import (
    SelfPlayConfig,
    compute_nstep_value_targets,
)
from territory_takeover.rl.alphazero.train import TrainConfig, train_loop


def test_compute_nstep_value_targets_synthetic_trajectory() -> None:
    """Hand-verified 5-step 2-player trajectory.

    Seats alternate: active_seats = [0, 1, 0, 1, 0].
    Per-step (active-seat) rewards: [0.1, 0.2, 0.0, 0.3, 0.5].
    Bootstrap values (unused here because horizon covers whole traj):
    any values, say np.zeros.
    Terminal scores: [+1.0, -1.0] (seat 0 won).
    n_step = 4, gamma = 0.5 (easy mental math), terminal=True.

    Per-seat reward streams:
    - r[0] = [0.1, 0, 0, 0, 0.5]
    - r[1] = [0, 0.2, 0, 0.3, 0]

    For t=0, K=4 (min(4, 5-0)=4):
        seat 0: 0.5^0·0.1 + 0.5^1·0 + 0.5^2·0 + 0.5^3·0 + 0.5^4·B
              = 0.1 + 0.0625·B
        seat 1: 0 + 0.5·0.2 + 0 + 0.5^3·0.3 + 0.5^4·B
              = 0.1 + 0.0375 + 0.0625·B
              = 0.1375 + 0.0625·B
        t + n = 4 < 5 = T, so B = bootstrap_values[4]. Set bootstrap to
        zeros so we can check the non-bootstrap piece cleanly:
        => seat 0 target = 0.1
        => seat 1 target = 0.1375
    For t=1, K=4 (min(4, 5-1)=4):
        seat 0: 0 + 0.5·0 + 0.25·0 + 0.125·0.5 + 0.0625·B = 0.0625 + ...
        seat 1: 0.2 + 0 + 0.25·0.3 + 0 + 0.0625·B = 0.275 + ...
        t + n = 5 = T, so horizon runs off; terminal=True so
        B = terminal_scores = [+1, -1].
        => seat 0 target = 0.0625 + 0.0625·1.0 = 0.125
        => seat 1 target = 0.275 + 0.0625·(-1.0) = 0.2125
    For t=4, K=1 (min(4, 5-4)=1):
        seat 0: 0.5 + 0.5·B
        seat 1: 0 + 0.5·B
        t + n = 8 > T, terminal=True, B=terminal_scores=[+1, -1].
        => seat 0 = 0.5 + 0.5·1 = 1.0
        => seat 1 = 0 + 0.5·(-1) = -0.5
    """
    active_seats = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    per_step_rewards = np.array([0.1, 0.2, 0.0, 0.3, 0.5], dtype=np.float32)
    bootstrap_values = np.zeros((5, 2), dtype=np.float32)
    terminal_scores = np.array([1.0, -1.0], dtype=np.float32)

    targets = compute_nstep_value_targets(
        active_seats=active_seats,
        per_step_rewards=per_step_rewards,
        bootstrap_values=bootstrap_values,
        terminal_scores=terminal_scores,
        n_step=4,
        gamma=0.5,
        terminal=True,
        num_players=2,
    )

    assert targets.shape == (5, 2)
    # Compare to hand computation with a tight tolerance; floats are exact
    # dyadic fractions at gamma=0.5.
    np.testing.assert_allclose(targets[0], [0.1, 0.1375], rtol=0, atol=1e-6)
    np.testing.assert_allclose(targets[1], [0.125, 0.2125], rtol=0, atol=1e-6)
    np.testing.assert_allclose(targets[4], [1.0, -0.5], rtol=0, atol=1e-6)
    # All targets are in [-1, 1] after clamp.
    assert (targets >= -1.0).all() and (targets <= 1.0).all()


def test_compute_nstep_targets_clamps_above_one() -> None:
    """Large per-step reward + positive bootstrap must be clamped to +1."""
    active_seats = np.array([0, 0, 0], dtype=np.int32)
    # Unrealistic but triggers the clamp cleanly.
    per_step_rewards = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    bootstrap_values = np.ones((3, 1), dtype=np.float32)
    terminal_scores = np.ones((1,), dtype=np.float32)

    targets = compute_nstep_value_targets(
        active_seats=active_seats,
        per_step_rewards=per_step_rewards,
        bootstrap_values=bootstrap_values,
        terminal_scores=terminal_scores,
        n_step=10,
        gamma=1.0,
        terminal=True,
        num_players=1,
    )
    # Uncapped: 0.8+0.8+0.8 + 1·1.0 = 3.4. After clamp: 1.0.
    assert targets[0, 0] == 1.0


def test_compute_nstep_targets_zero_rewards_reduces_to_terminal() -> None:
    """With zero per-step rewards and ``n_step > T``, the target collapses
    to ``gamma^(T-t) * terminal_scores`` at every sample (no per-step
    signal, every horizon runs off the end and hits the terminal
    bootstrap). At ``gamma=1.0`` this recovers exactly the terminal-mode
    target.
    """
    active_seats = np.array([0, 1, 0, 1], dtype=np.int32)
    per_step_rewards = np.zeros(4, dtype=np.float32)
    bootstrap_values = np.full((4, 2), 99.0, dtype=np.float32)  # unused
    terminal_scores = np.array([0.6, -0.6], dtype=np.float32)

    targets = compute_nstep_value_targets(
        active_seats=active_seats,
        per_step_rewards=per_step_rewards,
        bootstrap_values=bootstrap_values,
        terminal_scores=terminal_scores,
        n_step=100,
        gamma=1.0,
        terminal=True,
        num_players=2,
    )
    # Every row equals terminal_scores exactly under gamma=1.
    for t in range(4):
        np.testing.assert_allclose(
            targets[t], terminal_scores, rtol=0, atol=1e-6
        )


def test_train_loop_nstep_mode_runs_end_to_end(tmp_path: Path) -> None:
    """Copy of the terminal-mode smoke with ``value_target_mode='nstep'``."""
    net_cfg = AZNetConfig(
        board_size=6,
        num_players=2,
        num_res_blocks=1,
        channels=8,
        value_hidden=8,
    )
    train_cfg = TrainConfig(
        num_iterations=2,
        games_per_iteration=1,
        train_steps_per_iteration=3,
        batch_size=4,
        learning_rate=1e-3,
        buffer_capacity=128,
        snapshot_every=1,
    )
    self_play_cfg = SelfPlayConfig(
        board_size=6,
        num_players=2,
        puct_iterations=4,
        temperature_moves=4,
        max_half_moves=12,
        value_target_mode="nstep",
        n_step=4,
        gamma=0.99,
    )

    metrics = train_loop(
        net_cfg,
        train_cfg,
        self_play_cfg,
        out_dir=tmp_path,
        seed=0,
        device="cpu",
    )

    assert len(metrics.iterations) == 2
    assert (tmp_path / "net_final.pt").exists()
    assert (tmp_path / "iteration_log.csv").exists()


def test_replay_buffer_records_per_step_reward_and_step_index(
    tmp_path: Path,
) -> None:
    """After a nstep-mode self-play game, buffer columns must be populated."""
    import numpy as np

    from territory_takeover.rl.alphazero.evaluator import NNEvaluator
    from territory_takeover.rl.alphazero.network import AlphaZeroNet
    from territory_takeover.rl.alphazero.replay import ReplayBuffer
    from territory_takeover.rl.alphazero.selfplay import play_game
    from territory_takeover.rl.alphazero.spaces import (
        grid_channel_count,
        scalar_feature_dim,
    )

    net_cfg = AZNetConfig(
        board_size=6,
        num_players=2,
        num_res_blocks=1,
        channels=8,
        value_hidden=8,
    )
    net = AlphaZeroNet(net_cfg)
    evaluator = NNEvaluator(net, device="cpu")
    sp_cfg = SelfPlayConfig(
        board_size=6,
        num_players=2,
        puct_iterations=4,
        temperature_moves=4,
        max_half_moves=12,
        value_target_mode="nstep",
        n_step=4,
        gamma=0.99,
    )
    rng = np.random.default_rng(0)
    samples = play_game(evaluator, sp_cfg, rng=rng)

    assert len(samples) > 0
    # step_index is 0..T-1 in order.
    assert [s.step_index for s in samples] == list(range(len(samples)))
    # Per-step reward is non-negative and bounded by 1 (normalized by area).
    for s in samples:
        assert s.per_step_reward >= 0.0
        assert s.per_step_reward <= 1.0
    # Value targets are clamped to [-1, 1].
    for s in samples:
        assert (s.final_scores >= -1.0).all()
        assert (s.final_scores <= 1.0).all()

    # Round-trip through a save/load.
    buf = ReplayBuffer(
        capacity=128,
        grid_shape=(
            grid_channel_count(net_cfg.num_players),
            net_cfg.board_size,
            net_cfg.board_size,
        ),
        scalar_dim=scalar_feature_dim(net_cfg.num_players),
        num_players=net_cfg.num_players,
    )
    buf.extend(samples)
    save_path = tmp_path / "buf.npz"
    buf.save(save_path)
    restored = ReplayBuffer.load(save_path, capacity=128)
    assert len(restored) == len(samples)
    # The per_step_reward/step_index columns made the round-trip.
    assert (restored._per_step_reward[: len(samples)] >= 0.0).all()
    assert restored._step_index[0] == 0
