"""Tests for PPO primitives: rollout buffer, GAE, and update step."""

from __future__ import annotations

import numpy as np
import torch

from territory_takeover.rl.ppo.network import ActorCritic, PpoNetConfig
from territory_takeover.rl.ppo.ppo_core import (
    PpoConfig,
    RolloutBatch,
    RolloutBuffer,
    compute_gae,
    ppo_update,
)


def test_rollout_buffer_shapes_and_fill() -> None:
    buf = RolloutBuffer(
        num_steps=32, num_envs=4, grid_shape=(6, 8, 8), scalar_dim=5
    )
    assert buf.grid.shape == (32, 4, 6, 8, 8)
    assert buf.scalars.shape == (32, 4, 5)
    assert buf.mask.shape == (32, 4, 4)
    assert buf.actions.shape == (32, 4)
    assert buf.logprobs.shape == (32, 4)
    assert buf.rewards.shape == (32, 4)
    assert buf.dones.shape == (32, 4)
    assert buf.values.shape == (32, 4)

    # Fill with zeros; full() must flip at T.
    assert not buf.full()
    for t in range(32):
        buf.add(
            grid=torch.zeros(4, 6, 8, 8),
            scalars=torch.zeros(4, 5),
            mask=torch.ones(4, 4, dtype=torch.bool),
            action=torch.zeros(4, dtype=torch.long),
            logprob=torch.zeros(4),
            reward=torch.ones(4) * t,
            done=torch.zeros(4),
            value=torch.ones(4) * 0.5,
        )
    assert buf.full()

    # Reward at step 5 should be 5.0 across all envs.
    assert (buf.rewards[5] == 5.0).all()

    # reset rewinds without zeroing; a subsequent add overwrites in-place.
    buf.reset()
    assert not buf.full()
    buf.add(
        grid=torch.zeros(4, 6, 8, 8),
        scalars=torch.zeros(4, 5),
        mask=torch.ones(4, 4, dtype=torch.bool),
        action=torch.zeros(4, dtype=torch.long),
        logprob=torch.zeros(4),
        reward=torch.ones(4) * -1.0,
        done=torch.zeros(4),
        value=torch.ones(4) * 0.5,
    )
    assert (buf.rewards[0] == -1.0).all()


def test_rollout_buffer_raises_when_full() -> None:
    buf = RolloutBuffer(
        num_steps=2, num_envs=1, grid_shape=(2, 3, 3), scalar_dim=1
    )
    for _ in range(2):
        buf.add(
            grid=torch.zeros(1, 2, 3, 3),
            scalars=torch.zeros(1, 1),
            mask=torch.ones(1, 4, dtype=torch.bool),
            action=torch.zeros(1, dtype=torch.long),
            logprob=torch.zeros(1),
            reward=torch.zeros(1),
            done=torch.zeros(1),
            value=torch.zeros(1),
        )
    try:
        buf.add(
            grid=torch.zeros(1, 2, 3, 3),
            scalars=torch.zeros(1, 1),
            mask=torch.ones(1, 4, dtype=torch.bool),
            action=torch.zeros(1, dtype=torch.long),
            logprob=torch.zeros(1),
            reward=torch.zeros(1),
            done=torch.zeros(1),
            value=torch.zeros(1),
        )
    except RuntimeError as exc:
        assert "full" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on overflow")


def test_rollout_buffer_iter_minibatches_covers_all_transitions() -> None:
    buf = RolloutBuffer(
        num_steps=8, num_envs=4, grid_shape=(2, 4, 4), scalar_dim=1
    )
    # Fill actions with a unique integer per (t, n) so we can check coverage.
    unique_id = 0
    for _t in range(8):
        actions = torch.tensor(
            [unique_id + i for i in range(4)], dtype=torch.long
        )
        unique_id += 4
        buf.add(
            grid=torch.zeros(4, 2, 4, 4),
            scalars=torch.zeros(4, 1),
            mask=torch.ones(4, 4, dtype=torch.bool),
            action=actions,
            logprob=torch.zeros(4),
            reward=torch.zeros(4),
            done=torch.zeros(4),
            value=torch.zeros(4),
        )
    advantages = torch.zeros(8, 4)
    returns = torch.zeros(8, 4)
    rng = np.random.default_rng(0)
    batches = buf.iter_minibatches(advantages, returns, minibatch_size=8, rng=rng)
    # 32 transitions / 8 = 4 batches.
    assert len(batches) == 4
    collected = torch.cat([b.actions for b in batches])
    assert sorted(collected.tolist()) == list(range(32))


def test_rollout_buffer_iter_minibatches_uneven_split_raises() -> None:
    buf = RolloutBuffer(
        num_steps=3, num_envs=5, grid_shape=(2, 2, 2), scalar_dim=1
    )
    advantages = torch.zeros(3, 5)
    returns = torch.zeros(3, 5)
    rng = np.random.default_rng(0)
    try:
        buf.iter_minibatches(advantages, returns, minibatch_size=4, rng=rng)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError on uneven minibatch split")


def test_compute_gae_hand_crafted_4_step() -> None:
    """Analytic GAE on a 1-env, 4-step sequence.

    Rewards: [1, 1, 1, 1]. Values: [0, 0, 0, 0]. bootstrap=0. done=0 throughout.
    Under gamma=1, lam=1, GAE reduces to Monte Carlo return minus value:
        A[3] = 1 + 0 - 0 = 1
        A[2] = 1 + A[3] = 2
        A[1] = 1 + A[2] = 3
        A[0] = 1 + A[1] = 4
    """
    rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    values = torch.zeros(4, 1)
    dones = torch.zeros(4, 1)
    bootstrap = torch.zeros(1)
    adv, ret = compute_gae(rewards, values, dones, bootstrap, gamma=1.0, lam=1.0)
    torch.testing.assert_close(
        adv, torch.tensor([[4.0], [3.0], [2.0], [1.0]])
    )
    torch.testing.assert_close(ret, adv + values)


def test_compute_gae_respects_terminal_flag() -> None:
    """Terminal done flag should cut bootstrapping from next step."""
    rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    values = torch.zeros(4, 1)
    # Step 1 is terminal -> advantages at step 1 must not include step-2 reward.
    dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    bootstrap = torch.zeros(1)
    adv, _ = compute_gae(rewards, values, dones, bootstrap, gamma=1.0, lam=1.0)
    # A[1] = 1 (terminal, no bootstrap)
    # A[0] = 1 + gamma*lam*(1 - done[0])*A[1] = 1 + 1*1*1*1 = 2
    # A[3] = 1; A[2] = 1 + A[3] = 2
    torch.testing.assert_close(
        adv, torch.tensor([[2.0], [1.0], [2.0], [1.0]])
    )


def test_compute_gae_discounting() -> None:
    """gamma < 1 should apply geometric discount."""
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.zeros(3, 1)
    dones = torch.zeros(3, 1)
    bootstrap = torch.zeros(1)
    adv, _ = compute_gae(rewards, values, dones, bootstrap, gamma=0.5, lam=1.0)
    # A[2] = 1.0; A[1] = 1 + 0.5*1 = 1.5; A[0] = 1 + 0.5*1.5 = 1.75.
    torch.testing.assert_close(
        adv, torch.tensor([[1.75], [1.5], [1.0]])
    )


def test_compute_gae_lambda_zero_is_td0() -> None:
    """lam=0 reduces GAE to the 1-step TD error."""
    rewards = torch.tensor([[2.0], [3.0]])
    values = torch.tensor([[1.0], [0.5]])
    dones = torch.zeros(2, 1)
    bootstrap = torch.tensor([4.0])
    adv, _ = compute_gae(rewards, values, dones, bootstrap, gamma=0.9, lam=0.0)
    # A[0] = r[0] + gamma*V[1] - V[0] = 2 + 0.9*0.5 - 1 = 1.45
    # A[1] = r[1] + gamma*bootstrap - V[1] = 3 + 0.9*4 - 0.5 = 6.1
    torch.testing.assert_close(
        adv, torch.tensor([[1.45], [6.1]])
    )


def test_compute_gae_shape_validation() -> None:
    rewards = torch.zeros(3, 2)
    bad_values = torch.zeros(3, 3)
    dones = torch.zeros(3, 2)
    bootstrap = torch.zeros(2)
    try:
        compute_gae(rewards, bad_values, dones, bootstrap, gamma=0.99, lam=0.95)
    except ValueError as exc:
        assert "values shape" in str(exc)
    else:
        raise AssertionError("expected ValueError on mismatched values shape")

    bad_bootstrap = torch.zeros(3)
    try:
        compute_gae(rewards, torch.zeros(3, 2), dones, bad_bootstrap, gamma=0.99, lam=0.95)
    except ValueError as exc:
        assert "bootstrap_value" in str(exc)
    else:
        raise AssertionError("expected ValueError on bootstrap shape")


def _synthetic_batch(
    net_cfg: PpoNetConfig, batch_size: int, *, positive_advantage: bool
) -> RolloutBatch:
    """Build a deterministic RolloutBatch for PPO update tests.

    Advantages are all positive (or all negative) for a specific action so we
    can predict loss motion: a positive-advantage update should increase the
    log-prob of the chosen action and thus decrease policy loss over epochs.
    """
    c = net_cfg.grid_in_channels
    h = w = net_cfg.board_size
    s = net_cfg.scalar_feature_dim
    grid = torch.zeros(batch_size, c, h, w)
    scalars = torch.zeros(batch_size, s)
    mask = torch.ones(batch_size, 4, dtype=torch.bool)
    actions = torch.zeros(batch_size, dtype=torch.long)
    # Old logprobs at log(1/4) = -log(4) for a uniform policy.
    logprobs = torch.full((batch_size,), -float(np.log(4)))
    adv_sign = 1.0 if positive_advantage else -1.0
    advantages = torch.full((batch_size,), adv_sign)
    returns = torch.ones(batch_size) * 0.5
    values = torch.zeros(batch_size)
    return RolloutBatch(
        grid=grid,
        scalars=scalars,
        mask=mask,
        actions=actions,
        logprobs=logprobs,
        advantages=advantages,
        returns=returns,
        values=values,
    )


def test_ppo_update_returns_diagnostic_dict() -> None:
    torch.manual_seed(0)
    net_cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(net_cfg)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    batch = _synthetic_batch(net_cfg, batch_size=16, positive_advantage=True)
    stats = ppo_update(net, opt, batch, PpoConfig())
    expected_keys = {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
    }
    assert set(stats.keys()) == expected_keys
    for v in stats.values():
        assert np.isfinite(v), stats


def test_ppo_update_reduces_policy_loss_on_positive_advantage() -> None:
    """50 updates on a fixed positive-advantage batch should drive policy_loss down."""
    torch.manual_seed(0)
    net_cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(net_cfg)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    batch = _synthetic_batch(net_cfg, batch_size=32, positive_advantage=True)

    first_stats = ppo_update(net, opt, batch, PpoConfig(normalize_advantages=False))
    for _ in range(49):
        stats = ppo_update(net, opt, batch, PpoConfig(normalize_advantages=False))
    last_stats = stats

    # Policy loss should be strictly lower after optimization; the chosen
    # action's advantage is positive so its log-prob must rise.
    assert last_stats["policy_loss"] < first_stats["policy_loss"], (
        first_stats,
        last_stats,
    )


def test_ppo_update_respects_clip_epsilon() -> None:
    """When ratio exits the [1-eps, 1+eps] band, clip_frac should be > 0."""
    torch.manual_seed(0)
    net_cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(net_cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)  # aggressive LR to trip clip
    batch = _synthetic_batch(net_cfg, batch_size=32, positive_advantage=True)

    saw_clipping = False
    for _ in range(30):
        stats = ppo_update(net, opt, batch, PpoConfig(clip_eps=0.1))
        if stats["clip_frac"] > 0.0:
            saw_clipping = True
            break
    assert saw_clipping, "clip_frac never triggered despite aggressive LR"


def test_ppo_update_grad_norm_clipping_keeps_grads_finite() -> None:
    """Even with a huge advantage, max_grad_norm should keep training stable."""
    torch.manual_seed(0)
    net_cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(net_cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1.0)
    batch = _synthetic_batch(net_cfg, batch_size=8, positive_advantage=True)
    # Inflate the advantage.
    batch.advantages.mul_(1e4)

    stats = ppo_update(net, opt, batch, PpoConfig(max_grad_norm=0.5))
    assert np.isfinite(stats["loss"])
    for p in net.parameters():
        assert torch.isfinite(p.data).all(), "parameters exploded despite clipping"


def test_ppo_update_preserves_legal_mask_during_loss() -> None:
    """Illegal actions in the mask must produce very negative new logprobs."""
    torch.manual_seed(0)
    net_cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(net_cfg)
    # Force the batch action to be legal (action 0) and mask-out others.
    c = net_cfg.grid_in_channels
    h = w = net_cfg.board_size
    s = net_cfg.scalar_feature_dim
    batch = RolloutBatch(
        grid=torch.zeros(4, c, h, w),
        scalars=torch.zeros(4, s),
        mask=torch.tensor([[True, False, False, False]] * 4),
        actions=torch.zeros(4, dtype=torch.long),
        logprobs=torch.zeros(4),  # log(1) under a hard-masked policy
        advantages=torch.ones(4),
        returns=torch.zeros(4),
        values=torch.zeros(4),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    stats = ppo_update(net, opt, batch, PpoConfig())
    # Entropy of a single-legal-action distribution is 0; small positive from
    # numerical error tolerated.
    assert stats["entropy"] < 1e-3
