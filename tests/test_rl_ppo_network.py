"""Tests for the PPO actor-critic network."""

from __future__ import annotations

import numpy as np
import torch

from territory_takeover.engine import new_game
from territory_takeover.rl.ppo.network import ActorCritic, PpoNetConfig
from territory_takeover.rl.ppo.spaces import encode_observation, legal_mask_tensor


def _make_batch(
    batch_size: int, board_size: int, num_players: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a zero-filled batch with an all-legal mask. Shapes only."""
    grid = torch.zeros(
        batch_size, 2 * num_players + 2, board_size, board_size, dtype=torch.float32
    )
    scalars = torch.zeros(batch_size, 3 + num_players, dtype=torch.float32)
    mask = torch.ones(batch_size, 4, dtype=torch.bool)
    return grid, scalars, mask


def test_forward_output_shapes() -> None:
    cfg = PpoNetConfig(board_size=8, num_players=2)
    net = ActorCritic(cfg)
    grid, scalars, mask = _make_batch(8, 8, 2)
    dist, value = net(grid, scalars, mask)
    assert dist.logits.shape == (8, 4)
    assert value.shape == (8,)


def test_forward_4p_shapes() -> None:
    cfg = PpoNetConfig(board_size=15, num_players=4)
    net = ActorCritic(cfg)
    grid, scalars, mask = _make_batch(3, 15, 4)
    dist, value = net(grid, scalars, mask)
    assert dist.logits.shape == (3, 4)
    assert value.shape == (3,)


def test_forward_on_real_encoded_observation() -> None:
    """End-to-end: encode a GameState, feed it through the network."""
    state = new_game(board_size=8, num_players=2, seed=0)
    planes, scalars = encode_observation(state, player_id=0)
    mask = legal_mask_tensor(state, player_id=0)

    cfg = PpoNetConfig(board_size=8, num_players=2)
    net = ActorCritic(cfg)

    grid_b = torch.from_numpy(planes).unsqueeze(0)
    scalars_b = torch.from_numpy(scalars).unsqueeze(0)
    mask_b = mask.unsqueeze(0)

    dist, value = net(grid_b, scalars_b, mask_b)
    action = dist.sample()
    assert action.shape == (1,)
    assert 0 <= action.item() <= 3
    assert value.shape == (1,)


def test_sampled_actions_never_illegal() -> None:
    """Across 1000 samples per mask, masked actions must never be drawn."""
    torch.manual_seed(0)
    cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(cfg)
    grid, scalars, _ = _make_batch(1, 6, 2)

    mask_patterns = [
        torch.tensor([[True, False, False, False]]),
        torch.tensor([[False, True, False, False]]),
        torch.tensor([[True, True, False, False]]),
        torch.tensor([[False, False, True, True]]),
        torch.tensor([[True, False, True, False]]),
    ]
    for i, mask in enumerate(mask_patterns):
        dist, _ = net(grid, scalars, mask)
        samples = dist.sample(sample_shape=(1000,)).squeeze(-1)
        legal_idxs = torch.nonzero(mask[0]).squeeze(-1).tolist()
        for a in samples.tolist():
            assert a in legal_idxs, f"pattern={i} illegal sample={a} mask={mask}"


def test_log_prob_of_illegal_action_is_very_negative() -> None:
    """log_prob on an illegal action should be ~LOGIT_MASK_VALUE after softmax."""
    torch.manual_seed(0)
    cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(cfg)
    grid, scalars, _ = _make_batch(1, 6, 2)
    mask = torch.tensor([[True, False, True, False]])
    dist, _ = net(grid, scalars, mask)
    logp_illegal = dist.log_prob(torch.tensor([1]))
    # Illegal log-prob should be very negative (logit is LOGIT_MASK_VALUE).
    assert logp_illegal.item() < -1e6


def test_gradients_flow_through_trunk() -> None:
    """backward() from a legal log-prob fills gradients on trunk parameters."""
    torch.manual_seed(0)
    cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(cfg)
    grid, scalars, mask = _make_batch(4, 6, 2)
    mask = torch.tensor([[True, True, True, True]] * 4)
    dist, value = net(grid, scalars, mask)
    action = torch.zeros(4, dtype=torch.long)
    loss = -(dist.log_prob(action).mean() + value.mean())
    loss.backward()

    # Every parameter in conv trunk and MLP must have a non-None gradient.
    for name, p in net.named_parameters():
        assert p.grad is not None, f"param {name} has no grad"
        assert torch.isfinite(p.grad).all(), f"param {name} grad not finite"


def test_forward_is_deterministic_under_seed() -> None:
    torch.manual_seed(42)
    cfg = PpoNetConfig(board_size=6, num_players=2)
    net1 = ActorCritic(cfg)

    torch.manual_seed(42)
    net2 = ActorCritic(cfg)

    grid, scalars, mask = _make_batch(2, 6, 2)
    with torch.no_grad():
        d1, v1 = net1(grid, scalars, mask)
        d2, v2 = net2(grid, scalars, mask)

    torch.testing.assert_close(d1.logits, d2.logits)
    torch.testing.assert_close(v1, v2)


def test_trunk_requires_batch_dim() -> None:
    cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(cfg)
    unbatched_grid = torch.zeros(cfg.grid_in_channels, 6, 6)
    unbatched_scalars = torch.zeros(cfg.scalar_feature_dim)
    try:
        net.trunk(unbatched_grid, unbatched_scalars)
    except ValueError as exc:
        assert "4D" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-4D grid")


def test_param_count_bounded_for_small_config() -> None:
    """Guard against accidentally ballooning the network."""
    cfg = PpoNetConfig(board_size=8, num_players=2)
    net = ActorCritic(cfg)
    total = sum(p.numel() for p in net.parameters())
    # 4 conv layers of 64 channels -> ~170k; MLP dominates the rest.
    # Keep under 500k so Phase 3c's ResNet upgrade remains meaningful.
    assert total < 500_000, f"{total} params is too large for Phase 3b"


def test_large_board_forward_shapes() -> None:
    """20x20 / 4p matches the scale target; forward must still work."""
    cfg = PpoNetConfig(board_size=20, num_players=4)
    net = ActorCritic(cfg)
    grid, scalars, mask = _make_batch(2, 20, 4)
    dist, value = net(grid, scalars, mask)
    assert dist.logits.shape == (2, 4)
    assert value.shape == (2,)


def test_sample_consistent_with_numpy_seed() -> None:
    """Determinism smoke test at the distribution level."""
    torch.manual_seed(7)
    cfg = PpoNetConfig(board_size=6, num_players=2)
    net = ActorCritic(cfg)
    grid, scalars, mask = _make_batch(1, 6, 2)
    dist, _ = net(grid, scalars, mask)
    torch.manual_seed(7)
    s1 = dist.sample(sample_shape=(10,))
    torch.manual_seed(7)
    s2 = dist.sample(sample_shape=(10,))
    # Note: dist is reused so its logits are fixed; re-seeding torch makes the
    # sample draw deterministic.
    assert torch.equal(s1, s2)
    # Sanity: samples must fall in [0, 3].
    assert np.all((s1.numpy() >= 0) & (s1.numpy() <= 3))
