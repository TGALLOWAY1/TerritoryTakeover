"""Tests for the Phase 3c NN evaluator (cache + batch + virtual loss)."""

from __future__ import annotations

import numpy as np
import torch

from territory_takeover.actions import legal_action_mask
from territory_takeover.engine import new_game
from territory_takeover.rl.alphazero.evaluator import NNEvaluator, state_hash
from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig


def _tiny_net(num_players: int = 2, board_size: int = 6) -> AlphaZeroNet:
    torch.manual_seed(0)
    cfg = AZNetConfig(
        board_size=board_size,
        num_players=num_players,
        num_res_blocks=1,
        channels=8,
        value_hidden=8,
    )
    return AlphaZeroNet(cfg)


def test_evaluate_shapes() -> None:
    net = _tiny_net()
    ev = NNEvaluator(net, batch_size=4)
    state = new_game(board_size=6, num_players=2, spawn_positions=[(0, 0), (5, 5)])
    mask = legal_action_mask(state, state.current_player)
    prior, value = ev.evaluate(state, state.current_player, mask)

    assert prior.shape == (4,)
    assert value.shape == (2,)
    assert prior.dtype == np.float32
    assert value.dtype == np.float32


def test_priors_zero_on_illegal_and_sum_to_one() -> None:
    net = _tiny_net()
    ev = NNEvaluator(net)
    state = new_game(board_size=6, num_players=2, spawn_positions=[(0, 0), (5, 5)])
    mask = np.array([True, False, False, True], dtype=np.bool_)
    prior, _ = ev.evaluate(state, state.current_player, mask)

    assert prior[1] == 0.0
    assert prior[2] == 0.0
    assert prior[0] > 0.0
    assert prior[3] > 0.0
    np.testing.assert_allclose(prior.sum(), 1.0, atol=1e-6)


def test_cache_hit_returns_identical_values() -> None:
    net = _tiny_net()
    ev = NNEvaluator(net)
    state = new_game(board_size=6, num_players=2, spawn_positions=[(0, 0), (5, 5)])
    mask = legal_action_mask(state, state.current_player)

    prior1, value1 = ev.evaluate(state, state.current_player, mask)
    prior2, value2 = ev.evaluate(state, state.current_player, mask)

    np.testing.assert_array_equal(prior1, prior2)
    np.testing.assert_array_equal(value1, value2)


def test_cache_evicts_lru() -> None:
    net = _tiny_net()
    ev = NNEvaluator(net, cache_size=2)

    # Three distinct (state, player) keys pushed in order.
    s1 = new_game(board_size=6, num_players=2, spawn_positions=[(0, 0), (5, 5)])
    s2 = new_game(board_size=6, num_players=2, spawn_positions=[(0, 5), (5, 0)])
    s3 = new_game(board_size=6, num_players=2, spawn_positions=[(0, 1), (5, 4)])
    mask = np.array([True, True, True, True], dtype=np.bool_)

    ev.evaluate(s1, 0, mask)
    ev.evaluate(s2, 0, mask)
    ev.evaluate(s3, 0, mask)

    assert len(ev._cache) == 2
    assert state_hash(s1, 0) not in ev._cache
    assert state_hash(s2, 0) in ev._cache
    assert state_hash(s3, 0) in ev._cache


def test_batched_matches_single_eval() -> None:
    net = _tiny_net()
    ev_single = NNEvaluator(net, batch_size=1)
    ev_batched = NNEvaluator(net, batch_size=16)

    states = [
        new_game(board_size=6, num_players=2, spawn_positions=[(0, 0), (5, 5)]),
        new_game(board_size=6, num_players=2, spawn_positions=[(0, 5), (5, 0)]),
        new_game(board_size=6, num_players=2, spawn_positions=[(1, 1), (4, 4)]),
    ]
    masks = [legal_action_mask(s, s.current_player) for s in states]

    singles = [
        ev_single.evaluate(s, s.current_player, m) for s, m in zip(states, masks, strict=True)
    ]
    batched = ev_batched.evaluate_batch(
        [(s, s.current_player, m) for s, m in zip(states, masks, strict=True)]
    )

    for (p_s, v_s), (p_b, v_b) in zip(singles, batched, strict=True):
        np.testing.assert_allclose(p_s, p_b, atol=1e-6)
        np.testing.assert_allclose(v_s, v_b, atol=1e-6)


def test_virtual_loss_roundtrip() -> None:
    net = _tiny_net()
    ev = NNEvaluator(net)
    state = new_game(board_size=6, num_players=2, spawn_positions=[(0, 0), (5, 5)])
    key = state_hash(state, 0)

    assert not ev.has_virtual_loss(key)
    ev.apply_virtual_loss(key)
    ev.apply_virtual_loss(key)
    assert ev.has_virtual_loss(key)

    ev.revert_virtual_loss(key)
    assert ev.has_virtual_loss(key)
    ev.revert_virtual_loss(key)
    assert not ev.has_virtual_loss(key)


def test_state_hash_distinguishes_active_player() -> None:
    state = new_game(board_size=6, num_players=2, spawn_positions=[(0, 0), (5, 5)])
    assert state_hash(state, 0) != state_hash(state, 1)


def test_reset_clears_cache_and_pending() -> None:
    net = _tiny_net()
    ev = NNEvaluator(net)
    state = new_game(board_size=6, num_players=2, spawn_positions=[(0, 0), (5, 5)])
    mask = legal_action_mask(state, state.current_player)
    ev.evaluate(state, state.current_player, mask)
    ev.apply_virtual_loss(state_hash(state, 0))

    ev.reset()
    assert len(ev._cache) == 0
    assert len(ev._pending) == 0
