"""Tests for the Gymnasium and PettingZoo-AEC-style wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from territory_takeover.gym_env import (  # noqa: E402
    MultiAgentEnv,
    TerritoryTakeoverEnv,
)


def test_check_env_passes() -> None:
    env = TerritoryTakeoverEnv(board_size=8, num_players=2)
    check_env(env, skip_render_check=True)


def test_100_random_episodes_no_exceptions() -> None:
    env = TerritoryTakeoverEnv(board_size=6, num_players=2, max_steps=200)
    rng = np.random.default_rng(0)
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            mask = obs["action_mask"]
            legal = np.flatnonzero(mask)
            action = (
                int(rng.choice(legal)) if legal.size else 0
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            assert isinstance(reward, float)
            done = terminated or truncated


def test_observation_shapes_and_dtypes_stable() -> None:
    board_size = 6
    num_players = 2
    env = TerritoryTakeoverEnv(
        board_size=board_size, num_players=num_players, max_steps=100
    )
    obs, _ = env.reset(seed=42)

    def _check(o: dict[str, Any]) -> None:
        assert o["grid"].shape == (board_size, board_size)
        assert o["grid"].dtype == np.int8
        assert o["heads"].shape == (num_players, 2)
        assert o["heads"].dtype == np.int32
        assert o["action_mask"].shape == (4,)
        assert o["action_mask"].dtype == np.int8
        assert isinstance(o["current_player"], int)
        assert 0 <= o["current_player"] < num_players
        assert env.observation_space.contains(o)

    _check(obs)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal)) if legal.size else 0
        obs, _, terminated, truncated, _ = env.step(action)
        _check(obs)
        done = terminated or truncated


def test_set_opponent_policy_is_called() -> None:
    calls: list[int] = []

    def always_first_legal(
        obs: dict[str, Any], rng: np.random.Generator
    ) -> int:
        calls.append(int(obs["current_player"]))
        legal = np.flatnonzero(obs["action_mask"])
        return int(legal[0]) if legal.size else 0

    env = TerritoryTakeoverEnv(board_size=6, num_players=2, max_steps=100)
    env.set_opponent_policy(1, always_first_legal)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        action = int(legal[0]) if legal.size else 0
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert calls, "opponent policy was never invoked"
    assert all(pid == 1 for pid in calls)


def test_set_opponent_policy_rejects_agent_player() -> None:
    env = TerritoryTakeoverEnv(board_size=6, num_players=2, agent_player_id=0)
    with pytest.raises(ValueError):
        env.set_opponent_policy(0, lambda o, r: 0)


def test_render_ansi_returns_string() -> None:
    env = TerritoryTakeoverEnv(board_size=5, num_players=2, render_mode="ansi")
    env.reset(seed=0)
    out = env.render()
    assert isinstance(out, str)
    assert "GameState" in out


def test_render_rgb_array_shape_and_dtype() -> None:
    env = TerritoryTakeoverEnv(
        board_size=5, num_players=2, render_mode="rgb_array"
    )
    env.reset(seed=0)
    img = env.render()
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] == img.shape[1]
    assert img.shape[0] % 5 == 0


def test_step_before_reset_raises() -> None:
    env = TerritoryTakeoverEnv(board_size=5, num_players=2)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_step_after_done_raises() -> None:
    env = TerritoryTakeoverEnv(board_size=5, num_players=2, max_steps=5000)
    obs, _ = env.reset(seed=0)
    for _ in range(5000):
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        action = int(legal[0]) if legal.size else 0
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if terminated:
        with pytest.raises(RuntimeError):
            env.step(0)


def test_sparse_reward_scheme_adds_terminal() -> None:
    env = TerritoryTakeoverEnv(
        board_size=5, num_players=2, reward_scheme="sparse", max_steps=5000
    )
    obs, _ = env.reset(seed=1)
    total = 0.0
    terminal_in_info = False
    while True:
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        action = int(legal[0]) if legal.size else 0
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated:
            assert "terminal_rewards" in info
            terminal_in_info = True
            break
        if truncated:
            break
    assert terminal_in_info


def test_invalid_render_mode_raises() -> None:
    with pytest.raises(ValueError):
        TerritoryTakeoverEnv(board_size=5, num_players=2, render_mode="bogus")


def test_invalid_reward_scheme_raises() -> None:
    with pytest.raises(ValueError):
        TerritoryTakeoverEnv(board_size=5, num_players=2, reward_scheme="bogus")


def test_invalid_agent_player_id_raises() -> None:
    with pytest.raises(ValueError):
        TerritoryTakeoverEnv(board_size=5, num_players=2, agent_player_id=5)


# ---------------------------------------------------------- MultiAgentEnv


def test_multi_agent_env_basic_episode() -> None:
    env = MultiAgentEnv(board_size=5, num_players=2)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        obs, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            break
        legal = np.flatnonzero(obs["action_mask"])
        action = int(rng.choice(legal)) if legal.size else 0
        env.step(action)
    assert all(env.terminations[a] for a in env.possible_agents)


def test_multi_agent_env_spaces_per_agent() -> None:
    env = MultiAgentEnv(board_size=6, num_players=4)
    env.reset(seed=0)
    for a in env.possible_agents:
        assert env.observation_space(a).contains(env.observe(a))
        space = env.action_space(a)
        assert isinstance(space, gym.spaces.Discrete)
        assert space.n == 4


def test_multi_agent_env_render_modes() -> None:
    env_ansi = MultiAgentEnv(board_size=5, num_players=2, render_mode="ansi")
    env_ansi.reset(seed=0)
    assert isinstance(env_ansi.render(), str)

    env_rgb = MultiAgentEnv(
        board_size=5, num_players=2, render_mode="rgb_array"
    )
    env_rgb.reset(seed=0)
    img = env_rgb.render()
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
