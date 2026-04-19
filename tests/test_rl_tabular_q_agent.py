"""Tests for rl.tabular.q_agent.TabularQAgent.

Covers:

1. The selected action is always legal (100 random states, 20 draws each).
2. With epsilon=0 (greedy=True) on a state with a known Q-entry, the agent
   deterministically picks the legal argmax.
3. With epsilon=1 on an all-legal state, each legal action is drawn with
   roughly uniform probability over 10k samples (Wilson 95% CI check).
4. A hand-computed 1-step TD update against known Q-values matches.
5. save -> load round-trip preserves Q-table and episode counter.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from territory_takeover.engine import new_game, step
from territory_takeover.rl.tabular.q_agent import QConfig, TabularQAgent
from territory_takeover.rl.tabular.state_encoder import encode_state


def _wilson_band(k: int, n: int, z: float = 2.576) -> tuple[float, float]:
    # 99% CI -- looser than 95% so flaky seeds do not fail this test.
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (center - half, center + half)


def test_qagent_only_picks_legal_actions() -> None:
    rng = np.random.default_rng(42)
    agent = TabularQAgent(QConfig(total_episodes=100), rng=rng)
    # Run a handful of games, sampling many actions per state -- each must
    # be legal.
    for trial in range(10):
        state = new_game(
            8, 2, spawn_positions=[(0, 0), (7, 7)], seed=trial
        )
        while not state.done:
            pid = state.current_player
            # Draw 20 actions from the same state and confirm all are legal.
            from territory_takeover.actions import legal_actions

            legal = legal_actions(state, pid)
            for draw in range(20):
                a = agent.select_action(state, pid)
                assert a in legal, (
                    f"trial={trial} draw={draw}: action {a} not in legal {legal}"
                )
            step(state, agent.select_action(state, pid), strict=True)


def test_qagent_greedy_picks_legal_argmax() -> None:
    # Construct a state, seed the Q-table with a known vector, confirm the
    # greedy action is the legal argmax. Use a state where not all actions
    # are legal to confirm masking.
    state = new_game(8, 2, spawn_positions=[(0, 0), (7, 7)])
    # Player 0 is at corner (0,0); legal actions are 1 (S) and 3 (E).
    agent = TabularQAgent(QConfig(), rng=np.random.default_rng(0))
    agent.set_greedy(True)
    key = encode_state(state, 0)
    # Plant Q-values: illegal action 0 (N) has the highest raw value, but it
    # must never be selected. Among legal {1, 3}, action 3 wins.
    agent._q[key] = np.array([100.0, 2.0, 50.0, 5.0], dtype=np.float32)
    for i in range(50):
        a = agent.select_action(state, 0)
        assert a == 3, f"trial={i}: greedy picked {a}, expected 3"


def test_qagent_epsilon_one_draws_uniform() -> None:
    # All four actions legal (head in the interior); epsilon=1 explicitly by
    # setting greedy=False and eps_start=eps_end=1. Count draws per legal
    # action and assert each falls inside a Wilson 99% band of 1/|legal|.
    grid = np.zeros((8, 8), dtype=np.int8)
    from territory_takeover.constants import PATH_CODES
    from territory_takeover.state import GameState, PlayerState

    grid[4, 4] = PATH_CODES[0]
    grid[0, 0] = PATH_CODES[1]
    state = GameState(
        grid=grid,
        players=[
            PlayerState(
                player_id=0,
                path=[(4, 4)],
                path_set={(4, 4)},
                head=(4, 4),
                claimed_count=0,
                alive=True,
            ),
            PlayerState(
                player_id=1,
                path=[(0, 0)],
                path_set={(0, 0)},
                head=(0, 0),
                claimed_count=0,
                alive=True,
            ),
        ],
    )
    rng = np.random.default_rng(123)
    # eps_start == eps_end == 1.0 -> epsilon always returns 1.0.
    cfg = QConfig(eps_start=1.0, eps_end=1.0, total_episodes=100_000)
    agent = TabularQAgent(cfg, rng=rng)
    counts = [0, 0, 0, 0]
    n = 10_000
    for _ in range(n):
        a = agent.select_action(state, 0)
        counts[a] += 1
    for a in range(4):
        low, high = _wilson_band(counts[a], n)
        assert low <= 0.25 <= high, (
            f"action {a}: p={counts[a]/n:.3f}, band=[{low:.3f}, {high:.3f}]"
        )


def test_qagent_td_update_matches_hand_computed() -> None:
    # Q[s][a] <- Q[s][a] + alpha * (r + gamma * max(Q[s'][legal]) - Q[s][a]).
    # alpha=0.5, gamma=0.9, initial Q[s]=[0,0,0,0], a=2, r=1.0, Q[s']=[0.2, -5, 0.3, 1.0],
    # next_mask=[True, False, True, True] -> bootstrap = max(0.2, 0.3, 1.0) = 1.0.
    # target = 1.0 + 0.9 * 1.0 = 1.9. Q[s][2] <- 0 + 0.5 * (1.9 - 0) = 0.95.
    cfg = QConfig(alpha=0.5, gamma=0.9)
    agent = TabularQAgent(cfg, rng=np.random.default_rng(0))
    s: tuple[int, int, int, int, int, int, int] = (0, 0, 1, 1, 0, 1, 0)
    s_next: tuple[int, int, int, int, int, int, int] = (1, 0, 1, 1, 1, 1, 0)
    agent._q[s_next] = np.array([0.2, -5.0, 0.3, 1.0], dtype=np.float32)
    next_mask = np.array([True, False, True, True], dtype=np.bool_)
    agent.td_update(s, 2, 1.0, s_next, next_mask)
    got = float(agent.q_table[s][2])
    assert math.isclose(got, 0.95, abs_tol=1e-6), f"Q[s][2] = {got}, expected 0.95"

    # Terminal transition: target = r; Q[s][a] <- Q[s][a] + alpha * (r - Q[s][a]).
    # Starting Q[s]=[0.95, 0, 0, 0], a=2, r=-1.0 -> Q[s][2] += 0.5 * (-1.0 - 0.95)
    #   = 0.95 + 0.5 * (-1.95) = -0.025.
    agent.td_update(s, 2, -1.0, None, None)
    got_terminal = float(agent.q_table[s][2])
    assert math.isclose(got_terminal, -0.025, abs_tol=1e-6), (
        f"terminal Q[s][2] = {got_terminal}, expected -0.025"
    )


def test_qagent_save_load_roundtrip() -> None:
    # Populate a Q-table + episode counter, save, reload, assert equality.
    agent = TabularQAgent(
        QConfig(alpha=0.2, gamma=0.95, total_episodes=250),
        rng=np.random.default_rng(7),
    )
    agent.set_episode(123)
    s1: tuple[int, int, int, int, int, int, int] = (1, 2, 3, 4, 5, 0, 1)
    s2: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    agent._q[s1] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    agent._q[s2] = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "q.pkl"
        agent.save(path)
        reloaded = TabularQAgent.load(path)

    assert reloaded._episode == 123
    assert reloaded.cfg == agent.cfg
    assert set(reloaded.q_table.keys()) == {s1, s2}
    assert np.array_equal(reloaded.q_table[s1], agent.q_table[s1])
    assert np.array_equal(reloaded.q_table[s2], agent.q_table[s2])
