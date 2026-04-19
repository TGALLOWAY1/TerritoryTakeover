"""Tests for rl.tabular.reward.

Covers each reward path independently:

1. step_reward scales engine_reward linearly by per_cell_gain.
2. death_penalty is zero when path_len_at_death <= 0.
3. death_penalty scales linearly with path_len.
4. terminal_rank_bonus on a 4-player no-tie game returns the (+10, +3, -3,
   -10) ladder in rank order.
5. terminal_rank_bonus on a 2-player tie splits (+10, +3) equally.
6. terminal_rank_bonus on a 4-way tie gives each player the full-ladder
   mean.
"""

from __future__ import annotations

import numpy as np

from territory_takeover.rl.tabular.reward import (
    RewardConfig,
    death_penalty,
    step_reward,
    terminal_rank_bonus,
)
from territory_takeover.state import GameState, PlayerState


def _make_terminal_state(
    scores: list[tuple[int, int]],  # list of (path_len, claimed_count) per player
) -> GameState:
    """Build a terminal-looking state where only the per-player scores matter."""
    grid = np.zeros((4, 4), dtype=np.int8)
    players = []
    for i, (path_len, claimed) in enumerate(scores):
        # Fabricate a path of the requested length; the actual tiles don't
        # matter for the reward calculation (it reads len(path) and
        # claimed_count only).
        path = [(0, j) for j in range(path_len)]
        players.append(
            PlayerState(
                player_id=i,
                path=path,
                path_set=set(path),
                head=path[-1] if path else (-1, -1),
                claimed_count=claimed,
                alive=False,
            )
        )
    return GameState(grid=grid, players=players, done=True)


def test_step_reward_linear_in_per_cell_gain() -> None:
    # step_reward(engine_reward, cfg) must equal per_cell_gain * engine_reward.
    for i in range(5):
        gain = float(i + 1) * 0.5
        cfg = RewardConfig(per_cell_gain=gain)
        for engine_reward in (0.0, 1.0, 3.5, 10.0):
            got = step_reward(engine_reward, cfg)
            expected = gain * engine_reward
            assert got == expected, (
                f"trial={i} engine_reward={engine_reward} gain={gain}: "
                f"{got} != {expected}"
            )


def test_death_penalty_zero_for_nonpositive_path() -> None:
    cfg = RewardConfig()
    for path_len in (-1, 0):
        got = death_penalty(path_len, cfg)
        assert got == 0.0, f"path_len={path_len}: {got} != 0.0"


def test_death_penalty_linear_in_path_len() -> None:
    # death_penalty = -trap_penalty_per_cell * path_len for path_len > 0.
    for i in range(5):
        scale = float(i + 1)
        cfg = RewardConfig(trap_penalty_per_cell=scale)
        for path_len in (1, 5, 20, 50):
            got = death_penalty(path_len, cfg)
            expected = -scale * path_len
            assert got == expected, (
                f"trial={i} path_len={path_len} scale={scale}: {got} != {expected}"
            )


def test_terminal_rank_bonus_4p_no_tie() -> None:
    # Scores descending; rank 0 -> +10, rank 1 -> +3, rank 2 -> -3, rank 3 -> -10.
    cfg = RewardConfig()
    state = _make_terminal_state([(1, 0), (5, 0), (3, 0), (10, 0)])
    # scores = [1, 5, 3, 10] -> ranks: p3=0, p1=1, p2=2, p0=3.
    expected = {3: 10.0, 1: 3.0, 2: -3.0, 0: -10.0}
    for pid, exp in expected.items():
        got = terminal_rank_bonus(state, pid, cfg)
        assert got == exp, f"player {pid}: {got} != {exp}"


def test_terminal_rank_bonus_2p_tie_splits_evenly() -> None:
    # 2-player tie at top: both should get (+10 + +3) / 2 = +6.5.
    cfg = RewardConfig()
    state = _make_terminal_state([(5, 0), (5, 0)])
    for pid in (0, 1):
        got = terminal_rank_bonus(state, pid, cfg)
        assert got == 6.5, f"player {pid}: {got} != 6.5"


def test_terminal_rank_bonus_4_way_tie() -> None:
    # All four players equal: each gets the mean of (+10, +3, -3, -10) = 0.0.
    cfg = RewardConfig()
    state = _make_terminal_state([(5, 0), (5, 0), (5, 0), (5, 0)])
    for pid in range(4):
        got = terminal_rank_bonus(state, pid, cfg)
        assert got == 0.0, f"player {pid}: {got} != 0.0"


def test_terminal_rank_bonus_partial_tie() -> None:
    # Scores: p0=10, p1=5, p2=5, p3=1 -> p0 1st (+10), p1+p2 tied ~(3 + -3)/2=0,
    # p3 last (-10).
    cfg = RewardConfig()
    state = _make_terminal_state([(10, 0), (5, 0), (5, 0), (1, 0)])
    assert terminal_rank_bonus(state, 0, cfg) == 10.0
    assert terminal_rank_bonus(state, 1, cfg) == 0.0
    assert terminal_rank_bonus(state, 2, cfg) == 0.0
    assert terminal_rank_bonus(state, 3, cfg) == -10.0
