"""Tests for MaxNAgent, ParanoidAgent, and the underlying search functions."""

from __future__ import annotations

import time

import numpy as np

from territory_takeover.constants import PATH_CODES
from territory_takeover.engine import new_game
from territory_takeover.eval.heuristic import LinearEvaluator
from territory_takeover.search import (
    MaxNAgent,
    ParanoidAgent,
    UniformRandomAgent,
    maxn_search,
    paranoid_search,
    tournament,
)
from territory_takeover.search.agent import Agent
from territory_takeover.state import GameState


def _search_evaluator() -> LinearEvaluator:
    # Same compact weight set as the greedy-agent tournament test: these three
    # features are informative at shallow depth and keep move ordering stable.
    return LinearEvaluator(
        {
            "territory_total": 1.0,
            "mobility": 0.3,
            "enclosure_potential_area": 0.5,
        }
    )


def _count_nodes_maxn(
    state: GameState, root_player: int, depth: int, evaluator: LinearEvaluator
) -> int:
    counter: list[int] = [0]
    maxn_search(state, root_player, depth, evaluator, _node_counter=counter)
    return counter[0]


def _count_nodes_paranoid(
    state: GameState, root_player: int, depth: int, evaluator: LinearEvaluator
) -> int:
    counter: list[int] = [0]
    paranoid_search(state, root_player, depth, evaluator, _node_counter=counter)
    return counter[0]


def _build_forced_loss_state() -> tuple[GameState, int, int]:
    """Return (state, survive_action, die_action) for a hand-built position.

    6x6 board, 2 players. Player 0 is walled so only S (=1) and E (=3) are
    legal. Playing E marches into a cell whose every neighbor is non-empty,
    so on player 0's next turn ``legal_actions`` is empty and
    ``_advance_turn`` kills the player, ending the game. Playing S stays in
    open territory.
    """
    state = new_game(board_size=6, num_players=2, spawn_positions=[(2, 2), (5, 5)])
    opp = PATH_CODES[1]
    # Walls that turn N and W illegal for player 0.
    state.grid[1, 2] = opp
    state.grid[2, 1] = opp
    # Walls that turn (2, 3) into a dead-end once player 0 steps there.
    state.grid[1, 3] = opp
    state.grid[3, 3] = opp
    state.grid[2, 4] = opp
    return state, 1, 3  # S survives, E is the forced-loss move.


def test_maxn_search_returns_shape_num_players() -> None:
    ev = _search_evaluator()
    state = new_game(board_size=8, num_players=2, seed=0)
    action, vec = maxn_search(state, state.current_player, 2, ev)
    assert 0 <= action < 4, f"action {action} out of range"
    assert vec.shape == (2,), f"expected shape (2,), got {vec.shape}"
    assert vec.dtype == np.float64, f"expected float64, got {vec.dtype}"


def test_paranoid_search_returns_scalar() -> None:
    ev = _search_evaluator()
    state = new_game(board_size=8, num_players=2, seed=0)
    action, value = paranoid_search(state, state.current_player, 2, ev)
    assert 0 <= action < 4, f"action {action} out of range"
    assert isinstance(value, float), f"expected float, got {type(value)}"


def test_maxn_agent_depth3_10x10_under_500ms() -> None:
    ev = _search_evaluator()
    for i in range(3):
        state = new_game(board_size=10, num_players=2, seed=100 + i)
        agent = MaxNAgent(depth=3, evaluator=ev)
        t0 = time.perf_counter()
        action = agent.select_action(state, state.current_player)
        dt = time.perf_counter() - t0
        assert 0 <= action < 4, f"trial={i} action out of range"
        assert dt < 0.5, f"trial={i} maxn depth=3 took {dt:.3f}s (>=0.5s)"


def test_paranoid_agent_depth3_10x10_under_500ms() -> None:
    ev = _search_evaluator()
    for i in range(3):
        state = new_game(board_size=10, num_players=2, seed=100 + i)
        agent = ParanoidAgent(depth=3, evaluator=ev)
        t0 = time.perf_counter()
        action = agent.select_action(state, state.current_player)
        dt = time.perf_counter() - t0
        assert 0 <= action < 4, f"trial={i} action out of range"
        assert dt < 0.5, f"trial={i} paranoid depth=3 took {dt:.3f}s (>=0.5s)"


def test_paranoid_explores_fewer_nodes_than_maxn() -> None:
    ev = _search_evaluator()
    for i in range(5):
        state = new_game(board_size=10, num_players=2, seed=200 + i)
        root_player = state.current_player
        maxn_nodes = _count_nodes_maxn(state, root_player, 4, ev)
        paranoid_nodes = _count_nodes_paranoid(state, root_player, 4, ev)
        assert paranoid_nodes < maxn_nodes, (
            f"trial={i} paranoid={paranoid_nodes} not < maxn={maxn_nodes}"
        )


def test_maxn_agent_beats_random_on_15x15() -> None:
    ev = _search_evaluator()
    agent = MaxNAgent(depth=2, evaluator=ev)
    rand = UniformRandomAgent(rng=np.random.default_rng(17))
    results = tournament(agent, rand, num_games=50, board_size=15, seed=17)
    win_rate = results["wins_a"] / 50
    assert win_rate > 0.8, f"maxn win rate {win_rate:.2f} not > 0.8 (results={results})"


def test_paranoid_agent_beats_random_on_15x15() -> None:
    ev = _search_evaluator()
    agent = ParanoidAgent(depth=3, evaluator=ev)
    rand = UniformRandomAgent(rng=np.random.default_rng(43))
    results = tournament(agent, rand, num_games=50, board_size=15, seed=13)
    win_rate = results["wins_a"] / 50
    assert win_rate > 0.8, (
        f"paranoid win rate {win_rate:.2f} not > 0.8 (results={results})"
    )


def test_maxn_agent_picks_survive_move_in_forced_loss() -> None:
    ev = _search_evaluator()
    state, survive_action, die_action = _build_forced_loss_state()
    agent = MaxNAgent(depth=2, evaluator=ev)
    picked = agent.select_action(state, state.current_player)
    assert picked == survive_action, (
        f"maxn picked {picked}, expected survive={survive_action} (die={die_action})"
    )


def test_paranoid_agent_picks_survive_move_in_forced_loss() -> None:
    ev = _search_evaluator()
    state, survive_action, die_action = _build_forced_loss_state()
    agent = ParanoidAgent(depth=2, evaluator=ev)
    picked = agent.select_action(state, state.current_player)
    assert picked == survive_action, (
        f"paranoid picked {picked}, expected survive={survive_action} (die={die_action})"
    )


def test_maxn_agent_respects_time_budget() -> None:
    ev = _search_evaluator()
    for i in range(3):
        state = new_game(board_size=10, num_players=2, seed=300 + i)
        agent = MaxNAgent(depth=10, evaluator=ev)
        t0 = time.perf_counter()
        action = agent.select_action(
            state, state.current_player, time_budget_s=0.1
        )
        dt = time.perf_counter() - t0
        assert 0 <= action < 4, f"trial={i} action out of range"
        assert dt < 0.5, f"trial={i} budget=0.1 but took {dt:.3f}s"
        completed = agent.last_depth_completed or 0
        assert completed >= 2, f"trial={i} no iteration reached depth 2"


def test_paranoid_agent_respects_time_budget() -> None:
    ev = _search_evaluator()
    for i in range(3):
        state = new_game(board_size=10, num_players=2, seed=400 + i)
        agent = ParanoidAgent(depth=10, evaluator=ev)
        t0 = time.perf_counter()
        action = agent.select_action(
            state, state.current_player, time_budget_s=0.1
        )
        dt = time.perf_counter() - t0
        assert 0 <= action < 4, f"trial={i} action out of range"
        assert dt < 0.5, f"trial={i} budget=0.1 but took {dt:.3f}s"
        completed = agent.last_depth_completed or 0
        assert completed >= 2, f"trial={i} no iteration reached depth 2"


def test_maxn_agent_satisfies_protocol() -> None:
    agent: Agent = MaxNAgent(depth=2)
    agent.reset()
    assert agent.name == "maxn"


def test_paranoid_agent_satisfies_protocol() -> None:
    agent: Agent = ParanoidAgent(depth=2)
    agent.reset()
    assert agent.name == "paranoid"
