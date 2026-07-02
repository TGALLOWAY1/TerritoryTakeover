"""Tests for MaxNAgent, ParanoidAgent, and the underlying search functions."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from territory_takeover.actions import claiming_actions
from territory_takeover.constants import EMPTY, OWNED_CODES
from territory_takeover.engine import new_game, step
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
from territory_takeover.state import GameState, PlayerState


def _search_evaluator() -> LinearEvaluator:
    # Compact weight set that is informative at shallow depth under the
    # corrected rules: territory is the score, claiming mobility rewards
    # staying on the claim frontier, and plain mobility breaks ties.
    return LinearEvaluator(
        {
            "territory_total": 1.0,
            "claiming_mobility": 0.5,
            "mobility": 0.3,
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


def _make_state(
    grid: NDArray[np.int8], heads: list[tuple[int, int]]
) -> GameState:
    players = [
        PlayerState(
            player_id=i,
            head=h,
            territory_count=int(np.count_nonzero(grid == OWNED_CODES[i])),
            alive=True,
        )
        for i, h in enumerate(heads)
    ]
    state = GameState(grid=grid, players=players)
    state.alive_count = len(players)
    state.empty_count = int(np.count_nonzero(grid == EMPTY))
    return state


def _build_last_claim_state() -> tuple[GameState, int, int]:
    """Return (state, claim_action, traverse_action) for a hand-built position.

    4x4 board, 2 players::

        A A e B
        B B B B
        . . . .
        . . . .

    P0 (``A``) is sealed in the top-left pocket with exactly one reachable
    EMPTY cell at (0, 2). From the head (0, 1), E (=3) claims it — after
    which P0 has no reachable EMPTY cell and dies on its next turn, keeping
    territory 3. W (=2) is a traversal move back onto (0, 0) that stalls at
    territory 2 forever. A territory-maximizing search must take the claim:
    under the corrected rules inevitable death is not a catastrophe to be
    dodged at the cost of score.
    """
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[0, 0] = OWNED_CODES[0]
    grid[0, 1] = OWNED_CODES[0]
    grid[0, 3] = OWNED_CODES[1]
    grid[1, :] = OWNED_CODES[1]
    state = _make_state(grid, [(0, 1), (1, 3)])
    return state, 3, 2  # E claims the last cell, W merely traverses.


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


def test_maxn_prefers_claiming_over_traversal() -> None:
    # P0 has claimed one cell, so a traversal move back onto the spawn is
    # legal — but it scores strictly less territory than any claim. Depth-2
    # max-N with a territory-weighted evaluator must pick a claiming move.
    ev = _search_evaluator()
    state = new_game(board_size=6, num_players=2, spawn_positions=[(2, 2), (5, 5)])
    step(state, 3, strict=True)  # P0 east: claims (2, 3)
    step(state, 0, strict=True)  # P1 north: claims (4, 5)
    assert state.current_player == 0

    claims = claiming_actions(state, 0)
    assert 2 not in claims, "W must be the traversal move in this setup"

    agent = MaxNAgent(depth=2, evaluator=ev)
    picked = agent.select_action(state, 0)
    assert picked in claims, f"maxn picked traversal {picked}, expected one of {claims}"


def test_maxn_agent_claims_last_cell_despite_inevitable_death() -> None:
    ev = _search_evaluator()
    state, claim_action, traverse_action = _build_last_claim_state()
    agent = MaxNAgent(depth=2, evaluator=ev)
    picked = agent.select_action(state, state.current_player)
    assert picked == claim_action, (
        f"maxn picked {picked}, expected claim={claim_action} "
        f"(traverse={traverse_action})"
    )


def test_paranoid_agent_claims_last_cell_despite_inevitable_death() -> None:
    ev = _search_evaluator()
    state, claim_action, traverse_action = _build_last_claim_state()
    agent = ParanoidAgent(depth=2, evaluator=ev)
    picked = agent.select_action(state, state.current_player)
    assert picked == claim_action, (
        f"paranoid picked {picked}, expected claim={claim_action} "
        f"(traverse={traverse_action})"
    )


def test_last_claim_state_death_follows_the_claim() -> None:
    # Scenario sanity check: playing the claim really does exhaust P0's
    # reachable EMPTY cells, so P0 is marked dead (territory kept) as soon
    # as the turn cycles back to them.
    state, claim_action, _ = _build_last_claim_state()
    result = step(state, claim_action, strict=True)
    assert result.reward == 1.0
    assert state.players[0].territory_count == 3
    assert state.players[0].alive, "death is applied on turn advance, not instantly"

    step(state, 1, strict=True)  # P1 south: claims (2, 3); turn passes over P0
    assert not state.players[0].alive, "P0 must die once no EMPTY cell is reachable"
    assert state.players[0].territory_count == 3, "death keeps territory"


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
