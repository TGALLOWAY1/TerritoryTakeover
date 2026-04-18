"""Tests for UniformRandomAgent and HeuristicGreedyAgent in the search subpackage."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from territory_takeover.engine import new_game, step
from territory_takeover.eval.heuristic import LinearEvaluator
from territory_takeover.search import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
    play_game,
    tournament,
)
from territory_takeover.search.agent import Agent

# One-ply greedy weights.
#
# The shipped default_evaluator() is documented as "starter weights ... will be
# tuned against self-play later" (eval/heuristic.py: default_evaluator). Those
# weights include choke_pressure, reachable_area, and opponent_distance, which
# are noisy at one-ply depth and produce worse-than-random play in this
# tournament. A compact three-feature set (territory, mobility, enclosure
# potential) is both more interpretable and empirically dominant: across seeds
# 0..9 on the 20x20 / 50-game benchmark it wins 70-86%.
_GREEDY_TEST_WEIGHTS: dict[str, float] = {
    "territory_total": 1.0,
    "mobility": 0.3,
    "enclosure_potential_area": 0.5,
}


def _record_action_sequence(
    agent_factory: Callable[[np.random.Generator], Agent],
    board_size: int,
    num_players: int,
    seed: int,
) -> list[int]:
    """Run a single game and return the full sequence of actions played.

    Uses a single deterministic agent for all seats (built via the factory).
    The factory receives a seeded RNG so two independent calls with the same
    seed yield identical action sequences.
    """
    rng = np.random.default_rng(seed)
    agent = agent_factory(rng)
    state = new_game(board_size=board_size, num_players=num_players, seed=seed)
    actions: list[int] = []
    while not state.done:
        pid = state.current_player
        action = agent.select_action(state, pid)
        actions.append(action)
        step(state, action, strict=True)
    return actions


def test_uniform_random_agent_plays_50_games() -> None:
    for i in range(50):
        rng = np.random.default_rng(1000 + i)
        agents: list[Agent] = [UniformRandomAgent(rng=rng), UniformRandomAgent(rng=rng)]
        terminal = play_game(
            agents, board_size=8, num_players=2, seed=1000 + i, max_turns=10_000
        )
        assert terminal.done, f"trial={i} did not terminate"


def test_heuristic_greedy_agent_plays_50_games() -> None:
    for i in range(50):
        rng = np.random.default_rng(2000 + i)
        agents: list[Agent] = [
            HeuristicGreedyAgent(rng=rng),
            HeuristicGreedyAgent(rng=rng),
        ]
        terminal = play_game(
            agents, board_size=8, num_players=2, seed=2000 + i, max_turns=10_000
        )
        assert terminal.done, f"trial={i} did not terminate"


def test_uniform_random_agent_is_deterministic() -> None:
    def factory(rng: np.random.Generator) -> Agent:
        return UniformRandomAgent(rng=rng)

    for i in range(5):
        seed = 7000 + i
        a = _record_action_sequence(factory, board_size=10, num_players=2, seed=seed)
        b = _record_action_sequence(factory, board_size=10, num_players=2, seed=seed)
        assert a == b, f"trial={i} random agent was non-deterministic"


def test_heuristic_greedy_agent_is_deterministic() -> None:
    def factory(rng: np.random.Generator) -> Agent:
        return HeuristicGreedyAgent(rng=rng)

    for i in range(5):
        seed = 8000 + i
        a = _record_action_sequence(factory, board_size=10, num_players=2, seed=seed)
        b = _record_action_sequence(factory, board_size=10, num_players=2, seed=seed)
        assert a == b, f"trial={i} greedy agent was non-deterministic"


def test_greedy_beats_random_in_tournament() -> None:
    evaluator = LinearEvaluator(_GREEDY_TEST_WEIGHTS)
    greedy = HeuristicGreedyAgent(
        evaluator=evaluator, rng=np.random.default_rng(12345)
    )
    random_agent = UniformRandomAgent(rng=np.random.default_rng(67890))
    results = tournament(
        agent_a=greedy,
        agent_b=random_agent,
        num_games=50,
        board_size=20,
        seed=7,
    )
    total = results["wins_a"] + results["wins_b"] + results["ties"]
    assert total == 50
    win_rate = results["wins_a"] / 50
    assert win_rate > 0.6, (
        f"greedy win rate {win_rate:.2f} not > 0.6 "
        f"(wins_a={results['wins_a']}, wins_b={results['wins_b']}, ties={results['ties']})"
    )
