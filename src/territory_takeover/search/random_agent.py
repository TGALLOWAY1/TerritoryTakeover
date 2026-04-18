"""Baseline agents: uniform random and one-ply greedy over :class:`LinearEvaluator`.

Both agents implement :class:`territory_takeover.search.agent.Agent`. They accept
an optional :class:`numpy.random.Generator` so that seeded runs are
reproducible without touching global state, and they treat ``time_budget_s`` /
``max_iterations`` as no-ops (nothing anytime about them).

:class:`HeuristicGreedyAgent` simulates each legal action one ply deep on a
cheap :meth:`GameState.copy`, scores the resulting state for the *acting*
player via :meth:`LinearEvaluator.evaluate_for`, and picks the argmax with
random tie-breaking drawn from the agent's own RNG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.actions import legal_actions
from territory_takeover.engine import step
from territory_takeover.eval.heuristic import LinearEvaluator, default_evaluator

if TYPE_CHECKING:
    from territory_takeover.state import GameState

_TIE_EPS: float = 1e-12


class UniformRandomAgent:
    """Pick uniformly at random among the player's legal actions."""

    name: str

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        name: str = "random",
    ) -> None:
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self.name = name

    def select_action(
        self,
        state: GameState,
        player_id: int,
        time_budget_s: float | None = None,
        max_iterations: int | None = None,
    ) -> int:
        legal = legal_actions(state, player_id)
        if not legal:
            raise ValueError(
                f"UniformRandomAgent called for player {player_id} with no legal actions"
            )
        idx = int(self._rng.integers(len(legal)))
        return legal[idx]

    def reset(self) -> None:
        return None


class HeuristicGreedyAgent:
    """One-ply greedy over :class:`LinearEvaluator` scores for the acting player.

    For each legal action the agent copies ``state``, applies the action with
    :func:`territory_takeover.engine.step` under ``strict=True`` (any
    exception here indicates a caller bug, not an in-game illegal move), then
    scores the successor via :meth:`LinearEvaluator.evaluate_for` for
    ``player_id``. The highest-scoring action wins; ties are broken with the
    agent's RNG.
    """

    name: str

    def __init__(
        self,
        evaluator: LinearEvaluator | None = None,
        rng: np.random.Generator | None = None,
        name: str = "greedy",
    ) -> None:
        self._evaluator: LinearEvaluator = (
            evaluator if evaluator is not None else default_evaluator()
        )
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        self.name = name

    def select_action(
        self,
        state: GameState,
        player_id: int,
        time_budget_s: float | None = None,
        max_iterations: int | None = None,
    ) -> int:
        legal = legal_actions(state, player_id)
        if not legal:
            raise ValueError(
                f"HeuristicGreedyAgent called for player {player_id} with no legal actions"
            )

        scores: list[float] = []
        for action in legal:
            successor = state.copy()
            step(successor, action, strict=True)
            scores.append(self._evaluator.evaluate_for(successor, player_id))

        best = max(scores)
        tied = [i for i, s in enumerate(scores) if best - s <= _TIE_EPS]
        pick = tied[int(self._rng.integers(len(tied)))]
        return legal[pick]

    def reset(self) -> None:
        return None


__all__ = ["HeuristicGreedyAgent", "UniformRandomAgent"]
