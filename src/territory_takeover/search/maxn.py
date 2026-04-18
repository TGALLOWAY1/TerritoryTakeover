"""Max-N and paranoid tree-search agents.

:func:`maxn_search` implements the classic Luckhardt & Irani (1986) max-N
algorithm: each node's acting player maximizes **their own** component of
the returned value vector. :func:`paranoid_search` collapses the N-player
game to a 2-player one by assuming all opponents cooperate to minimize the
root player's score, which enables alpha-beta pruning.

:class:`MaxNAgent` and :class:`ParanoidAgent` wrap the two search functions
into the :class:`~territory_takeover.search.agent.Agent` protocol. Both
support iterative deepening when ``select_action`` is called with a
``time_budget_s``.

The search functions accept an optional keyword-only ``_node_counter``
(a single-element ``list[int]`` used as a ref cell) so pruning efficiency
can be compared in tests without changing the public return type.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Final

import numpy as np

from territory_takeover.actions import legal_actions
from territory_takeover.engine import step
from territory_takeover.eval.heuristic import LinearEvaluator, default_evaluator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


_INF: Final[float] = math.inf
_NEG_INF: Final[float] = -math.inf


def maxn_search(
    state: GameState,
    root_player: int,
    depth: int,
    evaluator: LinearEvaluator,
    *,
    _node_counter: list[int] | None = None,
) -> tuple[int, NDArray[np.float64]]:
    """Classic max-N search.

    Each node's ``state.current_player`` picks the child whose value vector
    has the largest entry at their own index. At depth 0 or a terminal
    state, returns ``evaluator.evaluate(state)`` verbatim. The action
    returned at leaves is ``-1`` (sentinel); the action returned at
    internal nodes is the action that the acting player should play.

    ``root_player`` does not affect the recursion itself; it is retained
    for API symmetry with :func:`paranoid_search` and so callers can index
    the returned vector at ``root_player`` to recover the root's score.
    """
    if _node_counter is not None:
        _node_counter[0] += 1

    if state.done or depth == 0:
        return -1, evaluator.evaluate(state)

    acting = state.current_player
    legal = legal_actions(state, acting)
    if not legal:
        # Defensive: engine invariant says a non-terminal state has a
        # current_player with legal moves. Score the position anyway.
        return -1, evaluator.evaluate(state)

    best_action: int = legal[0]
    best_vec: NDArray[np.float64] | None = None

    for action in legal:
        successor = state.copy()
        step(successor, action, strict=True)
        _, child_vec = maxn_search(
            successor,
            root_player,
            depth - 1,
            evaluator,
            _node_counter=_node_counter,
        )
        if best_vec is None or child_vec[acting] > best_vec[acting]:
            best_vec = child_vec
            best_action = action

    assert best_vec is not None  # legal was non-empty
    return best_action, best_vec


def paranoid_search(
    state: GameState,
    root_player: int,
    depth: int,
    evaluator: LinearEvaluator,
    alpha: float = _NEG_INF,
    beta: float = _INF,
    *,
    _node_counter: list[int] | None = None,
) -> tuple[int, float]:
    """Paranoid alpha-beta search.

    Treats ``root_player`` as the sole maximizer and every other player as
    a minimizer of ``root_player``'s score (``evaluator.evaluate_for``).
    Candidate moves are sorted by a shallow 1-ply score before recursing so
    alpha-beta cutoffs fire earlier.
    """
    if _node_counter is not None:
        _node_counter[0] += 1

    if state.done or depth == 0:
        return -1, evaluator.evaluate_for(state, root_player)

    acting = state.current_player
    maximizing = acting == root_player
    legal = legal_actions(state, acting)
    if not legal:
        return -1, evaluator.evaluate_for(state, root_player)

    # 1-ply move ordering: score each successor for root_player and sort.
    ordered: list[tuple[int, GameState, float]] = []
    for action in legal:
        successor = state.copy()
        step(successor, action, strict=True)
        score = evaluator.evaluate_for(successor, root_player)
        ordered.append((action, successor, score))
    ordered.sort(key=lambda t: t[2], reverse=maximizing)

    best_action: int = ordered[0][0]
    if maximizing:
        value = _NEG_INF
        for action, successor, _score in ordered:
            _, child_val = paranoid_search(
                successor,
                root_player,
                depth - 1,
                evaluator,
                alpha,
                beta,
                _node_counter=_node_counter,
            )
            if child_val > value:
                value = child_val
                best_action = action
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return best_action, value

    value = _INF
    for action, successor, _score in ordered:
        _, child_val = paranoid_search(
            successor,
            root_player,
            depth - 1,
            evaluator,
            alpha,
            beta,
            _node_counter=_node_counter,
        )
        if child_val < value:
            value = child_val
            best_action = action
        if value < beta:
            beta = value
        if alpha >= beta:
            break
    return best_action, value


# OrderedDict TT keys: (grid_bytes, current_player, depth_remaining).
# Value payload is agent-specific; see class-level annotations below.
_MaxNTTKey = tuple[bytes, int, int]
_ParanoidTTKey = tuple[bytes, int, int]

_MAX_ITERATIVE_DEEPENING_DEPTH: Final[int] = 64


class MaxNAgent:
    """Fixed-depth max-N agent with optional iterative deepening.

    ``depth`` sets the fixed-budget search depth. When
    :meth:`select_action` is called with ``time_budget_s`` the agent runs
    iterative deepening from depth 2 upward and returns the best action
    from the deepest fully-completed iteration before the deadline.
    """

    name: str

    def __init__(
        self,
        depth: int = 3,
        evaluator: LinearEvaluator | None = None,
        transposition_table_size: int | None = None,
        name: str = "maxn",
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        self._depth: int = depth
        self._evaluator: LinearEvaluator = (
            evaluator if evaluator is not None else default_evaluator()
        )
        self._tt_size: int | None = transposition_table_size
        self._tt: OrderedDict[_MaxNTTKey, tuple[int, NDArray[np.float64]]] = OrderedDict()
        self.name = name
        self.last_nodes: int | None = None
        self.last_depth_completed: int | None = None

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
                f"MaxNAgent called for player {player_id} with no legal actions"
            )
        if state.current_player != player_id:
            raise ValueError(
                f"MaxNAgent: state.current_player={state.current_player} "
                f"!= player_id={player_id}"
            )

        if time_budget_s is None:
            counter: list[int] = [0]
            action, _ = maxn_search(
                state, player_id, self._depth, self._evaluator, _node_counter=counter
            )
            self.last_nodes = counter[0]
            self.last_depth_completed = self._depth
            return action

        deadline = time.perf_counter() + time_budget_s
        best_action = legal[0]
        total_counter: list[int] = [0]
        depth = 2
        self.last_depth_completed = None
        while depth <= _MAX_ITERATIVE_DEEPENING_DEPTH:
            if time.perf_counter() >= deadline:
                break
            iter_counter: list[int] = [0]
            action, _ = maxn_search(
                state, player_id, depth, self._evaluator, _node_counter=iter_counter
            )
            if time.perf_counter() <= deadline:
                best_action = action
                self.last_depth_completed = depth
                total_counter[0] += iter_counter[0]
                depth += 1
            else:
                break
        self.last_nodes = total_counter[0]
        return best_action

    def reset(self) -> None:
        self._tt.clear()
        self.last_nodes = None
        self.last_depth_completed = None


class ParanoidAgent:
    """Paranoid alpha-beta agent with optional iterative deepening."""

    name: str

    def __init__(
        self,
        depth: int = 3,
        evaluator: LinearEvaluator | None = None,
        transposition_table_size: int | None = None,
        name: str = "paranoid",
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        self._depth: int = depth
        self._evaluator: LinearEvaluator = (
            evaluator if evaluator is not None else default_evaluator()
        )
        self._tt_size: int | None = transposition_table_size
        self._tt: OrderedDict[_ParanoidTTKey, tuple[int, float, int]] = OrderedDict()
        self.name = name
        self.last_nodes: int | None = None
        self.last_depth_completed: int | None = None

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
                f"ParanoidAgent called for player {player_id} with no legal actions"
            )
        if state.current_player != player_id:
            raise ValueError(
                f"ParanoidAgent: state.current_player={state.current_player} "
                f"!= player_id={player_id}"
            )

        if time_budget_s is None:
            counter: list[int] = [0]
            action, _ = paranoid_search(
                state, player_id, self._depth, self._evaluator, _node_counter=counter
            )
            self.last_nodes = counter[0]
            self.last_depth_completed = self._depth
            return action

        deadline = time.perf_counter() + time_budget_s
        best_action = legal[0]
        total_counter: list[int] = [0]
        depth = 2
        self.last_depth_completed = None
        while depth <= _MAX_ITERATIVE_DEEPENING_DEPTH:
            if time.perf_counter() >= deadline:
                break
            iter_counter: list[int] = [0]
            action, _ = paranoid_search(
                state, player_id, depth, self._evaluator, _node_counter=iter_counter
            )
            if time.perf_counter() <= deadline:
                best_action = action
                self.last_depth_completed = depth
                total_counter[0] += iter_counter[0]
                depth += 1
            else:
                break
        self.last_nodes = total_counter[0]
        return best_action

    def reset(self) -> None:
        self._tt.clear()
        self.last_nodes = None
        self.last_depth_completed = None


__all__ = [
    "MaxNAgent",
    "ParanoidAgent",
    "maxn_search",
    "paranoid_search",
]
