"""Rollout / playout policies used by MCTS.

The default :func:`uniform_rollout` simulates uniformly-random play to
the end of the episode and returns the per-player territory vector
normalized to ``[0, 1]`` (see :func:`_terminal_value`). Normalization to
a fixed range is what lets the UCB exploration constant ``c`` stay
meaningful across 20x20, 30x30, and 40x40 boards without retuning.

A rollout function has signature ``(GameState, np.random.Generator) ->
NDArray[np.float64]`` of shape ``(num_players,)``. The caller in
:mod:`territory_takeover.search.mcts.uct` is responsible for passing a
:meth:`GameState.copy` — rollouts mutate their input state in place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.actions import legal_actions
from territory_takeover.engine import step

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


RolloutFn = Callable[["GameState", np.random.Generator], "NDArray[np.float64]"]


def _terminal_value(state: GameState) -> NDArray[np.float64]:
    """Per-player ``(path_length + claimed_count) / board_area`` in ``[0, 1]``.

    The grid is square so ``state.grid.size`` is the total number of
    cells (board area). Path tiles plus claimed tiles is the same
    quantity used by :func:`territory_takeover.engine._compute_winner`,
    so the rollout's leaf value is consistent with the engine's notion
    of "territory".
    """
    area = float(state.grid.size)
    return np.array(
        [(len(p.path) + p.claimed_count) / area for p in state.players],
        dtype=np.float64,
    )


def uniform_rollout(
    state: GameState, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Play uniformly-random legal moves until the game ends; return leaf value.

    Mutates ``state`` in place. The engine's :func:`_advance_turn` skips
    seats with no legal moves (marking them ``alive = False``) so the
    current player at any non-terminal state is guaranteed to have at
    least one legal action; the empty-``legal`` branch is purely
    defensive.
    """
    while not state.done:
        legal = legal_actions(state, state.current_player)
        # Defensive: engine invariant says current_player always has legal
        # moves at a non-terminal state (_advance_turn skips dead seats).
        # If somehow not, fall back to action 0 with strict=False so the
        # engine marks the player dead and advances rather than raising.
        action = legal[int(rng.integers(len(legal)))] if legal else 0
        step(state, action, strict=False)
    return _terminal_value(state)


__all__ = ["RolloutFn", "uniform_rollout"]
