"""MCTS tree node.

Design decisions worth remembering:

- Values are stored as an N-dim vector (one entry per player), not a scalar.
  TerritoryTakeover has up to 4 players, so a single tree must be useful for
  any of them under any search variant (UCT, paranoid, max-N, RAVE).
  During selection, UCB is evaluated from the *acting parent's* player
  perspective — the UCT driver passes ``parent.player_to_move`` as
  ``player_id`` when scoring a child. This avoids the classic multi-player
  bug of the mover optimizing for the opponent who will act next.

- Rewards at leaves are per-player ``territory_total`` normalized to
  ``[0, 1]`` by dividing by ``board_area``. Normalization is the caller's
  responsibility (see ``rollout.py``); the node just stores whatever goes
  into ``total_value`` / ``terminal_value``. Keeping leaf values in a fixed
  range means the UCB exploration constant ``c`` stays meaningful across
  20x20, 30x30, and 40x40 boards without retuning.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


class MCTSNode:
    __slots__ = (
        "children",
        "incoming_action",
        "parent",
        "player_to_move",
        "state",
        "terminal",
        "terminal_value",
        "total_value",
        "untried_actions",
        "visits",
    )

    def __init__(
        self,
        state: GameState,
        parent: MCTSNode | None = None,
        incoming_action: int | None = None,
    ) -> None:
        self.state: GameState = state
        self.parent: MCTSNode | None = parent
        self.incoming_action: int | None = incoming_action
        self.player_to_move: int = state.current_player
        self.children: dict[int, MCTSNode] = {}
        self.untried_actions: list[int] = []
        self.visits: int = 0
        self.total_value: NDArray[np.float64] = np.zeros(len(state.players), dtype=np.float64)
        self.terminal: bool = state.done
        self.terminal_value: NDArray[np.float64] | None = None

    def is_fully_expanded(self) -> bool:
        # Fully expanded iff at least one child has been attached and no
        # untried action remains. A freshly-constructed node (no children,
        # untried not yet populated) is *not* fully expanded — the UCT
        # driver must populate `untried_actions` on first visit.
        return bool(self.children) and len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return self.terminal

    def q_value(self, player_id: int) -> float:
        if self.visits == 0:
            return 0.0
        return float(self.total_value[player_id] / self.visits)

    def ucb_score(self, player_id: int, c: float) -> float:
        if self.visits == 0:
            return math.inf
        parent = self.parent
        # Root has no parent; UCB is undefined there. Return the mean as a
        # defensive fallback rather than raising — callers should not score
        # the root, but we'd rather be robust than crash mid-search.
        if parent is None or parent.visits <= 0:
            return self.q_value(player_id)
        mean_for_parent_player = self.q_value(player_id)
        exploration = c * math.sqrt(math.log(parent.visits) / self.visits)
        return mean_for_parent_player + exploration
