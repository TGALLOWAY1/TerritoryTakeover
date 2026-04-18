"""Per-player scalar features for position evaluation.

Every function takes ``(state, player_id)`` and returns a plain int/float so
that these features can be composed into a heuristic evaluator or fed into
learned value heads without extra reshaping.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from territory_takeover.actions import legal_actions
from territory_takeover.eval.voronoi import reachable_area

if TYPE_CHECKING:
    from territory_takeover.state import GameState


def mobility(state: GameState, player_id: int) -> int:
    """Number of legal placements (0..4) from the player's current head."""
    return len(legal_actions(state, player_id))


def reachable_area_feature(state: GameState, player_id: int) -> int:
    """Cells this player wins in the Voronoi partition."""
    return reachable_area(state, player_id)


def head_opponent_distance(state: GameState, player_id: int) -> float:
    """Manhattan distance from this player's head to the nearest living opponent.

    Returns ``math.inf`` when no other player is alive.
    """
    r, c = state.players[player_id].head
    best = math.inf
    for other in state.players:
        if other.player_id == player_id or not other.alive:
            continue
        orr, orc = other.head
        d = float(abs(r - orr) + abs(c - orc))
        if d < best:
            best = d
    return best


def claimed_count(state: GameState, player_id: int) -> int:
    """Number of cells currently claimed by this player."""
    return state.players[player_id].claimed_count


def path_length(state: GameState, player_id: int) -> int:
    """Length of this player's snake path (including the head)."""
    return len(state.players[player_id].path)


def territory_total(state: GameState, player_id: int) -> int:
    """Scoring total: path length plus claimed-cell count."""
    p = state.players[player_id]
    return len(p.path) + p.claimed_count
