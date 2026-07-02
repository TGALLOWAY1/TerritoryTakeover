"""Per-player scalar features for position evaluation.

Every function takes ``(state, player_id)`` and returns a plain int/float so
that these features can be composed into a heuristic evaluator or fed into
learned value heads without extra reshaping.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.actions import claiming_actions, legal_actions
from territory_takeover.constants import DIRECTIONS, EMPTY, OWNED_CODES
from territory_takeover.eval.voronoi import reachable_area

if TYPE_CHECKING:
    from territory_takeover.state import GameState


def mobility(state: GameState, player_id: int) -> int:
    """Number of legal moves (0..4) from the player's current head.

    Includes traversal moves over own territory, not just claims.
    """
    return len(legal_actions(state, player_id))


def claiming_mobility(state: GameState, player_id: int) -> int:
    """Number of moves (0..4) from the head that claim a new cell."""
    return len(claiming_actions(state, player_id))


def reachable_area_feature(state: GameState, player_id: int) -> int:
    """Cells this player wins in the Voronoi partition (territory included)."""
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


def territory_total(state: GameState, player_id: int) -> int:
    """Scoring total: number of cells this player has visited/claimed."""
    return state.players[player_id].territory_count


def _solo_reachable_empties(state: GameState, player_id: int) -> int:
    """Number of EMPTY cells reachable from the player's territory.

    Single-player reachability (no race against opponents): BFS over EMPTY
    cells seeded from every empty neighbor of the player's own cells.
    Opponents' territory is a wall. This is the player's "absolute" claimable
    reach if the game froze everyone else.
    """
    grid = state.grid
    h, w = grid.shape
    own = OWNED_CODES[player_id]

    visited = np.zeros((h, w), dtype=np.bool_)
    q: deque[tuple[int, int]] = deque()
    owned_cells = np.argwhere(grid == own)
    for rr, cc in owned_cells:
        r, c = int(rr), int(cc)
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not visited[nr, nc]
                and grid.item(nr, nc) == EMPTY
            ):
                visited[nr, nc] = True
                q.append((nr, nc))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not visited[nr, nc]
                and grid.item(nr, nc) == EMPTY
            ):
                visited[nr, nc] = True
                q.append((nr, nc))

    return int(visited.sum())


def choke_pressure(state: GameState, player_id: int) -> float:
    """Scalar in [0, 1] indicating how squeezed this player is by opponents.

    Computes ``1 - (contested_voronoi_empties / solo_reachable_empties)``
    where:

    - ``contested_voronoi_empties`` is the Voronoi-partition reachable area
      minus the player's own territory (i.e. the EMPTY cells the player wins
      in the race against opponents),
    - ``solo_reachable_empties`` is the EMPTY-cell area reachable from the
      player's territory with opponents frozen (see
      :func:`_solo_reachable_empties`).

    Returns ``1.0`` when ``solo_reachable_empties`` is zero (already walled
    out) or when the player is dead. Higher values mean more pressure from
    opponents.
    """
    p = state.players[player_id]
    if not p.alive:
        return 1.0
    if p.head == (-1, -1):
        return 1.0

    solo = _solo_reachable_empties(state, player_id)
    if solo == 0:
        return 1.0

    contested = reachable_area(state, player_id) - p.territory_count
    if contested < 0:
        contested = 0
    ratio = contested / solo
    if ratio > 1.0:
        ratio = 1.0
    return 1.0 - ratio
