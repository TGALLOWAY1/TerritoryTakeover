"""Action definitions and legality checks.

Actions are encoded as direction indices ``0..3`` matching
:data:`territory_takeover.constants.DIRECTIONS` (N, S, W, E). Fixing the
action space at 4 keeps the MCTS prior / policy head shape constant
regardless of board size.

A target cell is legal iff it is in bounds and ``EMPTY``. Note on traversal:
the rules mention free traversal along one's own path before placement, but
since traversal doesn't change the head and placement must extend from the
head, traversal is a no-op for legal move generation. Revisit only if a rule
variant permits branching from arbitrary path tiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import DIRECTIONS, EMPTY

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .state import GameState


def legal_action_mask(state: GameState, player_id: int) -> NDArray[np.bool_]:
    """Return a bool array of shape (4,) marking legal direction indices."""
    r, c = state.players[player_id].head
    h, w = state.grid.shape
    g = state.grid
    mask = np.zeros(4, dtype=np.bool_)
    if r > 0 and g.item(r - 1, c) == EMPTY:
        mask[0] = True
    if r < h - 1 and g.item(r + 1, c) == EMPTY:
        mask[1] = True
    if c > 0 and g.item(r, c - 1) == EMPTY:
        mask[2] = True
    if c < w - 1 and g.item(r, c + 1) == EMPTY:
        mask[3] = True
    return mask


def legal_actions(state: GameState, player_id: int) -> list[int]:
    """Return the list of legal direction indices (0..3) from the head."""
    r, c = state.players[player_id].head
    h, w = state.grid.shape
    g = state.grid
    out: list[int] = []
    if r > 0 and g.item(r - 1, c) == EMPTY:
        out.append(0)
    if r < h - 1 and g.item(r + 1, c) == EMPTY:
        out.append(1)
    if c > 0 and g.item(r, c - 1) == EMPTY:
        out.append(2)
    if c < w - 1 and g.item(r, c + 1) == EMPTY:
        out.append(3)
    return out


def action_to_coord(
    state: GameState, player_id: int, action: int
) -> tuple[int, int]:
    """Resolve a direction index to an absolute (row, col). No bounds check."""
    r, c = state.players[player_id].head
    dr, dc = DIRECTIONS[action]
    return (r + dr, c + dc)
