"""Compact discrete state keys for tabular Q-learning.

A key compresses the full :class:`GameState` into a small tuple of small ints
so it can index a Python ``dict`` Q-table. The encoding trades global board
information for tractability:

    (head_r, head_c, nbr_N, nbr_S, nbr_W, nbr_E, phase)

Each ``nbr_*`` is one of four categorical classes (out-of-bounds, empty,
own territory, any-opponent territory) so that the same encoder shape works
for 2- and 4-player variants — all opponents collapse into a single ``opp``
class. ``phase`` is a coarse 4-level bucket over the fraction of the grid
that is still empty, which lets the agent learn early-vs-late strategy
without blowing up the state space.

The design deliberately drops everything but the current head's 4-neighborhood
and a global phase cue. That's the ceiling of what a tabular method can learn
here; documented in KEY_FINDINGS.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from territory_takeover.constants import DIRECTIONS, EMPTY, OWNED_CODES

if TYPE_CHECKING:
    from territory_takeover.state import GameState


# --- Neighbor categorical codes -------------------------------------------

NBR_OOB: Final[int] = 0
NBR_EMPTY: Final[int] = 1
NBR_OWN: Final[int] = 2
NBR_OPP: Final[int] = 3

# --- Game-phase buckets ---------------------------------------------------

PHASE_EARLY: Final[int] = 0  # empty_fraction > 0.80
PHASE_MID: Final[int] = 1    # 0.55 < empty_fraction <= 0.80
PHASE_LATE: Final[int] = 2   # 0.25 < empty_fraction <= 0.55
PHASE_END: Final[int] = 3    # empty_fraction <= 0.25

_PHASE_THRESHOLDS: Final[tuple[float, float, float]] = (0.80, 0.55, 0.25)


# --- Key type -------------------------------------------------------------

StateKey = tuple[int, int, int, int, int, int, int]
"""``(head_r, head_c, nbr_N, nbr_S, nbr_W, nbr_E, phase)``.

Concrete tuple type (not ``tuple[int, ...]``) so mypy can verify ``len`` and
indexing.
"""


def _classify_cell(code: int, own_code: int) -> int:
    """Map a raw grid value to one of the neighbor enums."""
    if code == EMPTY:
        return NBR_EMPTY
    if code == own_code:
        return NBR_OWN
    for c in OWNED_CODES:
        if code == c:
            return NBR_OPP
    # Unknown grid code — should not happen in a well-formed state. Fall back
    # to EMPTY rather than raising, since the Q-table is a heuristic anyway
    # and we do not want to destabilize training on a malformed tile.
    return NBR_EMPTY


def _game_phase(state: GameState) -> int:
    """Bucket the fraction of EMPTY cells into one of four phases.

    Uses a cheap ``(grid == 0).mean()`` over the whole int8 array — at 40x40
    that's still microseconds; the encoder is called once per decision, not
    per state copy.
    """
    grid = state.grid
    empty_fraction = float((grid == EMPTY).mean())
    e1, e2, e3 = _PHASE_THRESHOLDS
    if empty_fraction > e1:
        return PHASE_EARLY
    if empty_fraction > e2:
        return PHASE_MID
    if empty_fraction > e3:
        return PHASE_LATE
    return PHASE_END


def encode_state(state: GameState, player_id: int) -> StateKey:
    """Encode ``state`` from ``player_id``'s perspective.

    The head's absolute ``(r, c)`` are included so the agent can learn
    boundary-aware behavior. Each of the four orthogonal neighbors is
    classified with :func:`_classify_cell`; out-of-bounds neighbors produce
    :data:`NBR_OOB`. The trailing phase bucket is the same for all players
    (it's a global property of the board).
    """
    player = state.players[player_id]
    r, c = player.head
    grid = state.grid
    h, w = grid.shape
    own_code = OWNED_CODES[player_id]

    cells: list[int] = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.append(_classify_cell(int(grid.item(nr, nc)), own_code))
        else:
            cells.append(NBR_OOB)

    phase = _game_phase(state)
    return (int(r), int(c), cells[0], cells[1], cells[2], cells[3], phase)


__all__ = [
    "NBR_EMPTY",
    "NBR_OOB",
    "NBR_OPP",
    "NBR_OWN",
    "PHASE_EARLY",
    "PHASE_END",
    "PHASE_LATE",
    "PHASE_MID",
    "StateKey",
    "encode_state",
]
