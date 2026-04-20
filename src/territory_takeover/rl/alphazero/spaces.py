"""Observation encoder for Phase 3c AlphaZero.

Differs from Phase 3b's encoder in two ways:

1. **Fixed seat ordering.** Phase 3b rotated the active player onto channel
   ``0`` so the network saw a symmetric input. Phase 3c wants a 4-dim value
   head with ``value[i]`` = "expected normalized score for seat ``i``"
   (Petosa & Balch, 2019). That only makes sense if seat identity is
   preserved in the input — otherwise seat ``i``'s plane could land on any
   channel and the head would have nothing stable to learn. We keep seats in
   their native order ``0 .. N - 1``.

2. **Turn one-hot.** Because we no longer rotate, the network needs an
   explicit "whose turn is it" signal. We append ``N`` planes, exactly one
   of which is all-ones (for the active player), the rest zero.

Layout of ``grid_planes`` is therefore ``(3N + 2, H, W)``:

- Channels ``0 .. N - 1``: per-seat PATH cells.
- Channels ``N .. 2N - 1``: per-seat CLAIMED cells.
- Channel ``2N``: EMPTY mask.
- Channel ``2N + 1``: one-hot head of the active player.
- Channels ``2N + 2 .. 3N + 1``: per-seat turn one-hot (all-ones on the
  active player's plane, zero elsewhere).

Scalars match Phase 3b in shape ``(3 + N,)`` but without rotation: claimed
counts are in native seat order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.constants import CLAIMED_CODES, EMPTY, PATH_CODES

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


def grid_channel_count(num_players: int) -> int:
    """Return the number of grid input channels for ``num_players``."""
    return 3 * num_players + 2


def scalar_feature_dim(num_players: int) -> int:
    """Return the length of the scalar feature vector for ``num_players``."""
    return 3 + num_players


def encode_az_observation(
    state: GameState, active_player: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Encode ``state`` for Phase 3c: fixed seat order + turn one-hot.

    ``active_player`` drives only the head-one-hot channel and the turn
    one-hot — all per-seat planes are in their canonical order ``0 .. N-1``.
    The encoder never mutates ``state``.
    """
    num_players = len(state.players)
    h, w = state.grid.shape
    grid = state.grid

    planes = np.zeros((grid_channel_count(num_players), h, w), dtype=np.float32)

    for pid in range(num_players):
        planes[pid] = (grid == PATH_CODES[pid]).astype(np.float32)
        planes[num_players + pid] = (grid == CLAIMED_CODES[pid]).astype(np.float32)

    planes[2 * num_players] = (grid == EMPTY).astype(np.float32)

    head = state.players[active_player].head
    if state.players[active_player].alive and head != (-1, -1):
        planes[2 * num_players + 1, head[0], head[1]] = 1.0

    planes[2 * num_players + 2 + active_player] = 1.0

    total_cells = float(h * w)
    empty_count = float(np.count_nonzero(grid == EMPTY))
    active_occupancy = float(
        np.count_nonzero(grid == PATH_CODES[active_player])
        + np.count_nonzero(grid == CLAIMED_CODES[active_player])
    )

    scalars = np.zeros(scalar_feature_dim(num_players), dtype=np.float32)
    scalars[0] = min(state.turn_number / total_cells, 1.0)
    for pid in range(num_players):
        scalars[1 + pid] = state.players[pid].claimed_count / total_cells
    scalars[1 + num_players] = 1.0 - empty_count / total_cells
    scalars[2 + num_players] = active_occupancy / total_cells

    return planes, scalars


__all__ = [
    "encode_az_observation",
    "grid_channel_count",
    "scalar_feature_dim",
]
