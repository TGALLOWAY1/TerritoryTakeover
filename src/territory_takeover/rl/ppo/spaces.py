"""Observation encoder for Phase 3b PPO.

The encoder turns a :class:`GameState` into two float32 tensors that can be
consumed by the actor-critic network (see :mod:`.network`):

- ``grid_planes``: shape ``(2N + 2, H, W)``.
  - Channels ``0 .. N - 1``: one per player's PATH cells, with the active
    player rotated to channel ``0`` so every input looks symmetric to the
    network ("I'm always channel 0").
  - Channels ``N .. 2N - 1``: one per player's CLAIMED cells, same rotation.
  - Channel ``2N``: EMPTY mask.
  - Channel ``2N + 1``: one-hot head position of the active player (all zeros
    if the active player has been eliminated and has no head).

- ``scalar_features``: shape ``(3 + N,)``.
  - Index 0: normalized turn number (``turn / (H * W)``, clamped to ``[0, 1]``).
  - Indices 1..N: per-player normalized claimed counts (``claimed / (H * W)``),
    rotated so the active player is index 1.
  - Index N + 1: board fill ratio (``1 - empty / (H * W)``).
  - Index N + 2: fraction of the active player's cells on the board
    (path + claimed, normalized) — cheap "how long is my snake" feature that
    correlates with enclosure potential.

Rotation is a pure channel / index permutation — the underlying player IDs
are untouched, so this can be inverted at inference time if needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.constants import CLAIMED_CODES, EMPTY, PATH_CODES

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


def _rotated_player_order(active_player: int, num_players: int) -> list[int]:
    """Return player IDs with ``active_player`` first.

    Example: ``_rotated_player_order(2, 4) == [2, 3, 0, 1]``. Rotating rather
    than only moving the active to slot 0 preserves relative ordering of
    opponents, which may matter for larger player counts.
    """
    return [(active_player + i) % num_players for i in range(num_players)]


def encode_observation(
    state: GameState, player_id: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Encode ``state`` from ``player_id``'s perspective.

    Returns a pair ``(grid_planes, scalar_features)`` of float32 numpy arrays.

    The encoder never mutates ``state``. It reads ``state.grid`` directly
    (vectorized numpy comparisons) rather than iterating ``path_set`` for each
    player, which scales better to 40x40 boards.
    """
    num_players = len(state.players)
    h, w = state.grid.shape
    grid = state.grid

    planes = np.zeros((2 * num_players + 2, h, w), dtype=np.float32)
    order = _rotated_player_order(player_id, num_players)

    for out_idx, pid in enumerate(order):
        planes[out_idx] = (grid == PATH_CODES[pid]).astype(np.float32)
        planes[num_players + out_idx] = (grid == CLAIMED_CODES[pid]).astype(
            np.float32
        )

    planes[2 * num_players] = (grid == EMPTY).astype(np.float32)

    head_plane = planes[2 * num_players + 1]
    head = state.players[player_id].head
    if state.players[player_id].alive and head != (-1, -1):
        head_plane[head[0], head[1]] = 1.0

    total_cells = float(h * w)
    empty_count = float(np.count_nonzero(grid == EMPTY))
    active_occupancy = float(
        np.count_nonzero(grid == PATH_CODES[player_id])
        + np.count_nonzero(grid == CLAIMED_CODES[player_id])
    )

    scalars = np.zeros(3 + num_players, dtype=np.float32)
    scalars[0] = min(state.turn_number / total_cells, 1.0)
    for out_idx, pid in enumerate(order):
        scalars[1 + out_idx] = state.players[pid].claimed_count / total_cells
    scalars[1 + num_players] = 1.0 - empty_count / total_cells
    scalars[2 + num_players] = active_occupancy / total_cells

    return planes, scalars


__all__ = [
    "encode_observation",
]
