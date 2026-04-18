"""Multi-source BFS Voronoi partition over a :class:`GameState`.

Each living player's head is seeded at distance 0; a cell is labelled with the
player whose head reaches it first, ``-1`` when the cell is unreachable
(walls, claimed territory, opponent paths) or equidistant from more than one
player (contested). This is the standard Tron-AI notion of territory.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.constants import EMPTY

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


def voronoi_partition(state: GameState) -> NDArray[np.int8]:
    """Return an (H, W) int8 array labelling each cell with its BFS owner.

    Values: ``0..3`` for the winning player, ``-1`` for unreachable or
    contested cells. Walls are any non-``EMPTY`` tile (paths, including other
    players' heads, and claimed territory). Living players' own head cells are
    seeded at distance 0 and therefore carry their owner in the output.

    The hot loop runs over flat Python lists rather than numpy arrays because
    numpy's scalar indexing overhead dominates for small boards; on a 40x40
    grid this keeps a full partition under the ~200 us perf target.
    """
    grid = state.grid
    h, w = grid.shape
    n = h * w
    # Flat Python lists beat numpy scalar indexing in this Python-level loop.
    grid_flat: list[int] = grid.ravel().tolist()
    owner_flat: list[int] = [-1] * n
    dist_flat: list[int] = [-1] * n
    queue: deque[tuple[int, int, int]] = deque()
    popleft = queue.popleft
    append = queue.append

    for p in state.players:
        if p.alive:
            r, c = p.head
            idx = r * w + c
            owner_flat[idx] = p.player_id
            dist_flat[idx] = 0
            append((r, c, idx))

    last_row = h - 1
    last_col = w - 1

    while queue:
        r, c, idx = popleft()
        d_next = dist_flat[idx] + 1
        src = owner_flat[idx]

        # North
        if r > 0:
            nidx = idx - w
            if grid_flat[nidx] == EMPTY:
                nd = dist_flat[nidx]
                if nd == -1:
                    dist_flat[nidx] = d_next
                    owner_flat[nidx] = src
                    append((r - 1, c, nidx))
                elif nd == d_next and owner_flat[nidx] != src:
                    owner_flat[nidx] = -1

        # South
        if r < last_row:
            nidx = idx + w
            if grid_flat[nidx] == EMPTY:
                nd = dist_flat[nidx]
                if nd == -1:
                    dist_flat[nidx] = d_next
                    owner_flat[nidx] = src
                    append((r + 1, c, nidx))
                elif nd == d_next and owner_flat[nidx] != src:
                    owner_flat[nidx] = -1

        # West
        if c > 0:
            nidx = idx - 1
            if grid_flat[nidx] == EMPTY:
                nd = dist_flat[nidx]
                if nd == -1:
                    dist_flat[nidx] = d_next
                    owner_flat[nidx] = src
                    append((r, c - 1, nidx))
                elif nd == d_next and owner_flat[nidx] != src:
                    owner_flat[nidx] = -1

        # East
        if c < last_col:
            nidx = idx + 1
            if grid_flat[nidx] == EMPTY:
                nd = dist_flat[nidx]
                if nd == -1:
                    dist_flat[nidx] = d_next
                    owner_flat[nidx] = src
                    append((r, c + 1, nidx))
                elif nd == d_next and owner_flat[nidx] != src:
                    owner_flat[nidx] = -1

    return np.array(owner_flat, dtype=np.int8).reshape(h, w)


def reachable_area(state: GameState, player_id: int) -> int:
    """Number of cells won by ``player_id`` in the Voronoi partition."""
    return int(np.count_nonzero(voronoi_partition(state) == player_id))
