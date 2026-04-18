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

from territory_takeover.actions import legal_actions
from territory_takeover.constants import DIRECTIONS, EMPTY, PATH_CODES
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


def enclosure_potential(
    state: GameState, player_id: int, *, cheap: bool = False
) -> tuple[int, int] | None:
    """Shortest loop this player could close from its head, plus projected area.

    BFS from the head over empty cells (walls: anything non-EMPTY, including
    opponent paths, claimed territory, and the player's own path). As soon as
    an empty cell is adjacent to a self-path tile that is neither the head nor
    its immediate predecessor, that's a closing point: stop at BFS distance
    ``d`` and simulate closing the loop.

    The hypothetical loop consists of the shortest BFS trail back to the head
    plus the existing path segment from the contact point to the head. An
    edge flood fill (identical in spirit to
    :func:`territory_takeover.engine.detect_and_apply_enclosure`) measures the
    area that would be enclosed by that loop.

    Returns ``(distance_to_close, projected_enclosed_area)`` or ``None`` when:

    - the player is dead / unspawned,
    - the path is too short to close a loop (< 3 cells),
    - the head has no empty neighbours,
    - no self-path contact is reachable,
    - or (only when ``cheap=False``) the projected enclosed area is zero.

    When ``cheap=True``, the flood-fill step is skipped entirely: the function
    returns ``(distance_to_close, 0)`` as soon as a closing contact is found.
    Use this when you only need the distance component and the worst-case
    500 us budget matters (unbound BFS on 40x40 can exceed that otherwise).
    """
    p = state.players[player_id]
    if not p.alive:
        return None
    path = p.path
    if len(path) < 3:
        return None
    head = path[-1]
    if head == (-1, -1):
        return None
    predecessor = path[-2]

    grid = state.grid
    h, w = grid.shape
    hr, hc = head

    # Parent grid stores packed (r * w + c) of the predecessor empty cell in
    # the BFS tree; -1 = unvisited, -2 = seeded directly from the head.
    parent = np.full((h, w), -1, dtype=np.int32)
    dist = np.full((h, w), -1, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()

    # Closing-eligible self-path cells: everything except head and predecessor.
    own_code = PATH_CODES[player_id]

    def _is_closer(r: int, c: int) -> bool:
        if grid.item(r, c) != own_code:
            return False
        rc = (r, c)
        return rc != head and rc != predecessor

    # Seed with head's empty neighbours. If any neighbour is a closing-eligible
    # self-path cell, a zero-step loop isn't possible (head is already on the
    # path; the predecessor contact is excluded by definition), so we only
    # enqueue empty cells.
    for dr, dc in DIRECTIONS:
        nr, nc = hr + dr, hc + dc
        if 0 <= nr < h and 0 <= nc < w and grid.item(nr, nc) == EMPTY:
            dist[nr, nc] = 1
            parent[nr, nc] = -2
            q.append((nr, nc))

    contact_empty: tuple[int, int] | None = None
    contact_path: tuple[int, int] | None = None
    while q:
        r, c = q.popleft()
        # Closing check first: if this empty cell touches a closing-eligible
        # self-path tile, we've found the shortest close.
        closed_here = False
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and _is_closer(nr, nc):
                contact_empty = (r, c)
                contact_path = (nr, nc)
                closed_here = True
                break
        if closed_here:
            break
        # Otherwise expand over empty neighbours.
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and dist[nr, nc] == -1
                and grid.item(nr, nc) == EMPTY
            ):
                dist[nr, nc] = dist[r, c] + 1
                parent[nr, nc] = r * w + c
                q.append((nr, nc))

    if contact_empty is None or contact_path is None:
        return None

    distance = int(dist[contact_empty])
    if cheap:
        return distance, 0

    # Reconstruct the BFS trail of empty cells from head-adjacent seed to
    # contact_empty (inclusive on both ends).
    trail: list[tuple[int, int]] = []
    cr, cc = contact_empty
    while True:
        trail.append((cr, cc))
        pidx = int(parent[cr, cc])
        if pidx == -2:
            break
        cr, cc = divmod(pidx, w)

    # Build a wall mask for the edge flood fill. Every non-empty grid cell is
    # a wall (including the player's full existing path, which already contains
    # the loop_segment from contact_path back to head). On top of that, the
    # BFS trail is the simulated continuation — empty cells that would be
    # filled in by the hypothetical moves — so those become walls too.
    wall = (grid != EMPTY).copy()
    for r, c in trail:
        wall[r, c] = True

    # Edge flood fill: everything reachable from the border (over non-wall
    # cells) is "outside". Whatever is left is enclosed.
    reachable_mask = np.zeros((h, w), dtype=np.bool_)
    fq: deque[tuple[int, int]] = deque()
    for cc in range(w):
        if not wall[0, cc]:
            reachable_mask[0, cc] = True
            fq.append((0, cc))
        if h > 1 and not wall[h - 1, cc]:
            reachable_mask[h - 1, cc] = True
            fq.append((h - 1, cc))
    for rr in range(1, h - 1):
        if not wall[rr, 0]:
            reachable_mask[rr, 0] = True
            fq.append((rr, 0))
        if w > 1 and not wall[rr, w - 1]:
            reachable_mask[rr, w - 1] = True
            fq.append((rr, w - 1))

    while fq:
        rr, ccx = fq.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = rr + dr, ccx + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not reachable_mask[nr, nc]
                and not wall[nr, nc]
            ):
                reachable_mask[nr, nc] = True
                fq.append((nr, nc))

    enclosed_count = int(((~wall) & ~reachable_mask).sum())
    if enclosed_count == 0:
        return None
    return distance, enclosed_count


def choke_pressure(state: GameState, player_id: int) -> float:
    """Scalar in [0, 1] indicating how squeezed this player is by opponents.

    Computes ``1 - (contested_voronoi_area / solo_bfs_area)`` where:

    - ``contested_voronoi_area`` is the Voronoi-partition reachable area
      (opponents active),
    - ``solo_bfs_area`` is the single-source BFS reachable area from the head
      over empty space with opponents' paths/territory still treated as walls
      but without the Voronoi race: i.e. the player's "absolute" reach if no
      contested ties existed.

    Returns ``1.0`` when ``solo_bfs_area`` is zero (already trapped) or when
    the player is dead. Higher values mean more pressure from opponents.
    """
    p = state.players[player_id]
    if not p.alive:
        return 1.0
    head = p.head
    if head == (-1, -1):
        return 1.0

    grid = state.grid
    h, w = grid.shape
    hr, hc = head

    visited = np.zeros((h, w), dtype=np.bool_)
    q: deque[tuple[int, int]] = deque()
    for dr, dc in DIRECTIONS:
        nr, nc = hr + dr, hc + dc
        if 0 <= nr < h and 0 <= nc < w and grid.item(nr, nc) == EMPTY:
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

    solo = int(visited.sum())
    if solo == 0:
        return 1.0

    # voronoi_partition seeds the head at distance 0 and attributes it to the
    # player, so reachable_area always includes the head cell. solo counts
    # empty cells only; subtract the head to make the two measures comparable.
    contested = reachable_area(state, player_id) - 1
    if contested < 0:
        contested = 0
    ratio = contested / solo
    if ratio > 1.0:
        ratio = 1.0
    return 1.0 - ratio
