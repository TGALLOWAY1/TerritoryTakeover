"""Debugging aids for game states: ASCII, matplotlib, GIF export, invariant checks.

These are optional helpers for training/debugging. matplotlib and pillow are
imported lazily so the core engine stays numpy-only; install with
``pip install territory_takeover[viz]`` to enable the image-producing entry
points.
"""

from __future__ import annotations

from collections import deque
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

from .constants import EMPTY, OWNED_CODES
from .state import GameState

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_OWNED_CHARS: tuple[str, ...] = ("1", "2", "3", "4")

# Indexed by tile code: _TILE_COLORS[v] for v in 0..4.
_TILE_COLORS: tuple[str, ...] = (
    "#ffffff",
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
)
_HEAD_EDGE_COLORS: tuple[str, ...] = ("#0a3d66", "#7a1516", "#1a5e1a", "#4a2a6b")

# Public aliases so downstream renderers (e.g. :mod:`territory_takeover.viz_html`)
# share exactly one palette with the matplotlib/GIF renderer. Keep them in
# lockstep if the underscore-prefixed tuples are ever tuned.
TILE_COLORS: tuple[str, ...] = _TILE_COLORS
HEAD_EDGE_COLORS: tuple[str, ...] = _HEAD_EDGE_COLORS

_EMPTY_HEAD: tuple[int, int] = (-1, -1)


def render_ascii(state: GameState) -> str:
    """Return a human-readable ASCII grid with bracketed player heads.

    One char per cell: ``.`` for empty, ``1..4`` for owned territory. Each
    player's head cell is rendered as ``[d]`` (three chars wide) so rows will
    not be perfectly column-aligned around heads by design — the brackets
    make the head impossible to miss at a glance.
    """
    grid = state.grid
    h, w = grid.shape
    cells: list[list[str]] = []
    for r in range(h):
        row: list[str] = []
        for c in range(w):
            v = int(grid.item(r, c))
            if v == EMPTY:
                row.append(".")
            elif v in OWNED_CODES:
                row.append(_OWNED_CHARS[OWNED_CODES.index(v)])
            else:
                row.append("?")
        cells.append(row)

    for p in state.players:
        if p.head == _EMPTY_HEAD:
            continue
        hr, hc = p.head
        if 0 <= hr < h and 0 <= hc < w:
            cells[hr][hc] = f"[{_OWNED_CHARS[p.player_id]}]"

    header = (
        f"turn={state.turn_number} current={state.current_player} "
        f"done={state.done} winner={state.winner}"
    )
    body = "\n".join("".join(row) for row in cells)
    return header + "\n" + body


def _draw_state(ax: Axes, state: GameState, show_heads: bool) -> None:
    """Render `state` into `ax`. Shared by render_matplotlib and save_game_gif."""
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(list(_TILE_COLORS))
    h, w = state.grid.shape
    ax.imshow(state.grid, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"turn={state.turn_number} current={state.current_player} "
        f"winner={state.winner}"
    )

    if show_heads:
        for p in state.players:
            if p.head == _EMPTY_HEAD:
                continue
            hr, hc = p.head
            if not (0 <= hr < h and 0 <= hc < w):
                continue
            edge = _HEAD_EDGE_COLORS[p.player_id]
            rect = patches.Rectangle(
                (hc - 0.5, hr - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=edge,
                linewidth=2.0,
            )
            ax.add_patch(rect)


def render_matplotlib(
    state: GameState,
    ax: Axes | None = None,
    show_heads: bool = True,
) -> Axes:
    """Render `state` with matplotlib. Creates a figure if `ax` is None.

    Owned tiles use 4 vivid colors, one per player. Heads are drawn as hollow
    outlined squares when `show_heads=True`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    _draw_state(ax, state, show_heads=show_heads)
    return ax


def save_game_gif(
    trajectory: list[GameState],
    path: str,
    fps: int = 4,
) -> None:
    """Render a full game trajectory to an animated GIF.

    `trajectory` should be a list of `GameState` snapshots (e.g. captured with
    `state.copy()` after each step). Callers are expected to downsample long
    games themselves — we render every frame as given to keep this simple.
    """
    if not trajectory:
        raise ValueError("trajectory is empty; nothing to render")
    if fps < 1:
        raise ValueError(f"fps must be >= 1; got {fps}")

    import matplotlib.pyplot as plt
    from PIL import Image

    frames: list[Image.Image] = []
    for snap in trajectory:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        _draw_state(ax, snap, show_heads=True)
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    duration_ms = round(1000.0 / fps)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def validate_state(state: GameState, deep: bool = False) -> list[str]:
    """Check invariants on `state`. Returns a list of human-readable violations.

    An empty list means the state is internally consistent. Cheap checks (run
    by default):

    - Each player's ``head`` sits on that player's OWNED code in the grid
      (for players that have spawned).
    - ``territory_count`` matches the count of that player's OWNED tiles on
      the grid.
    - ``empty_count`` (when seeded) matches the grid's EMPTY-cell count.
    - ``alive_count`` (when seeded) matches the number of alive players.

    When ``deep=True``, each player's owned region is additionally verified
    to be 4-connected (territory grows one adjacent cell at a time from the
    head, so a disconnected region indicates state corruption).
    """
    violations: list[str] = []
    grid = state.grid
    h, w = grid.shape

    for p in state.players:
        pid = p.player_id
        if p.head == _EMPTY_HEAD and p.territory_count == 0:
            continue

        hr, hc = p.head
        if not (0 <= hr < h and 0 <= hc < w):
            violations.append(f"player {pid}: head {p.head} out of bounds")
        elif int(grid.item(hr, hc)) != OWNED_CODES[pid]:
            violations.append(
                f"player {pid}: grid at head {p.head} is "
                f"{int(grid.item(hr, hc))}, expected OWNED code {OWNED_CODES[pid]}"
            )

        expected_owned = int((grid == OWNED_CODES[pid]).sum())
        if p.territory_count != expected_owned:
            violations.append(
                f"player {pid}: territory_count={p.territory_count} but grid "
                f"has {expected_owned} OWNED tiles"
            )

    if state.empty_count >= 0:
        actual_empty = int((grid == EMPTY).sum())
        if state.empty_count != actual_empty:
            violations.append(
                f"empty_count={state.empty_count} but grid has "
                f"{actual_empty} EMPTY cells"
            )

    if state.alive_count >= 0:
        actual_alive = sum(1 for p in state.players if p.alive)
        if state.alive_count != actual_alive:
            violations.append(
                f"alive_count={state.alive_count} but {actual_alive} players "
                f"have alive=True"
            )

    if deep:
        violations.extend(_deep_connectivity_check(state))

    return violations


def _deep_connectivity_check(state: GameState) -> list[str]:
    """Verify each player's owned region is 4-connected (BFS from the head)."""
    grid = state.grid
    h, w = grid.shape
    violations: list[str] = []

    for p in state.players:
        if p.head == _EMPTY_HEAD:
            continue
        own = OWNED_CODES[p.player_id]
        total = int((grid == own).sum())
        if total == 0:
            continue
        hr, hc = p.head
        if not (0 <= hr < h and 0 <= hc < w) or int(grid.item(hr, hc)) != own:
            continue  # already reported by the cheap checks

        visited = np.zeros((h, w), dtype=np.bool_)
        visited[hr, hc] = True
        q: deque[tuple[int, int]] = deque([(hr, hc)])
        reached = 1
        while q:
            rr, cc = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and not visited[nr, nc]
                    and int(grid.item(nr, nc)) == own
                ):
                    visited[nr, nc] = True
                    reached += 1
                    q.append((nr, nc))

        if reached != total:
            violations.append(
                f"deep: player {p.player_id} territory is disconnected "
                f"({reached} of {total} cells reachable from head)"
            )
    return violations
