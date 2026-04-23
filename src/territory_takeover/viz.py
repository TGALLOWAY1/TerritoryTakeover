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

from .constants import CLAIMED_CODES, EMPTY, PATH_CODES
from .state import GameState

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_PATH_CHARS: tuple[str, ...] = ("1", "2", "3", "4")
_CLAIMED_CHARS: tuple[str, ...] = ("A", "B", "C", "D")

# Indexed by tile code: _TILE_COLORS[v] for v in 0..8.
_TILE_COLORS: tuple[str, ...] = (
    "#ffffff",
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#aec7e8", "#ff9896", "#98df8a", "#c5b0d5",
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

    One char per cell: ``.`` for empty, ``1..4`` for path tiles, ``A..D`` for
    claimed territory. Each player's head cell is rendered as ``[d]`` (three
    chars wide) so rows will not be perfectly column-aligned around heads by
    design — the brackets make the head impossible to miss at a glance.
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
            elif v in PATH_CODES:
                row.append(_PATH_CHARS[PATH_CODES.index(v)])
            elif v in CLAIMED_CODES:
                row.append(_CLAIMED_CHARS[CLAIMED_CODES.index(v)])
            else:
                row.append("?")
        cells.append(row)

    for p in state.players:
        if p.head == _EMPTY_HEAD:
            continue
        hr, hc = p.head
        if 0 <= hr < h and 0 <= hc < w:
            cells[hr][hc] = f"[{_PATH_CHARS[p.player_id]}]"

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
    ax.imshow(state.grid, cmap=cmap, vmin=0, vmax=8, interpolation="nearest")
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

    Path tiles use 4 vivid colors; claimed territory uses 4 lighter variants of
    the same hues. Heads are drawn as hollow outlined squares when
    `show_heads=True`.
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

    - Every cell in each player's ``path_set`` matches that player's PATH code
      in the grid.
    - ``head == path[-1]`` for each non-empty player.
    - ``set(path) == path_set`` (no ordered-vs-set drift or dup entries).
    - No two players share a path cell.
    - ``claimed_count`` matches the count of that player's CLAIMED tiles on
      the grid.

    When ``deep=True``, an expensive boundary flood-fill verifies every
    CLAIMED cell is unreachable from the grid boundary when treating PATH
    tiles as walls (i.e. the cell is actually fenced in).
    """
    violations: list[str] = []
    grid = state.grid
    h, w = grid.shape

    seen: dict[tuple[int, int], int] = {}
    for p in state.players:
        pid = p.player_id
        if p.head == _EMPTY_HEAD and not p.path:
            continue

        if not p.path:
            violations.append(f"player {pid}: head={p.head} but path is empty")
        elif p.path[-1] != p.head:
            violations.append(
                f"player {pid}: head {p.head} != path[-1] {p.path[-1]}"
            )

        path_as_set = set(p.path)
        if len(path_as_set) != len(p.path):
            violations.append(
                f"player {pid}: path has duplicate cells (len={len(p.path)}, "
                f"unique={len(path_as_set)})"
            )
        if path_as_set != p.path_set:
            violations.append(
                f"player {pid}: set(path) does not match path_set "
                f"(symmetric_diff={sorted(path_as_set ^ p.path_set)[:5]})"
            )

        expected_code = PATH_CODES[pid]
        for cell in p.path_set:
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                violations.append(f"player {pid}: path cell {cell} out of bounds")
                continue
            actual = int(grid.item(r, c))
            if actual != expected_code:
                violations.append(
                    f"player {pid}: grid at {cell} is {actual}, "
                    f"expected PATH code {expected_code}"
                )
            other = seen.get(cell)
            if other is not None:
                violations.append(
                    f"cell {cell} claimed by both player {other} and player {pid}"
                )
            else:
                seen[cell] = pid

        expected_claimed = int((grid == CLAIMED_CODES[pid]).sum())
        if p.claimed_count != expected_claimed:
            violations.append(
                f"player {pid}: claimed_count={p.claimed_count} but grid has "
                f"{expected_claimed} CLAIMED tiles"
            )

    if deep:
        violations.extend(_deep_fence_check(state))

    return violations


def _deep_fence_check(state: GameState) -> list[str]:
    """Flood from boundary over (EMPTY or CLAIMED); any CLAIMED reached is unfenced."""
    grid = state.grid
    h, w = grid.shape
    path_codes = set(PATH_CODES)

    traversable = np.ones((h, w), dtype=np.bool_)
    for code in path_codes:
        traversable &= grid != code

    visited = np.zeros((h, w), dtype=np.bool_)
    q: deque[tuple[int, int]] = deque()

    for cc in range(w):
        if traversable[0, cc]:
            visited[0, cc] = True
            q.append((0, cc))
        if h > 1 and traversable[h - 1, cc]:
            visited[h - 1, cc] = True
            q.append((h - 1, cc))
    for rr in range(1, h - 1):
        if traversable[rr, 0]:
            visited[rr, 0] = True
            q.append((rr, 0))
        if w > 1 and traversable[rr, w - 1]:
            visited[rr, w - 1] = True
            q.append((rr, w - 1))

    while q:
        rr, cc = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not visited[nr, nc]
                and traversable[nr, nc]
            ):
                visited[nr, nc] = True
                q.append((nr, nc))

    claimed_set = set(CLAIMED_CODES)
    violations: list[str] = []
    reached_claimed = visited & np.isin(grid, list(claimed_set))
    if reached_claimed.any():
        coords = np.argwhere(reached_claimed)
        first = tuple(int(x) for x in coords[0])
        total = int(reached_claimed.sum())
        violations.append(
            f"deep: {total} claimed cell(s) reachable from boundary "
            f"(first: {first}, code={int(grid.item(first[0], first[1]))})"
        )
    return violations
