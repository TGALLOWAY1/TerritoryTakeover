"""Turn resolution, move application, and win detection.

Currently implements game initialization (`new_game`) and in-place reset (`reset`);
move/turn mechanics will follow.
"""

from __future__ import annotations

import numpy as np

from .constants import EMPTY, PATH_CODES
from .state import GameState, PlayerState

_SPAWN_INSET: int = 4


def _default_spawns(board_size: int, num_players: int) -> list[tuple[int, int]]:
    """Canonical symmetric spawns defined for 2- and 4-player games.

    4 players: all four corners inset by `_SPAWN_INSET`.
    2 players: diagonal corners (top-left, bottom-right).
    """
    inset = _SPAWN_INSET
    far = board_size - 1 - inset
    if num_players == 4:
        return [(inset, inset), (inset, far), (far, inset), (far, far)]
    if num_players == 2:
        return [(inset, inset), (far, far)]
    raise ValueError(
        f"Default spawns are defined only for num_players in (2, 4); got {num_players}"
    )


def _seed_player_state(state: GameState, spawns: list[tuple[int, int]]) -> None:
    """Write spawn tiles into `state.grid` and (re)seed each `PlayerState` in place."""
    for i, spawn in enumerate(spawns):
        r, c = spawn
        state.grid[r, c] = PATH_CODES[i]
        p = state.players[i]
        p.path = [spawn]
        p.path_set = {spawn}
        p.head = spawn
        p.claimed_count = 0
        p.alive = True


def new_game(
    board_size: int = 40,
    num_players: int = 4,
    spawn_positions: list[tuple[int, int]] | None = None,
    seed: int | None = None,
) -> GameState:
    """Build a freshly-initialized `GameState`.

    Args:
        board_size: Side length of the square grid.
        num_players: Number of players (1\u20134; tile encoding only supports up to 4).
        spawn_positions: Optional per-player spawn cells. If omitted, symmetric
            corner spawns are used (supported for `num_players` in (2, 4) only).
        seed: Optional RNG seed. When `spawn_positions is None` and `seed is not None`,
            the default spawn list is permuted so player-to-corner assignment is
            randomized but reproducible. Ignored when explicit `spawn_positions`
            are supplied.

    Raises:
        ValueError: if `board_size` < 1, `num_players` not in [1, 4], spawn count
            mismatches `num_players`, any spawn is out of bounds, or spawn cells
            are not unique.
    """
    if board_size < 1:
        raise ValueError(f"board_size must be >= 1; got {board_size}")
    if not 1 <= num_players <= 4:
        raise ValueError(f"num_players must be in [1, 4]; got {num_players}")

    if spawn_positions is None:
        spawns = _default_spawns(board_size, num_players)
        if seed is not None:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(spawns))
            spawns = [spawns[int(i)] for i in order]
    else:
        if len(spawn_positions) != num_players:
            raise ValueError(
                f"spawn_positions length {len(spawn_positions)} != num_players {num_players}"
            )
        for r, c in spawn_positions:
            if not (0 <= r < board_size and 0 <= c < board_size):
                raise ValueError(
                    f"spawn ({r}, {c}) is out of bounds for board_size {board_size}"
                )
        if len(set(spawn_positions)) != num_players:
            raise ValueError(f"spawn_positions contain duplicates: {spawn_positions}")
        spawns = list(spawn_positions)

    grid = np.zeros((board_size, board_size), dtype=np.int8)
    players = [
        PlayerState(
            player_id=i,
            path=[],
            path_set=set(),
            head=(-1, -1),
            claimed_count=0,
            alive=True,
        )
        for i in range(num_players)
    ]
    state = GameState(
        grid=grid,
        players=players,
        current_player=0,
        turn_number=0,
        winner=None,
        done=False,
    )
    _seed_player_state(state, spawns)
    return state


def reset(state: GameState) -> None:
    """Restore `state` in place to a fresh default-spawn initial configuration.

    Reuses the existing grid buffer (no reallocation) so this is cheap inside
    RL training loops. Infers `board_size` from `state.grid` and `num_players`
    from `len(state.players)`. Requires `num_players in (2, 4)`; callers that
    need custom spawns should call `new_game` again instead.
    """
    height, width = state.grid.shape
    if height != width:
        raise ValueError(f"reset requires a square grid; got shape {state.grid.shape}")
    board_size = height
    num_players = len(state.players)

    spawns = _default_spawns(board_size, num_players)

    state.grid.fill(EMPTY)
    _seed_player_state(state, spawns)
    state.current_player = 0
    state.turn_number = 0
    state.winner = None
    state.done = False
