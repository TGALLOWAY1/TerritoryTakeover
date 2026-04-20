"""Game state dataclasses and cheap-copy semantics for tree search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .constants import (
    CLAIMED_CODES,
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    DEFAULT_NUM_PLAYERS,
    EMPTY,
    PATH_CODES,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


_REPR_CHARS: dict[int, str] = {
    EMPTY: ".",
    PATH_CODES[0]: "1",
    PATH_CODES[1]: "2",
    PATH_CODES[2]: "3",
    PATH_CODES[3]: "4",
    CLAIMED_CODES[0]: "A",
    CLAIMED_CODES[1]: "B",
    CLAIMED_CODES[2]: "C",
    CLAIMED_CODES[3]: "D",
}


@dataclass(slots=True)
class PlayerState:
    player_id: int
    path: list[tuple[int, int]]
    path_set: set[tuple[int, int]]
    head: tuple[int, int]
    claimed_count: int
    alive: bool


@dataclass(slots=True)
class GameState:
    grid: NDArray[np.int8]
    players: list[PlayerState]
    current_player: int = 0
    turn_number: int = 0
    winner: int | None = None
    done: bool = False
    # Cached count of players with `alive=True`. Maintained incrementally by
    # the engine so terminal detection can avoid an O(N) scan every `step`.
    # -1 is a "not yet seeded" sentinel; the engine seeds it at game creation
    # and `copy()` propagates the current value.
    alive_count: int = -1
    # Scratch reachability mask reused by `detect_and_apply_enclosure` to
    # avoid allocating a fresh `(H, W)` buffer per BFS. Not cloned on `copy()`
    # — each `GameState` gets its own scratch; contents are meaningless between
    # calls (the engine uses a monotonically increasing stamp to avoid having
    # to zero between calls).
    _scratch_reachable: NDArray[np.int32] | None = None
    # Monotonically increasing stamp consumed by the enclosure BFS. Paired with
    # `_scratch_reachable` so two values per call suffice to distinguish
    # "visited this call" from "confirmed pocket".
    _enclosure_stamp: int = 0
    # Cached count of EMPTY cells on the grid. Maintained incrementally by the
    # engine (decrements on each path placement and each enclosure claim) so
    # `detect_and_apply_enclosure` can early-exit when the boundary BFS reaches
    # every EMPTY cell (`reachable_count == empty_count`) — skipping the
    # O(H*W) mask/assignment entirely on the common "trigger fires, nothing
    # enclosed" path. -1 is a "not yet seeded" sentinel; callers that skip
    # `new_game`/`reset` get a lazy rebuild the first time the engine needs it.
    empty_count: int = -1

    @classmethod
    def empty(
        cls,
        height: int = DEFAULT_BOARD_HEIGHT,
        width: int = DEFAULT_BOARD_WIDTH,
        num_players: int = DEFAULT_NUM_PLAYERS,
    ) -> GameState:
        """Construct a blank board with `num_players` stateless players.

        Player heads are placed at (-1, -1) as a sentinel; engine setup
        (see :func:`territory_takeover.engine.new_game`) is responsible for
        seeding real starting positions.
        """
        grid = np.zeros((height, width), dtype=np.int8)
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
        return cls(
            grid=grid,
            players=players,
            alive_count=num_players,
            empty_count=height * width,
        )

    def copy(self) -> GameState:
        """Independent copy suitable for tree search.

        - `grid` via `np.ndarray.copy` (C memcpy).
        - Per-player `path` (list) and `path_set` (set) shallow-copied;
          their contents (tuples of ints) are immutable, so sharing is safe.
        - Scalars (int/bool/None) and the immutable `head` tuple are copied by value.
        - `_scratch_reachable` is NOT copied — cloned states get a fresh
          lazily-allocated scratch buffer (it carries no game information).
        """
        new_players = [
            PlayerState(
                player_id=p.player_id,
                path=p.path.copy(),
                path_set=p.path_set.copy(),
                head=p.head,
                claimed_count=p.claimed_count,
                alive=p.alive,
            )
            for p in self.players
        ]
        return GameState(
            grid=self.grid.copy(),
            players=new_players,
            current_player=self.current_player,
            turn_number=self.turn_number,
            winner=self.winner,
            done=self.done,
            alive_count=self.alive_count,
            _scratch_reachable=None,
            _enclosure_stamp=0,
            empty_count=self.empty_count,
        )

    def __repr__(self) -> str:
        header = (
            f"GameState(turn={self.turn_number}, current_player={self.current_player}, "
            f"winner={self.winner}, done={self.done})"
        )
        rows = []
        for row in self.grid:
            rows.append("".join(_REPR_CHARS.get(int(v), "?") for v in row))
        return header + "\n" + "\n".join(rows)
