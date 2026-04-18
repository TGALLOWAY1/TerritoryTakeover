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

    @classmethod
    def empty(
        cls,
        height: int = DEFAULT_BOARD_HEIGHT,
        width: int = DEFAULT_BOARD_WIDTH,
        num_players: int = DEFAULT_NUM_PLAYERS,
    ) -> GameState:
        """Construct a blank board with `num_players` stateless players.

        Player heads are placed at (-1, -1) as a sentinel; engine setup
        (not yet implemented) is responsible for seeding real starting positions.
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
        return cls(grid=grid, players=players)

    def copy(self) -> GameState:
        """Independent copy suitable for tree search.

        - `grid` via `np.ndarray.copy` (C memcpy).
        - Per-player `path` (list) and `path_set` (set) shallow-copied;
          their contents (tuples of ints) are immutable, so sharing is safe.
        - Scalars (int/bool/None) and the immutable `head` tuple are copied by value.
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
