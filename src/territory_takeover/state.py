"""Game state dataclasses and cheap-copy semantics for tree search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .constants import (
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    DEFAULT_NUM_PLAYERS,
    EMPTY,
    OWNED_CODES,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


_REPR_CHARS: dict[int, str] = {
    EMPTY: ".",
    OWNED_CODES[0]: "1",
    OWNED_CODES[1]: "2",
    OWNED_CODES[2]: "3",
    OWNED_CODES[3]: "4",
}


@dataclass(slots=True)
class PlayerState:
    """Per-player view of the game.

    The grid is the source of truth for cell ownership; `territory_count`
    is a cache that every mutation must keep in lockstep with the grid's
    count of that player's OWNED cells. `head` is the player's current
    position (always one of their owned cells).
    """

    player_id: int
    head: tuple[int, int]
    territory_count: int
    alive: bool
    # Liveness-witness cache for `engine.has_reachable_empty`: an EMPTY cell
    # adjacent to this player's territory found by the last successful check.
    # Because territory only grows (and stays connected) and EMPTY cells are
    # only ever claimed, the witness proves liveness until the moment that
    # exact cell stops being EMPTY — making the per-turn liveness check O(1)
    # amortized. None = no witness (never checked, or last check failed).
    alive_witness: tuple[int, int] | None = None


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
    # Scratch reachability mask reused by the liveness BFS
    # (`engine.has_reachable_empty`) to avoid allocating a fresh `(H, W)`
    # buffer per call. Not cloned on `copy()` — each `GameState` gets its own
    # scratch; contents are meaningless between calls (the engine uses a
    # monotonically increasing stamp to avoid zeroing between calls).
    _scratch_reachable: NDArray[np.int32] | None = None
    # Monotonically increasing stamp consumed by the liveness BFS. Paired
    # with `_scratch_reachable` so one value per call suffices to mark
    # "visited this call".
    _reach_stamp: int = 0
    # Cached count of EMPTY cells on the grid. Maintained incrementally by
    # the engine (decrements on each claim) so full-board termination is a
    # constant-time check. -1 is a "not yet seeded" sentinel; callers that
    # skip `new_game`/`reset` get a lazy rebuild the first time the engine
    # needs it.
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
                head=(-1, -1),
                territory_count=0,
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
        - Scalars (int/bool/None) and the immutable `head` tuple are copied
          by value.
        - `_scratch_reachable` is NOT copied — cloned states get a fresh
          lazily-allocated scratch buffer (it carries no game information).
        """
        new_players = [
            PlayerState(
                player_id=p.player_id,
                head=p.head,
                territory_count=p.territory_count,
                alive=p.alive,
                alive_witness=p.alive_witness,
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
            _reach_stamp=0,
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
