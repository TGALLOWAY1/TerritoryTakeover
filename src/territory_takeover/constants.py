"""Tile codes, directions, and board defaults."""

from typing import Final

EMPTY: Final[int] = 0

PLAYER_1_OWNED: Final[int] = 1
PLAYER_2_OWNED: Final[int] = 2
PLAYER_3_OWNED: Final[int] = 3
PLAYER_4_OWNED: Final[int] = 4

# (drow, dcol) in order N, S, W, E.
DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

DEFAULT_BOARD_HEIGHT: Final[int] = 40
DEFAULT_BOARD_WIDTH: Final[int] = 40
DEFAULT_NUM_PLAYERS: Final[int] = 4

OWNED_CODES: Final[tuple[int, ...]] = (
    PLAYER_1_OWNED,
    PLAYER_2_OWNED,
    PLAYER_3_OWNED,
    PLAYER_4_OWNED,
)
