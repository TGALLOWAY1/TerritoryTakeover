"""Tile codes, directions, and board defaults."""

from typing import Final

EMPTY: Final[int] = 0

PLAYER_1_PATH: Final[int] = 1
PLAYER_2_PATH: Final[int] = 2
PLAYER_3_PATH: Final[int] = 3
PLAYER_4_PATH: Final[int] = 4

PLAYER_1_CLAIMED: Final[int] = 5
PLAYER_2_CLAIMED: Final[int] = 6
PLAYER_3_CLAIMED: Final[int] = 7
PLAYER_4_CLAIMED: Final[int] = 8

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

PATH_CODES: Final[tuple[int, ...]] = (
    PLAYER_1_PATH,
    PLAYER_2_PATH,
    PLAYER_3_PATH,
    PLAYER_4_PATH,
)
CLAIMED_CODES: Final[tuple[int, ...]] = (
    PLAYER_1_CLAIMED,
    PLAYER_2_CLAIMED,
    PLAYER_3_CLAIMED,
    PLAYER_4_CLAIMED,
)
