"""territory_takeover — grid-based territory capture game engine."""

from .constants import (
    CLAIMED_CODES,
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    DEFAULT_NUM_PLAYERS,
    DIRECTIONS,
    EMPTY,
    PATH_CODES,
    PLAYER_1_CLAIMED,
    PLAYER_1_PATH,
    PLAYER_2_CLAIMED,
    PLAYER_2_PATH,
    PLAYER_3_CLAIMED,
    PLAYER_3_PATH,
    PLAYER_4_CLAIMED,
    PLAYER_4_PATH,
)
from .state import GameState, PlayerState

__all__ = [
    "CLAIMED_CODES",
    "DEFAULT_BOARD_HEIGHT",
    "DEFAULT_BOARD_WIDTH",
    "DEFAULT_NUM_PLAYERS",
    "DIRECTIONS",
    "EMPTY",
    "PATH_CODES",
    "PLAYER_1_CLAIMED",
    "PLAYER_1_PATH",
    "PLAYER_2_CLAIMED",
    "PLAYER_2_PATH",
    "PLAYER_3_CLAIMED",
    "PLAYER_3_PATH",
    "PLAYER_4_CLAIMED",
    "PLAYER_4_PATH",
    "GameState",
    "PlayerState",
]
