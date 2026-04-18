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
from .engine import (
    IllegalMoveError,
    StepResult,
    compute_terminal_reward,
    new_game,
    reset,
    step,
)
from .state import GameState, PlayerState


def __getattr__(name: str) -> object:
    """Lazy-import the optional gymnasium wrappers so core imports stay light."""
    if name in ("TerritoryTakeoverEnv", "MultiAgentEnv"):
        from . import gym_env

        return getattr(gym_env, name)
    raise AttributeError(f"module 'territory_takeover' has no attribute {name!r}")

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
    "IllegalMoveError",
    "MultiAgentEnv",
    "PlayerState",
    "StepResult",
    "TerritoryTakeoverEnv",
    "compute_terminal_reward",
    "new_game",
    "reset",
    "step",
]
