"""territory_takeover — grid-based territory capture game engine."""

from .constants import (
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    DEFAULT_NUM_PLAYERS,
    DIRECTIONS,
    EMPTY,
    OWNED_CODES,
    PLAYER_1_OWNED,
    PLAYER_2_OWNED,
    PLAYER_3_OWNED,
    PLAYER_4_OWNED,
)
from .engine import (
    IllegalMoveError,
    StepResult,
    compute_terminal_reward,
    has_reachable_empty,
    new_game,
    reset,
    step,
)
from .rollout import simulate_random_rollout
from .state import GameState, PlayerState


def __getattr__(name: str) -> object:
    """Lazy-import the optional gymnasium wrappers so core imports stay light."""
    if name in ("TerritoryTakeoverEnv", "MultiAgentEnv"):
        from . import gym_env

        return getattr(gym_env, name)
    raise AttributeError(f"module 'territory_takeover' has no attribute {name!r}")

__all__ = [
    "DEFAULT_BOARD_HEIGHT",
    "DEFAULT_BOARD_WIDTH",
    "DEFAULT_NUM_PLAYERS",
    "DIRECTIONS",
    "EMPTY",
    "OWNED_CODES",
    "PLAYER_1_OWNED",
    "PLAYER_2_OWNED",
    "PLAYER_3_OWNED",
    "PLAYER_4_OWNED",
    "GameState",
    "IllegalMoveError",
    "MultiAgentEnv",
    "PlayerState",
    "StepResult",
    "TerritoryTakeoverEnv",
    "compute_terminal_reward",
    "has_reachable_empty",
    "new_game",
    "reset",
    "simulate_random_rollout",
    "step",
]
