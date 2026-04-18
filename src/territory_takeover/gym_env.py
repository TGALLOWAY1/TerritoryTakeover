"""Gymnasium and PettingZoo-AEC-style wrappers around the engine.

The single-agent :class:`TerritoryTakeoverEnv` auto-plays opponents between the
agent's turns so standard single-agent RL code (SB3, cleanrl, etc.) can drive
the game without knowing about the engine's turn-based nature. A
:class:`MultiAgentEnv` in the same module exposes a PettingZoo-AEC-style
interface (duck-typed — no pettingzoo dependency) for self-play / MARL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import legal_action_mask
from .engine import compute_terminal_reward, new_game, step

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .state import GameState

    Policy = Callable[[dict[str, Any], np.random.Generator], int]


_RGB_PALETTE: NDArray[np.uint8] = np.array(
    [
        (30, 30, 30),     # 0 EMPTY
        (220, 60, 60),    # 1 P1 path
        (60, 200, 90),    # 2 P2 path
        (70, 130, 230),   # 3 P3 path
        (230, 210, 60),   # 4 P4 path
        (140, 50, 50),    # 5 P1 claimed
        (50, 130, 70),    # 6 P2 claimed
        (60, 90, 150),    # 7 P3 claimed
        (150, 140, 50),   # 8 P4 claimed
    ],
    dtype=np.uint8,
)


def _uniform_random_policy(
    obs: dict[str, Any], rng: np.random.Generator
) -> int:
    """Pick uniformly from legal actions; fall back to 0 if none are legal."""
    mask: NDArray[np.int8] = obs["action_mask"]
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return 0
    return int(rng.choice(legal))


def _make_observation_space(board_size: int, num_players: int) -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(
                low=0, high=8, shape=(board_size, board_size), dtype=np.int8
            ),
            "current_player": spaces.Discrete(num_players),
            "heads": spaces.Box(
                low=-1,
                high=board_size - 1,
                shape=(num_players, 2),
                dtype=np.int32,
            ),
            "action_mask": spaces.MultiBinary(4),
        }
    )


def _encode_observation(state: GameState, viewing_player: int) -> dict[str, Any]:
    """Build an observation dict from the engine state.

    The grid and heads fields are global (fully observable). ``current_player``
    is set to ``viewing_player`` and ``action_mask`` reflects that player's
    legal moves, so each caller sees the board from their own perspective.
    """
    heads = np.array(
        [p.head if p.alive else (-1, -1) for p in state.players],
        dtype=np.int32,
    )
    mask = legal_action_mask(state, viewing_player).astype(np.int8)
    return {
        "grid": state.grid.copy(),
        "current_player": int(viewing_player),
        "heads": heads,
        "action_mask": mask,
    }


class TerritoryTakeoverEnv(gym.Env[dict[str, Any], int]):
    """Single-agent Gymnasium wrapper around the TerritoryTakeover engine.

    Opponents are auto-played via a per-player policy registered with
    :meth:`set_opponent_policy` (default: uniform random over legal actions).
    """

    metadata = {  # noqa: RUF012  # gym.Env declares this as instance attr
        "render_modes": ["ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        board_size: int = 40,
        num_players: int = 4,
        agent_player_id: int = 0,
        opponent_policies: dict[int, Policy] | None = None,
        max_steps: int | None = None,
        render_mode: str | None = None,
        reward_scheme: str = "step",
    ) -> None:
        super().__init__()
        if not 0 <= agent_player_id < num_players:
            raise ValueError(
                f"agent_player_id {agent_player_id} out of range for "
                f"num_players {num_players}"
            )
        if reward_scheme not in ("step", "sparse"):
            raise ValueError(
                f"reward_scheme must be 'step' or 'sparse'; got {reward_scheme!r}"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )

        self.board_size = board_size
        self.num_players = num_players
        self.agent_player_id = agent_player_id
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.reward_scheme = reward_scheme

        self.observation_space = _make_observation_space(board_size, num_players)
        self.action_space = spaces.Discrete(4)

        self._opponent_policies: dict[int, Policy] = dict(opponent_policies or {})
        self._state: GameState | None = None
        self._rng: np.random.Generator = np.random.default_rng()
        self._agent_steps: int = 0

    # ------------------------------------------------------------------ API

    def set_opponent_policy(self, player_id: int, policy: Policy) -> None:
        """Install ``policy`` as the opponent for ``player_id``.

        The policy receives the player's observation dict (including
        ``action_mask``) and the env's RNG, and must return an action index.
        """
        if not 0 <= player_id < self.num_players:
            raise ValueError(
                f"player_id {player_id} out of range for num_players {self.num_players}"
            )
        if player_id == self.agent_player_id:
            raise ValueError(
                f"Cannot set opponent policy for the agent player "
                f"({self.agent_player_id})"
            )
        self._opponent_policies[player_id] = policy

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._state = new_game(
            board_size=self.board_size,
            num_players=self.num_players,
            seed=seed,
        )
        self._agent_steps = 0
        self._advance_to_agent()
        assert self._state is not None
        obs = _encode_observation(self._state, self.agent_player_id)
        info: dict[str, Any] = {
            "current_player": int(self._state.current_player),
            "turn_number": int(self._state.turn_number),
            "done_before_agent_turn": bool(self._state.done),
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        if self._state.done:
            raise RuntimeError(
                "step() called on a finished episode; call reset() first"
            )

        state = self._state
        info: dict[str, Any] = {"opponent_moves": []}

        if state.current_player == self.agent_player_id:
            result = step(state, int(action), strict=False)
            reward = float(result.reward)
            info.update(result.info)
        else:
            # Shouldn't normally happen — _advance_to_agent leaves it as the
            # agent's turn — but handle gracefully if someone constructs a
            # state where the agent starts dead.
            reward = 0.0
            info["skipped_agent_turn"] = True

        if not state.done:
            self._advance_to_agent(info["opponent_moves"])

        self._agent_steps += 1
        terminated = bool(state.done)
        truncated = bool(
            self.max_steps is not None
            and self._agent_steps >= self.max_steps
            and not terminated
        )

        if terminated and self.reward_scheme == "sparse":
            terminal = compute_terminal_reward(state, scheme="sparse")
            reward += float(terminal[self.agent_player_id])
            info["terminal_rewards"] = [float(r) for r in terminal]
            info["winner"] = state.winner

        obs = _encode_observation(state, self.agent_player_id)
        return obs, reward, terminated, truncated, info

    def render(  # type: ignore[override]
        self,
    ) -> str | NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        if self._state is None:
            raise RuntimeError("render() called before reset()")
        if self.render_mode == "ansi":
            return repr(self._state)
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        raise NotImplementedError(f"render_mode={self.render_mode!r}")

    def close(self) -> None:
        self._state = None

    # ------------------------------------------------------------- internal

    def _advance_to_agent(
        self, opponent_moves: list[dict[str, Any]] | None = None
    ) -> None:
        """Auto-play opponents until it's the agent's turn or the game ends."""
        state = self._state
        assert state is not None
        while not state.done and state.current_player != self.agent_player_id:
            pid = state.current_player
            policy = self._opponent_policies.get(pid, _uniform_random_policy)
            opp_obs = _encode_observation(state, pid)
            opp_action = int(policy(opp_obs, self._rng))
            result = step(state, opp_action, strict=False)
            if opponent_moves is not None:
                opponent_moves.append(dict(result.info))

    def _render_rgb(self, cell_size: int = 8) -> NDArray[np.uint8]:
        assert self._state is not None
        grid = self._state.grid
        # Map int8 codes to RGB triples via palette lookup, then upscale.
        small = _RGB_PALETTE[grid.astype(np.intp)]
        return np.repeat(np.repeat(small, cell_size, axis=0), cell_size, axis=1)


class MultiAgentEnv:
    """PettingZoo-AEC-style interface for TerritoryTakeover.

    Duck-types the AEC API (``agents``, ``agent_selection``, ``reset``,
    ``step``, ``observe``, ``last``, ``render``, ``close``) without inheriting
    from ``pettingzoo.AECEnv``, so the core package stays numpy-only.

    Per-agent rewards are summed since the last time that agent observed; when
    the episode terminates, the sparse terminal reward (+1 winner / -1 losers
    / 0 tie) is added for every agent.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["ansi", "rgb_array"],
        "render_fps": 4,
    }

    observation_spaces: dict[str, spaces.Dict]
    action_spaces: dict[str, spaces.Space[Any]]

    def __init__(
        self,
        board_size: int = 40,
        num_players: int = 4,
        render_mode: str | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.board_size = board_size
        self.num_players = num_players
        self.render_mode = render_mode

        self.possible_agents: list[str] = [
            f"player_{i}" for i in range(num_players)
        ]
        obs_space = _make_observation_space(board_size, num_players)
        act_space: spaces.Space[Any] = spaces.Discrete(4)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

        self.agents: list[str] = list(self.possible_agents)
        self.agent_selection: str = self.possible_agents[0]
        self.rewards: dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self._cumulative_rewards: dict[str, float] = {
            a: 0.0 for a in self.possible_agents
        }
        self.terminations: dict[str, bool] = {
            a: False for a in self.possible_agents
        }
        self.truncations: dict[str, bool] = {
            a: False for a in self.possible_agents
        }
        self.infos: dict[str, dict[str, Any]] = {
            a: {} for a in self.possible_agents
        }
        self._state: GameState | None = None

    # ------------------------------------------------------------------ API

    def observation_space(self, agent: str) -> spaces.Dict:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> None:
        self._state = new_game(
            board_size=self.board_size,
            num_players=self.num_players,
            seed=seed,
        )
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        self.agent_selection = f"player_{self._state.current_player}"

    def step(self, action: int) -> None:
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        state = self._state
        moving_agent = self.agent_selection

        # Zero out the moving agent's pending reward before the move; it's
        # about to be overwritten with the freshly-earned step reward.
        self.rewards = {a: 0.0 for a in self.possible_agents}

        if state.done:
            # Episode already ended — no-op step, keep state stable.
            return

        result = step(state, int(action), strict=False)
        self.rewards[moving_agent] = float(result.reward)
        self._cumulative_rewards[moving_agent] += float(result.reward)
        self.infos[moving_agent] = dict(result.info)

        if state.done:
            terminal = compute_terminal_reward(state, scheme="sparse")
            for i, a in enumerate(self.possible_agents):
                self.rewards[a] += float(terminal[i])
                self._cumulative_rewards[a] += float(terminal[i])
                self.terminations[a] = True
                self.infos[a]["winner"] = state.winner
            # Keep all agents in `self.agents` so callers can drain final
            # rewards; PettingZoo AEC convention.
        else:
            # Prune dead players from the active-agents list (AEC convention).
            self.agents = [
                f"player_{p.player_id}" for p in state.players if p.alive
            ]
            self.agent_selection = f"player_{state.current_player}"
            # Mark non-alive players as terminated if they just died.
            for p in state.players:
                if not p.alive:
                    self.terminations[f"player_{p.player_id}"] = True

    def observe(self, agent: str) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("observe() called before reset()")
        pid = int(agent.split("_")[1])
        return _encode_observation(self._state, pid)

    def last(
        self,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        a = self.agent_selection
        return (
            self.observe(a),
            self.rewards[a],
            self.terminations[a],
            self.truncations[a],
            self.infos[a],
        )

    def render(self) -> str | NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        if self._state is None:
            raise RuntimeError("render() called before reset()")
        if self.render_mode == "ansi":
            return repr(self._state)
        if self.render_mode == "rgb_array":
            grid = self._state.grid
            cell_size = 8
            small = _RGB_PALETTE[grid.astype(np.intp)]
            return np.repeat(
                np.repeat(small, cell_size, axis=0), cell_size, axis=1
            )
        raise NotImplementedError(f"render_mode={self.render_mode!r}")

    def close(self) -> None:
        self._state = None
