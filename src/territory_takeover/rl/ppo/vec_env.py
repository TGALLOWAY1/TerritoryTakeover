"""Synchronous vectorized env wrapper for Phase 3b PPO.

Runs ``N`` independent :class:`territory_takeover.state.GameState` instances
in lockstep and exposes the shapes :class:`.ppo_core.RolloutBuffer` consumes:

- ``grid``:    ``(N, 2P+2, H, W)`` float32
- ``scalars``: ``(N, 3+P)`` float32
- ``mask``:    ``(N, 4)`` bool
- ``rewards``: ``(N,)`` float32
- ``dones``:   ``(N,)`` bool (True on termination OR truncation)

Opponents between the learning agent's turns are auto-played via per-env
:data:`Policy` callables (default: uniform random over legal actions).
Finished episodes are auto-reset in place; the terminal observation is
surfaced via ``info[i]["terminal_observation"]`` so the training loop can
bootstrap value targets correctly.

Sync-only by design: the CNN forward pass dominates Python loop overhead at
the env counts we use (8--32), and a single-process wrapper keeps stack
traces and slot states introspectable. A drop-in async variant can replace
this class later without changing the public surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from territory_takeover.engine import compute_terminal_reward, new_game, step
from territory_takeover.gym_env import (
    _encode_observation as _encode_dict_obs,
)
from territory_takeover.gym_env import (
    _uniform_random_policy,
)
from territory_takeover.rl.ppo.spaces import encode_observation, legal_mask_array

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


Policy = Callable[[dict[str, Any], np.random.Generator], int]
"""Per-player opponent policy: ``(obs_dict, rng) -> action_index``.

Matches the signature used by :mod:`territory_takeover.gym_env` so policies
written for the single-env wrapper drop in unchanged.
"""


class Observation(TypedDict):
    """Batched observation dict returned by :meth:`SyncVectorTerritoryEnv.reset`/``step``."""

    grid: NDArray[np.float32]
    scalars: NDArray[np.float32]
    mask: NDArray[np.bool_]


@dataclass(slots=True)
class _EnvSlot:
    """Per-env mutable state held by :class:`SyncVectorTerritoryEnv`."""

    state: GameState
    agent_player_id: int
    opponent_policies: dict[int, Policy]
    episode_steps: int = 0
    episode_return: float = 0.0
    last_reset_seed: int | None = None


class SyncVectorTerritoryEnv:
    """N-wide synchronous vectorized TerritoryTakeover env.

    Parameters
    ----------
    num_envs:
        Number of parallel games.
    board_size:
        Square-grid side length.
    num_players:
        2 or 4 (default engine spawns are only defined for these).
    agent_player_ids:
        Seat the learning agent occupies in each env. An ``int`` is broadcast
        to all envs; a list must have length ``num_envs``.
    opponent_policies:
        Either a single ``dict[int, Policy]`` (shared across envs) or a list
        of such dicts of length ``num_envs``. Missing keys default to uniform
        random over legal actions. Keys must NOT include the learning agent's
        seat for that env.
    reward_scheme:
        ``"step"`` returns the engine per-step reward as-is. ``"sparse"``
        additionally adds ``{+1, 0, -1}`` on the terminal transition, per
        :func:`territory_takeover.engine.compute_terminal_reward`.
    max_steps:
        Truncation horizon, counted in learning-agent steps per episode.
        ``None`` disables truncation.
    seed:
        Master RNG seed. Per-env game seeds are drawn from this generator so
        the whole wrapper is reproducible from a single int.
    """

    def __init__(
        self,
        num_envs: int,
        board_size: int = 40,
        num_players: int = 4,
        agent_player_ids: int | list[int] = 0,
        opponent_policies: dict[int, Policy] | list[dict[int, Policy]] | None = None,
        reward_scheme: str = "step",
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1; got {num_envs}")
        if reward_scheme not in ("step", "sparse"):
            raise ValueError(
                f"reward_scheme must be 'step' or 'sparse'; got {reward_scheme!r}"
            )

        if isinstance(agent_player_ids, int):
            agent_ids: list[int] = [agent_player_ids] * num_envs
        else:
            if len(agent_player_ids) != num_envs:
                raise ValueError(
                    f"agent_player_ids length {len(agent_player_ids)} != num_envs {num_envs}"
                )
            agent_ids = list(agent_player_ids)
        for aid in agent_ids:
            if not 0 <= aid < num_players:
                raise ValueError(
                    f"agent_player_id {aid} out of range for num_players {num_players}"
                )

        opp_list: list[dict[int, Policy]]
        if opponent_policies is None:
            opp_list = [{} for _ in range(num_envs)]
        elif isinstance(opponent_policies, dict):
            opp_list = [dict(opponent_policies) for _ in range(num_envs)]
        else:
            if len(opponent_policies) != num_envs:
                raise ValueError(
                    f"opponent_policies list length {len(opponent_policies)} "
                    f"!= num_envs {num_envs}"
                )
            opp_list = [dict(d) for d in opponent_policies]
        for i, (aid, opps) in enumerate(zip(agent_ids, opp_list, strict=True)):
            if aid in opps:
                raise ValueError(
                    f"env {i}: opponent_policies contains the learning agent's seat {aid}"
                )

        self.num_envs = num_envs
        self.board_size = board_size
        self.num_players = num_players
        self.reward_scheme = reward_scheme
        self.max_steps = max_steps

        self.single_observation_shapes: dict[str, tuple[int, ...]] = {
            "grid": (2 * num_players + 2, board_size, board_size),
            "scalars": (3 + num_players,),
            "mask": (4,),
        }
        self.single_action_n: int = 4

        self._master_rng: np.random.Generator = np.random.default_rng(seed)
        # Per-env RNG used by opponent policies. Derived once from the master
        # RNG so opponent action sampling is reproducible across runs.
        self._env_rngs: list[np.random.Generator] = [
            np.random.default_rng(int(self._master_rng.integers(0, 2**63 - 1)))
            for _ in range(num_envs)
        ]

        self._slots: list[_EnvSlot] = []
        for i in range(num_envs):
            slot = _EnvSlot(
                state=self._new_state(),
                agent_player_id=agent_ids[i],
                opponent_policies=opp_list[i],
            )
            self._slots.append(slot)
            self._reset_slot(i)

        self._closed: bool = False

    # ------------------------------------------------------------------ API

    def reset(self, seeds: list[int] | None = None) -> Observation:
        """Reset every slot. Returns the stacked post-reset observation.

        If ``seeds`` is given, each slot is seeded with the corresponding int;
        otherwise seeds are drawn from the master RNG.
        """
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(
                f"seeds length {len(seeds)} != num_envs {self.num_envs}"
            )
        for i in range(self.num_envs):
            s = seeds[i] if seeds is not None else None
            self._reset_slot(i, seed=s)
        return self._stack_obs()

    def step(
        self, actions: NDArray[np.int64]
    ) -> tuple[
        Observation,
        NDArray[np.float32],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        """Apply one learning-agent action per env; auto-play opponents; auto-reset.

        Parameters
        ----------
        actions:
            Int array of shape ``(num_envs,)`` — the action chosen by the
            learning agent in each slot.

        Returns
        -------
        obs:
            Stacked observation dict after the step (post auto-reset for
            finished envs).
        rewards:
            ``(num_envs,)`` float32 — engine reward for the learning agent's
            move in each env, plus the sparse terminal bonus on terminating
            transitions when ``reward_scheme == "sparse"``. Truncation alone
            does NOT add a terminal bonus.
        dones:
            ``(num_envs,)`` bool — True for both termination and truncation.
            The caller can distinguish via ``info[i]``.
        infos:
            Per-env dicts. On terminal / truncated transitions include keys
            ``{"terminal_observation", "episode_steps", "episode_return",
            "winner", "truncated"}``.
        """
        if self._closed:
            raise RuntimeError("step() called on a closed vec env")
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f"actions shape {actions.shape} != ({self.num_envs},)"
            )

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            reward, terminated, truncated, info = self._step_slot(i, int(actions[i]))
            rewards[i] = reward
            dones[i] = terminated or truncated
            infos[i] = info
            if terminated or truncated:
                self._reset_slot(i)

        return self._stack_obs(), rewards, dones, infos

    def close(self) -> None:
        """Drop references to game states. Idempotent."""
        self._slots = []
        self._closed = True

    # ------------------------------------------------------------- internal

    def _new_state(self) -> GameState:
        """Build a fresh ``GameState`` seeded from the master RNG."""
        s = int(self._master_rng.integers(0, 2**63 - 1))
        return new_game(
            board_size=self.board_size,
            num_players=self.num_players,
            seed=s,
        )

    def _reset_slot(self, index: int, seed: int | None = None) -> None:
        """Rebuild slot ``index`` with a fresh ``GameState`` and advance to agent."""
        slot = self._slots[index]
        if seed is None:
            seed = int(self._master_rng.integers(0, 2**63 - 1))
        slot.state = new_game(
            board_size=self.board_size,
            num_players=self.num_players,
            seed=seed,
        )
        slot.episode_steps = 0
        slot.episode_return = 0.0
        slot.last_reset_seed = seed
        self._advance_to_agent(slot, self._env_rngs[index])

    def _advance_to_agent(
        self, slot: _EnvSlot, rng: np.random.Generator
    ) -> None:
        """Auto-play opponents until the learning agent is to move or game ends."""
        state = slot.state
        while not state.done and state.current_player != slot.agent_player_id:
            pid = state.current_player
            policy = slot.opponent_policies.get(pid, _uniform_random_policy)
            opp_obs = _encode_dict_obs(state, pid)
            action = int(policy(opp_obs, rng))
            step(state, action, strict=False)

    def _step_slot(
        self, index: int, action: int
    ) -> tuple[float, bool, bool, dict[str, Any]]:
        """Apply one learning-agent step + opponent advance. Returns (reward, term, trunc, info)."""
        slot = self._slots[index]
        state = slot.state
        info: dict[str, Any] = {}

        if state.done:
            # Defensive — _reset_slot runs after every terminal, so this path
            # should be unreachable under normal use.
            raise RuntimeError(
                f"_step_slot called on a finished env (agent={slot.agent_player_id})"
            )

        # It's always the learning agent's turn here (guaranteed by the prior
        # reset / advance); assert cheaply in case of caller misuse.
        if state.current_player != slot.agent_player_id:
            raise RuntimeError(
                f"vec env slot out of sync: current_player={state.current_player} "
                f"!= agent_player_id={slot.agent_player_id}"
            )

        result = step(state, action, strict=False)
        reward = float(result.reward)

        if not state.done:
            self._advance_to_agent(slot, self._env_rngs[index])

        slot.episode_steps += 1
        terminated = bool(state.done)
        truncated = bool(
            self.max_steps is not None
            and slot.episode_steps >= self.max_steps
            and not terminated
        )

        if terminated and self.reward_scheme == "sparse":
            terminal = compute_terminal_reward(state, scheme="sparse")
            reward += float(terminal[slot.agent_player_id])
            info["winner"] = state.winner

        slot.episode_return += reward

        if terminated or truncated:
            info["terminal_observation"] = self._obs_for_slot(slot)
            info["episode_steps"] = slot.episode_steps
            info["episode_return"] = slot.episode_return
            info["truncated"] = truncated
            if not terminated:
                info["winner"] = None

        return reward, terminated, truncated, info

    def _obs_for_slot(
        self, slot: _EnvSlot
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Encode one slot's observation from the learning agent's perspective."""
        grid, scalars = encode_observation(slot.state, slot.agent_player_id)
        if slot.state.done or not slot.state.players[slot.agent_player_id].alive:
            mask = np.zeros(4, dtype=np.bool_)
        else:
            mask = legal_mask_array(slot.state, slot.agent_player_id)
        return grid, scalars, mask

    def _stack_obs(self) -> Observation:
        """Stack per-slot observations along axis 0."""
        per_env = [self._obs_for_slot(s) for s in self._slots]
        grids = np.stack([g for g, _, _ in per_env], axis=0)
        scalars = np.stack([s for _, s, _ in per_env], axis=0)
        masks = np.stack([m for _, _, m in per_env], axis=0)
        return Observation(grid=grids, scalars=scalars, mask=masks)


__all__ = [
    "Observation",
    "Policy",
    "SyncVectorTerritoryEnv",
]
