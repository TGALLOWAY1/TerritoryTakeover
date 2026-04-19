"""Tabular Q-learning agent conforming to :class:`Agent`.

The agent carries a ``dict[StateKey, NDArray[float32]]`` Q-table. Missing
keys read as zero vectors; writes happen only through :meth:`td_update`.

Policy:
  - ``self._greedy`` flag bypasses exploration entirely (eval mode).
  - Otherwise, with probability ``self.epsilon`` the agent picks uniformly
    among *legal* actions (never among all 4 — the zero-probability-illegal
    requirement is strict).
  - On exploitation, illegal actions are masked to ``-inf`` before argmax so
    they can never be chosen, even if a stale Q-value happens to be positive.

TD update:
  - If ``s_next`` is ``None`` (terminal): target = ``r``.
  - Else: target = ``r + gamma * max(Q[s_next][legal_next])``. The same
    legality mask is applied to the next-state max so we never bootstrap
    from a Q-value for an illegal action.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.actions import legal_action_mask

from .state_encoder import StateKey, encode_state

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


_N_ACTIONS: int = 4


@dataclass(slots=True)
class QConfig:
    """Hyperparameters for :class:`TabularQAgent`."""

    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_fraction: float = 0.5
    total_episodes: int = 500_000


@dataclass(slots=True)
class _QState:
    """Mutable container for everything that gets pickled on save/load."""

    cfg: QConfig
    q: dict[StateKey, NDArray[np.float32]] = field(default_factory=dict)
    episode: int = 0


class TabularQAgent:
    """Tabular Q-learning agent with legal-action masking.

    Implements the structural :class:`territory_takeover.search.agent.Agent`
    Protocol (``name``, ``select_action``, ``reset``). Stores the RNG on
    ``_rng`` so the harness's ``_reseed_agent`` hook can overwrite it for
    reproducibility.
    """

    name: str

    def __init__(
        self,
        cfg: QConfig | None = None,
        rng: np.random.Generator | None = None,
        name: str = "tabQ",
    ) -> None:
        self._cfg: QConfig = cfg if cfg is not None else QConfig()
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )
        self.name = name
        self._q: dict[StateKey, NDArray[np.float32]] = {}
        self._episode: int = 0
        self._greedy: bool = False

    # --- Config access ----------------------------------------------------

    @property
    def cfg(self) -> QConfig:
        return self._cfg

    @property
    def q_table(self) -> dict[StateKey, NDArray[np.float32]]:
        """Read-only-by-convention accessor to the underlying table."""
        return self._q

    @property
    def epsilon(self) -> float:
        """Current exploration rate given ``self._episode`` and the schedule."""
        if self._greedy:
            return 0.0
        total = max(self._cfg.total_episodes, 1)
        decay_len = max(
            int(self._cfg.eps_decay_fraction * total), 1
        )
        if self._episode >= decay_len:
            return self._cfg.eps_end
        frac = self._episode / decay_len
        return self._cfg.eps_start + frac * (self._cfg.eps_end - self._cfg.eps_start)

    def set_episode(self, ep: int) -> None:
        """Mutate the internal episode counter used by ``epsilon``."""
        self._episode = int(ep)

    def set_greedy(self, greedy: bool) -> None:
        """Toggle eval-mode (epsilon=0, no exploration, no writes)."""
        self._greedy = bool(greedy)

    # --- Agent protocol ---------------------------------------------------

    def select_action(
        self,
        state: GameState,
        player_id: int,
        time_budget_s: float | None = None,
        max_iterations: int | None = None,
    ) -> int:
        """Choose a legal action via masked ε-greedy Q-maximization."""
        del time_budget_s, max_iterations
        mask = legal_action_mask(state, player_id)
        if not bool(mask.any()):
            raise ValueError(
                f"TabularQAgent.select_action: no legal actions for player {player_id}"
            )

        if not self._greedy and self._rng.random() < self.epsilon:
            # Uniform over legal actions only.
            legal_idx = np.flatnonzero(mask)
            return int(legal_idx[self._rng.integers(legal_idx.size)])

        key = encode_state(state, player_id)
        q_vals = self._q.get(key)
        if q_vals is None:
            # Unseen state: fall back to uniform over legal actions to avoid
            # an arbitrary tie-break bias from an all-zero Q-vector.
            legal_idx = np.flatnonzero(mask)
            return int(legal_idx[self._rng.integers(legal_idx.size)])

        masked = np.where(mask, q_vals, -np.inf)
        best = masked.max()
        tied = np.flatnonzero(masked == best)
        return int(tied[self._rng.integers(tied.size)])

    def reset(self) -> None:
        """No-op: Q-table persists across games; harness-hook only."""
        return None

    # --- Q-table access ---------------------------------------------------

    def q_values(self, key: StateKey) -> NDArray[np.float32]:
        """Return a *copy* of the Q-vector for ``key``; zeros on miss."""
        entry = self._q.get(key)
        if entry is None:
            return np.zeros(_N_ACTIONS, dtype=np.float32)
        return entry.copy()

    def td_update(
        self,
        s: StateKey,
        a: int,
        r: float,
        s_next: StateKey | None,
        next_mask: NDArray[np.bool_] | None,
    ) -> None:
        """Apply one tabular TD(0) update.

        Args:
            s: state key at time t.
            a: action taken at time t.
            r: reward received over the (s, a) transition.
            s_next: state key at time t+1, or ``None`` for a terminal transition.
            next_mask: legal-action mask at ``s_next``; required when ``s_next``
                is not None so we never bootstrap from illegal-action Q-values.
        """
        if not 0 <= a < _N_ACTIONS:
            raise ValueError(f"TabularQAgent.td_update: action {a} out of range")
        entry = self._q.get(s)
        if entry is None:
            entry = np.zeros(_N_ACTIONS, dtype=np.float32)
            self._q[s] = entry

        if s_next is None:
            target = float(r)
        else:
            if next_mask is None:
                raise ValueError(
                    "TabularQAgent.td_update: next_mask must be provided when "
                    "s_next is not None"
                )
            next_entry = self._q.get(s_next)
            if next_entry is None or not bool(next_mask.any()):
                bootstrap = 0.0
            else:
                masked = np.where(next_mask, next_entry, -np.inf)
                max_val = float(masked.max())
                # If all next actions were illegal `masked.max()` would be
                # -inf — the `any()` guard above already handled that case.
                bootstrap = max_val
            target = float(r) + self._cfg.gamma * bootstrap

        entry[a] = entry[a] + self._cfg.alpha * (target - entry[a])

    # --- Persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the Q-table + config + episode counter to ``path``."""
        blob = _QState(cfg=self._cfg, q=self._q, episode=self._episode)
        with Path(path).open("wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        path: str | Path,
        rng: np.random.Generator | None = None,
        name: str = "tabQ",
    ) -> TabularQAgent:
        """Restore a :class:`TabularQAgent` from a pickle written by :meth:`save`."""
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        if not isinstance(blob, _QState):
            raise ValueError(
                f"TabularQAgent.load: expected _QState blob, got {type(blob).__name__}"
            )
        agent = cls(cfg=blob.cfg, rng=rng, name=name)
        agent._q = blob.q
        agent._episode = blob.episode
        return agent


__all__ = ["QConfig", "TabularQAgent"]
