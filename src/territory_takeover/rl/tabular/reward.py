"""Reward shaping for the tabular Q-learning baseline.

Three reward paths:

- **Per-step** — each cell claimed grants ``+per_cell_gain``. The engine
  already emits this as ``step_result.reward`` (1.0 on a claiming move, 0.0
  on traversal/illegal); we just scale it.
- **Blockout / death** — the transition in which ``PlayerState.alive`` flips
  ``True -> False`` (the player got walled out or the board filled) gets a
  one-time penalty of ``-trap_penalty_per_cell * empty_cells_remaining``:
  dying while the board is still open means opportunity was surrendered,
  while dying at a full board is free.
- **Terminal rank bonus** — at game end, each seat receives a bonus based on
  its final rank by territory count. Default ladder ``(+10, +3, -3, -10)``.
  Ties split the average of the tied-rank slots so the signal stays
  zero-sum for 2-player ties.

Each reward path is intentionally small and pure — the training loop composes
them; there's no global reward "orchestrator" to keep the code testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from territory_takeover.state import GameState


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward-shaping hyperparameters.

    ``rank_bonuses`` is indexed by 0-based rank (``rank_bonuses[0]`` = bonus
    for first place). For 2-player games only the first two entries apply;
    for 4-player games all four do.
    """

    per_cell_gain: float = 1.0
    trap_penalty_per_cell: float = 1.0
    rank_bonuses: tuple[float, float, float, float] = (10.0, 3.0, -3.0, -10.0)


def step_reward(engine_reward: float, cfg: RewardConfig) -> float:
    """Scale the engine's per-step reward by ``cfg.per_cell_gain``.

    The engine returns ``1.0`` on a claiming move and ``0.0`` on a traversal
    or illegal move. Multiplying by ``per_cell_gain`` is a no-op with the
    default ``1.0`` but lets us sweep reward scale without touching the
    training loop.
    """
    return cfg.per_cell_gain * engine_reward


def death_penalty(empty_cells_remaining: int, cfg: RewardConfig) -> float:
    """One-time penalty applied when a player is marked ``alive = False``.

    The magnitude scales with the number of EMPTY cells still on the board
    at the moment of death: being walled out early (much of the board still
    unclaimed) surrenders far more opportunity than dying at a full board,
    which is free (``0.0``).
    """
    if empty_cells_remaining <= 0:
        return 0.0
    return -cfg.trap_penalty_per_cell * float(empty_cells_remaining)


def terminal_rank_bonus(
    state: GameState,
    player_id: int,
    cfg: RewardConfig,
) -> float:
    """Rank-based terminal bonus for ``player_id`` given terminal ``state``.

    Scores players by ``territory_count``. Higher score => better
    rank (rank 0 is 1st place). Ties are resolved by averaging the rank
    slots the tied players occupy — so a 2-way tie for 1st gives each tied
    player ``(rank_bonuses[0] + rank_bonuses[1]) / 2``. This keeps the
    signal zero-sum under ``(+10, +3, -3, -10)`` in ties.
    """
    scores = [float(p.territory_count) for p in state.players]
    n = len(scores)
    if n == 0:
        return 0.0
    bonuses = cfg.rank_bonuses[:n]

    # Assign each player a rank with tie-averaging.
    # Sort player indices by descending score; group equal scores into
    # contiguous "buckets"; each player in a bucket gets the mean of the
    # bonuses that span that bucket.
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    # Assignments indexed by player_id.
    per_player: list[float] = [0.0] * n

    i = 0
    while i < n:
        # Extend the tie block while the next player has the same score.
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # Ranks [i..j] are tied; average their bonuses.
        mean_bonus = sum(bonuses[i : j + 1]) / (j - i + 1)
        for k in range(i, j + 1):
            per_player[order[k]] = mean_bonus
        i = j + 1

    return per_player[player_id]


__all__ = [
    "RewardConfig",
    "death_penalty",
    "step_reward",
    "terminal_rank_bonus",
]
