"""Local 2-player evaluation driver with explicit spawn passthrough.

``territory_takeover.search.harness.tournament`` does not plumb
``spawn_positions`` through to :func:`new_game`, which is a problem on 8x8
/ 2p where the default spawn insets collide. This module duplicates the
minimal alternating-seat tournament pattern but forwards spawns.

For 4p diagnostics we use the default spawns (which are well-defined, if
center-clustered) and run :func:`run_match_4p` to get per-seat win rates.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.engine import new_game, step

if TYPE_CHECKING:
    from territory_takeover.search.agent import Agent


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (center - half, center + half)


def _play_game_with_spawns(
    agents: list[Agent],
    board_size: int,
    num_players: int,
    spawn_positions: list[tuple[int, int]] | None,
    seed: int | None,
) -> int | None:
    for agent in agents:
        agent.reset()
    state = new_game(
        board_size=board_size,
        num_players=num_players,
        spawn_positions=spawn_positions,
        seed=seed,
    )
    while not state.done:
        pid = state.current_player
        action = agents[pid].select_action(state, pid)
        step(state, action, strict=True)
    return state.winner


def evaluate_vs(
    agent: Agent,
    opponent: Agent,
    num_games: int,
    board_size: int,
    spawn_positions: list[tuple[int, int]] | None,
    seed: int,
) -> dict[str, float]:
    """Alternating-seat 2-player head-to-head.

    Returns a dict with ``win_rate``, ``wins``, ``losses``, ``ties``,
    ``ci_low``, ``ci_high``, ``games``. Seats are alternated so spawn
    position effects wash out.
    """
    if num_games < 0:
        raise ValueError(f"num_games must be >= 0; got {num_games}")

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.generate_state(max(num_games, 1), dtype=np.uint32)

    wins = 0
    losses = 0
    ties = 0
    for i in range(num_games):
        game_seed = int(child_seeds[i])
        if i % 2 == 0:
            seats: list[Agent] = [agent, opponent]
            agent_seat = 0
        else:
            seats = [opponent, agent]
            agent_seat = 1
        winner = _play_game_with_spawns(
            seats, board_size, 2, spawn_positions, game_seed
        )
        if winner is None:
            ties += 1
        elif winner == agent_seat:
            wins += 1
        else:
            losses += 1

    wr = wins / num_games if num_games > 0 else 0.0
    low, high = _wilson_ci(wins, num_games)
    return {
        "games": float(num_games),
        "wins": float(wins),
        "losses": float(losses),
        "ties": float(ties),
        "win_rate": wr,
        "ci_low": low,
        "ci_high": high,
    }


def evaluate_vs_4p(
    agent: Agent,
    opponent: Agent,
    num_games: int,
    board_size: int,
    spawn_positions: list[tuple[int, int]] | None,
    seed: int,
) -> dict[str, float]:
    """4-player evaluation: one ``agent`` seat, three ``opponent`` seats.

    Rotates the agent seat so every starting corner is tested equally. The
    reported ``win_rate`` is the fraction of games the agent wins outright
    (no partial credit for ties).
    """
    if num_games < 0:
        raise ValueError(f"num_games must be >= 0; got {num_games}")

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.generate_state(max(num_games, 1), dtype=np.uint32)

    wins = 0
    losses = 0
    ties = 0
    for i in range(num_games):
        game_seed = int(child_seeds[i])
        agent_seat = i % 4
        seats: list[Agent] = [opponent, opponent, opponent, opponent]
        seats[agent_seat] = agent
        winner = _play_game_with_spawns(
            seats, board_size, 4, spawn_positions, game_seed
        )
        if winner is None:
            ties += 1
        elif winner == agent_seat:
            wins += 1
        else:
            losses += 1

    wr = wins / num_games if num_games > 0 else 0.0
    low, high = _wilson_ci(wins, num_games)
    return {
        "games": float(num_games),
        "wins": float(wins),
        "losses": float(losses),
        "ties": float(ties),
        "win_rate": wr,
        "ci_low": low,
        "ci_high": high,
    }


__all__ = ["evaluate_vs", "evaluate_vs_4p"]
