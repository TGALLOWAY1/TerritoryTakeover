"""Game-driving utilities for search agents.

:func:`play_game` runs a full game between a list of :class:`Agent`
instances (one per seat), returning the terminal :class:`GameState`.
:func:`tournament` plays ``num_games`` alternating-seat games between two
agents and returns win counts. Both functions derive per-game seeds from a
single :class:`numpy.random.SeedSequence` so a whole tournament is
reproducible from one integer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.engine import new_game, step

if TYPE_CHECKING:
    from collections.abc import Sequence

    from territory_takeover.state import GameState

    from .agent import Agent


def play_game(
    agents: Sequence[Agent],
    board_size: int,
    num_players: int,
    seed: int | None = None,
    max_turns: int | None = None,
) -> GameState:
    """Run a full game and return the terminal :class:`GameState`.

    ``agents`` must have length ``num_players`` — ``agents[i]`` controls seat
    ``i``. Each agent has :meth:`Agent.reset` called before the first move.
    If ``max_turns`` is set and the turn counter reaches it before the game
    ends naturally, the loop returns the current state as-is (useful as a
    safety net in tests).
    """
    if len(agents) != num_players:
        raise ValueError(
            f"agents length {len(agents)} != num_players {num_players}"
        )

    for agent in agents:
        agent.reset()

    state = new_game(board_size=board_size, num_players=num_players, seed=seed)

    while not state.done:
        if max_turns is not None and state.turn_number >= max_turns:
            break
        pid = state.current_player
        action = agents[pid].select_action(state, pid)
        step(state, action, strict=True)

    return state


def tournament(
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
    board_size: int,
    seed: int | None = None,
) -> dict[str, int]:
    """Run an alternating-seat head-to-head tournament of two agents.

    Seats are alternated game-to-game so spawn / first-move effects average
    out: in even-indexed games ``agent_a`` is player 0, in odd-indexed games
    ``agent_b`` is player 0. Per-game seeds are derived from ``seed`` via
    :class:`numpy.random.SeedSequence` so the full run is reproducible.

    Returns a dict with keys ``wins_a``, ``wins_b``, ``ties``.
    """
    if num_games < 0:
        raise ValueError(f"num_games must be >= 0; got {num_games}")

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.generate_state(max(num_games, 1), dtype=np.uint32)

    wins_a = 0
    wins_b = 0
    ties = 0
    for i in range(num_games):
        game_seed = int(child_seeds[i])
        seats: list[Agent]
        if i % 2 == 0:
            seats = [agent_a, agent_b]
            a_seat = 0
        else:
            seats = [agent_b, agent_a]
            a_seat = 1
        terminal = play_game(seats, board_size=board_size, num_players=2, seed=game_seed)
        winner = terminal.winner
        if winner is None:
            ties += 1
        elif winner == a_seat:
            wins_a += 1
        else:
            wins_b += 1

    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties}


__all__ = ["play_game", "tournament"]
