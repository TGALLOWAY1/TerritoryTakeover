"""Game-driving utilities for search agents.

:func:`play_game` runs a full game between a list of :class:`Agent`
instances (one per seat), returning the terminal :class:`GameState`.
:func:`tournament` plays ``num_games`` alternating-seat games between two
agents and returns win counts. Both functions derive per-game seeds from a
single :class:`numpy.random.SeedSequence` so a whole tournament is
reproducible from one integer.

:func:`run_match` and :func:`round_robin` are the Phase-2 evaluation
driver. They instrument each ``select_action`` call with wall-clock time
and agent-reported iteration counts, rotate seat assignments to
neutralize spawn-position advantages, and optionally dispatch games
across a :class:`multiprocessing.Pool`. Wilson 95% confidence intervals
on pair win-rates are computed in-process (no scipy dependency).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

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


# --- Phase-2 evaluation driver ---------------------------------------------


@dataclass(frozen=True)
class GameLog:
    """One full game's worth of post-hoc-analyzable data.

    ``seat_assignment`` holds the *agent name* in each seat; it is the only
    link back to the agent list after the game is flattened to CSV.
    ``actions`` is a list of ``(seat_id, action_int)`` tuples in move order.
    ``decision_times_s[seat]`` and ``iterations[seat]`` are parallel lists
    (one entry per turn that seat actually moved). ``iterations`` is 0 for
    agents that do not expose a counter; aggregate stats treat those as
    unknown and emit NaN rather than a misleading zero.
    """

    game_index: int
    game_seed: int
    rotation_offset: int
    seat_assignment: list[str]
    actions: list[tuple[int, int]]
    final_scores: list[int]
    winner_seat: int | None
    decision_times_s: list[list[float]]
    iterations: list[list[int]]


@dataclass(frozen=True)
class AgentStats:
    """Aggregated per-agent statistics over a match or round-robin."""

    name: str
    games: int
    wins: int
    ties: int
    losses: int
    avg_territory: float
    avg_decision_time_s: float
    avg_iters_per_s: float
    n_decisions: int


@dataclass(frozen=True)
class MatchResult:
    """Return value of :func:`run_match`."""

    agent_names: list[str]
    num_games: int
    board_size: int
    per_agent: list[AgentStats]
    games: list[GameLog]


@dataclass(frozen=True)
class PairRow:
    """One row of a :class:`Table` — head-to-head stats for an agent pair."""

    agent_a: str
    agent_b: str
    games: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class Table:
    """Return value of :func:`round_robin`."""

    rows: list[PairRow]
    per_agent: list[AgentStats]
    games: list[GameLog] = field(default_factory=list)


# --- Wilson 95% CI ---------------------------------------------------------


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` trials.

    Returns ``(low, high)`` for a 95% confidence level by default. For
    ``n == 0`` returns the uninformative ``(0.0, 1.0)``.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (center - half, center + half)


# --- Seat rotation ---------------------------------------------------------


def _seat_assignment(num_agents: int, game_index: int, swap_seats: bool) -> list[int]:
    """Return a list ``seat_to_agent[seat]`` for game ``game_index``.

    Cyclic single-shift rotation: ``seat k`` gets ``agent (k + game_index)
    mod num_agents`` when ``swap_seats`` is True. With
    ``num_games % num_agents == 0`` every agent plays every seat equally.
    """
    if not swap_seats:
        return list(range(num_agents))
    return [(k + game_index) % num_agents for k in range(num_agents)]


# --- Per-agent RNG re-seeding ----------------------------------------------


def _reseed_agent(agent: Agent, seed: int) -> None:
    """Overwrite ``agent._rng`` with a fresh seeded ``Generator`` if present.

    All in-tree agents that carry randomness store the generator on
    ``_rng``. This helper is best-effort: agents without that attribute are
    left alone, which is correct for deterministic agents (e.g. greedy with
    no ties to break beyond RNG is still deterministic once seeded).
    """
    if hasattr(agent, "_rng"):
        agent._rng = np.random.default_rng(seed)


# --- Running a single game -------------------------------------------------


def _run_one_game(
    agents_pickled: bytes,
    seat_to_agent: list[int],
    agent_names: list[str],
    game_index: int,
    game_seed: int,
    agent_seeds: list[int],
    rotation_offset: int,
    board_size: int,
    num_players: int,
) -> GameLog:
    """Run one full game in the current process and return a :class:`GameLog`.

    ``agents_pickled`` carries fresh copies of the original agent list so
    that worker processes (and serial calls that want full isolation
    between games) do not share mutable agent state. Each seat is assigned
    ``agents[seat_to_agent[seat]]``. Per-agent RNG is re-seeded from the
    per-game ``SeedSequence`` before the game starts.
    """
    agents_list: list[Agent] = pickle.loads(agents_pickled)

    seat_agents: list[Agent] = [agents_list[seat_to_agent[s]] for s in range(num_players)]
    for s, agent in enumerate(seat_agents):
        agent.reset()
        _reseed_agent(agent, agent_seeds[s])

    state = new_game(board_size=board_size, num_players=num_players, seed=game_seed)

    actions: list[tuple[int, int]] = []
    decision_times_s: list[list[float]] = [[] for _ in range(num_players)]
    iterations: list[list[int]] = [[] for _ in range(num_players)]

    while not state.done:
        seat = state.current_player
        agent = seat_agents[seat]

        t0 = time.perf_counter()
        action = agent.select_action(state, seat)
        elapsed = time.perf_counter() - t0

        stats = getattr(agent, "last_search_stats", None)
        iters = 0
        if isinstance(stats, dict):
            iters = int(stats.get("iterations", 0) or 0)
        if iters == 0:
            nodes = getattr(agent, "last_nodes", None)
            if nodes is not None:
                iters = int(nodes)

        decision_times_s[seat].append(elapsed)
        iterations[seat].append(iters)
        actions.append((seat, int(action)))

        step(state, action, strict=True)

    final_scores = [len(p.path) + p.claimed_count for p in state.players]

    return GameLog(
        game_index=game_index,
        game_seed=game_seed,
        rotation_offset=rotation_offset,
        seat_assignment=[agent_names[seat_to_agent[s]] for s in range(num_players)],
        actions=actions,
        final_scores=final_scores,
        winner_seat=state.winner,
        decision_times_s=decision_times_s,
        iterations=iterations,
    )


def _worker(args: tuple[Any, ...]) -> GameLog:
    """Top-level function for :class:`multiprocessing.Pool` dispatch.

    Must be top-level (picklable) and accept a single positional arg.
    """
    return _run_one_game(*args)


# --- Aggregation -----------------------------------------------------------


def _aggregate(
    agent_names: list[str],
    games: list[GameLog],
) -> list[AgentStats]:
    """Fold a list of :class:`GameLog` into one :class:`AgentStats` per agent."""
    n_agents = len(agent_names)
    wins = [0] * n_agents
    ties = [0] * n_agents
    losses = [0] * n_agents
    games_played = [0] * n_agents
    total_territory = [0.0] * n_agents
    total_time_s = [0.0] * n_agents
    total_iters = [0] * n_agents
    total_iter_time_s = [0.0] * n_agents
    n_decisions = [0] * n_agents
    n_iter_decisions = [0] * n_agents

    name_to_index = {name: i for i, name in enumerate(agent_names)}

    for game in games:
        for seat, agent_name in enumerate(game.seat_assignment):
            ai = name_to_index[agent_name]
            games_played[ai] += 1
            total_territory[ai] += float(game.final_scores[seat])
            if game.winner_seat is None:
                ties[ai] += 1
            elif game.winner_seat == seat:
                wins[ai] += 1
            else:
                losses[ai] += 1

            for t, it in zip(game.decision_times_s[seat], game.iterations[seat], strict=True):
                total_time_s[ai] += t
                n_decisions[ai] += 1
                if it > 0:
                    total_iters[ai] += it
                    total_iter_time_s[ai] += t
                    n_iter_decisions[ai] += 1

    per_agent: list[AgentStats] = []
    for i, name in enumerate(agent_names):
        g = games_played[i]
        avg_territory = total_territory[i] / g if g > 0 else 0.0
        avg_time = total_time_s[i] / n_decisions[i] if n_decisions[i] > 0 else 0.0
        if n_iter_decisions[i] > 0 and total_iter_time_s[i] > 0:
            avg_iters_per_s = total_iters[i] / total_iter_time_s[i]
        else:
            avg_iters_per_s = float("nan")
        per_agent.append(
            AgentStats(
                name=name,
                games=g,
                wins=wins[i],
                ties=ties[i],
                losses=losses[i],
                avg_territory=avg_territory,
                avg_decision_time_s=avg_time,
                avg_iters_per_s=avg_iters_per_s,
                n_decisions=n_decisions[i],
            )
        )
    return per_agent


# --- run_match -------------------------------------------------------------


def run_match(
    agents: list[Agent],
    num_games: int,
    board_size: int,
    swap_seats: bool = True,
    seed: int = 0,
    parallel: bool = False,
    num_players: int | None = None,
) -> MatchResult:
    """Run ``num_games`` games with the given agent line-up.

    Seat assignment is either fixed (``swap_seats=False``) or cyclically
    rotated by one seat per game (``swap_seats=True``). With
    ``swap_seats=True`` and ``num_games`` a multiple of ``len(agents)``,
    every agent plays from every starting corner an equal number of times.

    Per-game seeds come from :class:`numpy.random.SeedSequence` spawned
    from ``seed``; each spawned sequence further spawns one child per seat
    which is used to reset the corresponding agent's ``_rng`` before the
    game starts. Because the spawn tree is deterministic, serial and
    parallel runs from the same ``seed`` produce bit-identical game logs.

    ``parallel=True`` dispatches games across a
    :class:`multiprocessing.Pool` sized at ``os.cpu_count()``. Workers
    receive pickled copies of ``agents`` so nothing but plain data crosses
    the boundary.
    """
    if num_games < 0:
        raise ValueError(f"num_games must be >= 0; got {num_games}")
    if not agents:
        raise ValueError("agents must be non-empty")

    n_agents = len(agents)
    players = num_players if num_players is not None else n_agents

    if num_players is None and swap_seats and num_games > 0 and num_games % n_agents != 0:
        raise ValueError(
            f"swap_seats=True requires num_games ({num_games}) to be a multiple of "
            f"len(agents) ({n_agents}) so every agent plays every seat equally"
        )

    if players != n_agents and swap_seats and num_games > 0:
        # Seat rotation is not defined when seat count differs from agent count.
        raise ValueError(
            "swap_seats=True is only supported when len(agents) == num_players"
        )

    agent_names = [a.name for a in agents]

    root_ss = np.random.SeedSequence(seed)
    game_seqs = root_ss.spawn(max(num_games, 1))

    agents_pickled = pickle.dumps(agents)

    job_args: list[tuple[Any, ...]] = []
    for i in range(num_games):
        game_ss = game_seqs[i]
        # One child seed for new_game(), then one per seat for agent RNGs.
        child_seqs = game_ss.spawn(1 + players)
        game_seed = int(child_seqs[0].generate_state(1, dtype=np.uint32)[0])
        agent_seeds = [
            int(child_seqs[1 + s].generate_state(1, dtype=np.uint32)[0])
            for s in range(players)
        ]
        rotation_offset = i % n_agents if swap_seats else 0
        seat_to_agent = _seat_assignment(n_agents, i, swap_seats)
        if players != n_agents:
            # Broadcast the first agent across extra seats — only reached when
            # caller explicitly opts in with num_players > len(agents).
            seat_to_agent = (seat_to_agent + [0] * players)[:players]
        job_args.append(
            (
                agents_pickled,
                seat_to_agent,
                agent_names,
                i,
                game_seed,
                agent_seeds,
                rotation_offset,
                board_size,
                players,
            )
        )

    games: list[GameLog]
    if parallel and num_games > 1:
        with mp.Pool() as pool:
            games = list(pool.imap(_worker, job_args))
    else:
        games = [_worker(args) for args in job_args]

    per_agent = _aggregate(agent_names, games)

    return MatchResult(
        agent_names=agent_names,
        num_games=num_games,
        board_size=board_size,
        per_agent=per_agent,
        games=games,
    )


# --- round_robin -----------------------------------------------------------


def round_robin(
    agents: list[Agent],
    games_per_pair: int,
    board_size: int,
    seed: int = 0,
    parallel: bool = False,
    num_players: int = 4,
) -> Table:
    """Play every unordered pair of agents in a 2-vs-2 seat layout.

    For each pair ``(A, B)`` two turn-alternating seat layouts are used —
    ``[A, B, A, B]`` and ``[B, A, B, A]`` — each for ``games_per_pair / 2``
    games. This avoids the adjacent-vs-diagonal geometric asymmetry you'd
    get from ``[A, A, B, B]``. Per-pair Wilson 95% CIs on ``wins_a / games``
    are reported.

    Only 4-player layouts are supported (the 2-vs-2 mix is meaningless for
    other player counts); pass ``num_players=4`` explicitly to make the
    intent loud in call sites.
    """
    if games_per_pair < 0:
        raise ValueError(f"games_per_pair must be >= 0; got {games_per_pair}")
    if games_per_pair % 2 != 0:
        raise ValueError(
            f"games_per_pair must be even so the two 2v2 layouts split cleanly; "
            f"got {games_per_pair}"
        )
    if num_players != 4:
        raise ValueError(
            f"round_robin is defined for 4-player games only; got num_players={num_players}"
        )
    if len(agents) < 2:
        raise ValueError("round_robin needs at least 2 agents")

    half = games_per_pair // 2
    rows: list[PairRow] = []
    all_games: list[GameLog] = []

    root_ss = np.random.SeedSequence(seed)
    pair_seqs = root_ss.spawn(len(list(combinations(range(len(agents)), 2))))

    for pair_idx, (ai, bi) in enumerate(combinations(range(len(agents)), 2)):
        agent_a = agents[ai]
        agent_b = agents[bi]
        pair_seed = int(pair_seqs[pair_idx].generate_state(1, dtype=np.uint32)[0])

        # Layout 1: [A, B, A, B]
        seats_1: list[Agent] = [agent_a, agent_b, agent_a, agent_b]
        m1 = run_match(
            seats_1,
            num_games=half,
            board_size=board_size,
            swap_seats=False,
            seed=pair_seed,
            parallel=parallel,
        )
        # Layout 2: [B, A, B, A]
        seats_2: list[Agent] = [agent_b, agent_a, agent_b, agent_a]
        m2 = run_match(
            seats_2,
            num_games=half,
            board_size=board_size,
            swap_seats=False,
            seed=pair_seed ^ 0xA5A5A5A5,
            parallel=parallel,
        )

        wins_a = 0
        wins_b = 0
        ties = 0
        # Layout 1: seats (0, 2) are A, seats (1, 3) are B.
        for g in m1.games:
            if g.winner_seat is None:
                ties += 1
            elif g.winner_seat in (0, 2):
                wins_a += 1
            else:
                wins_b += 1
        # Layout 2: seats (1, 3) are A, seats (0, 2) are B.
        for g in m2.games:
            if g.winner_seat is None:
                ties += 1
            elif g.winner_seat in (1, 3):
                wins_a += 1
            else:
                wins_b += 1

        games_total = games_per_pair
        win_rate_a = wins_a / games_total if games_total > 0 else 0.0
        ci_low, ci_high = _wilson_ci(wins_a, games_total)
        rows.append(
            PairRow(
                agent_a=agent_a.name,
                agent_b=agent_b.name,
                games=games_total,
                wins_a=wins_a,
                wins_b=wins_b,
                ties=ties,
                win_rate_a=win_rate_a,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
        all_games.extend(m1.games)
        all_games.extend(m2.games)

    per_agent = _aggregate([a.name for a in agents], all_games)

    return Table(rows=rows, per_agent=per_agent, games=all_games)


__all__ = [
    "AgentStats",
    "GameLog",
    "MatchResult",
    "PairRow",
    "Table",
    "play_game",
    "round_robin",
    "run_match",
    "tournament",
]
