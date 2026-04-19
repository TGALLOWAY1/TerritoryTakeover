"""Tests for the Phase-2 evaluation harness (run_match, round_robin)."""

from __future__ import annotations

import time

from territory_takeover.search import UniformRandomAgent
from territory_takeover.search.agent import Agent
from territory_takeover.search.harness import _wilson_ci, round_robin, run_match


def _random_agents(names: list[str]) -> list[Agent]:
    return [UniformRandomAgent(name=n) for n in names]


def test_run_match_four_way_8_games_under_two_minutes() -> None:
    agents = _random_agents(["r0", "r1", "r2", "r3"])
    t0 = time.perf_counter()
    result = run_match(
        agents=agents,
        num_games=8,
        board_size=20,
        swap_seats=True,
        seed=0,
        parallel=False,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 120.0, f"4-way 8-game match took {elapsed:.1f}s, expected < 120s"
    assert result.num_games == 8
    assert len(result.games) == 8
    assert result.agent_names == ["r0", "r1", "r2", "r3"]
    assert len(result.per_agent) == 4

    # Total wins + ties (summed across seats) = num_games * num_players.
    total_wtl = sum(s.wins + s.ties + s.losses for s in result.per_agent)
    assert total_wtl == 8 * 4, f"wins+ties+losses total {total_wtl} != 32"

    # Each game has num_players seats filled in.
    for i, game in enumerate(result.games):
        assert len(game.seat_assignment) == 4, f"game {i}: {game.seat_assignment!r}"
        assert len(game.final_scores) == 4


def test_run_match_seat_rotation_hand_check() -> None:
    """With 4 uniquely-named agents across 4 games, each agent visits each seat once."""
    agents = _random_agents(["a0", "a1", "a2", "a3"])
    result = run_match(
        agents=agents,
        num_games=4,
        board_size=12,
        swap_seats=True,
        seed=1,
        parallel=False,
    )

    seen: dict[str, set[int]] = {name: set() for name in ["a0", "a1", "a2", "a3"]}
    for i, game in enumerate(result.games):
        for seat, name in enumerate(game.seat_assignment):
            seen[name].add(seat)
        assert game.rotation_offset == i, (
            f"game {i} rotation_offset={game.rotation_offset} != {i}"
        )

    for name in ["a0", "a1", "a2", "a3"]:
        assert seen[name] == {0, 1, 2, 3}, (
            f"agent {name} visited seats {seen[name]}, expected all four"
        )


def test_run_match_parallel_equals_serial() -> None:
    """Parallel and serial runs with the same seed yield identical per-game logs."""
    for trial_seed in (7, 11):
        agents_serial = _random_agents(["a", "b", "c", "d"])
        agents_parallel = _random_agents(["a", "b", "c", "d"])

        r_serial = run_match(
            agents=agents_serial,
            num_games=4,
            board_size=12,
            swap_seats=True,
            seed=trial_seed,
            parallel=False,
        )
        r_parallel = run_match(
            agents=agents_parallel,
            num_games=4,
            board_size=12,
            swap_seats=True,
            seed=trial_seed,
            parallel=True,
        )

        assert len(r_serial.games) == len(r_parallel.games) == 4
        for i in range(4):
            gs = r_serial.games[i]
            gp = r_parallel.games[i]
            msg = f"trial_seed={trial_seed}, game={i}"
            assert gs.actions == gp.actions, msg
            assert gs.final_scores == gp.final_scores, msg
            assert gs.winner_seat == gp.winner_seat, msg
            assert gs.seat_assignment == gp.seat_assignment, msg


def test_run_match_rejects_unbalanced_swap_seats() -> None:
    agents = _random_agents(["a", "b", "c"])
    import pytest

    with pytest.raises(ValueError, match="multiple"):
        run_match(agents=agents, num_games=4, board_size=10, swap_seats=True, seed=0)


def test_round_robin_wilson_ci_smoke() -> None:
    agents = _random_agents(["x", "y"])
    table = round_robin(
        agents=agents,
        games_per_pair=4,
        board_size=10,
        seed=3,
        parallel=False,
    )
    assert len(table.rows) == 1
    row = table.rows[0]
    assert row.games == 4
    assert 0.0 <= row.ci_low <= row.win_rate_a <= row.ci_high <= 1.0
    assert row.wins_a + row.wins_b + row.ties == 4


def test_wilson_ci_closed_form() -> None:
    # Known reference: 50 successes in 100 trials -> 95% Wilson CI ~ [0.404, 0.596].
    low, high = _wilson_ci(50, 100)
    assert abs(low - 0.4038) < 1e-3, f"low={low}"
    assert abs(high - 0.5962) < 1e-3, f"high={high}"
    assert abs((low + high) / 2 - 0.5) < 1e-9  # symmetric around 0.5 for k=n/2

    # Asymmetry: k=90, n=100 -> mean < 0.9, skewed toward 0.5.
    low_skew, high_skew = _wilson_ci(90, 100)
    mid = (low_skew + high_skew) / 2
    assert mid < 0.9

    # n=0 returns uninformative bounds.
    low0, high0 = _wilson_ci(0, 0)
    assert (low0, high0) == (0.0, 1.0)

    # k=0, n=10 should have low=0 and high small but > 0.
    low_zero, high_zero = _wilson_ci(0, 10)
    assert low_zero < 1e-9
    assert 0.0 < high_zero < 0.5
