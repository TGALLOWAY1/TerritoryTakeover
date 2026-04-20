"""Tests for the pairwise Bradley–Terry Elo estimator."""

from __future__ import annotations

import math

import pytest

from territory_takeover.rl.eval.elo import (
    GameOutcome,
    compute_elo,
    outcomes_from_rank,
)


def test_outcomes_from_rank_produces_pairwise_pairs() -> None:
    outcomes = outcomes_from_rank(["A", "B", "C", "D"], [1, 2, 3, 4])
    assert len(outcomes) == 6

    # A beats everyone.
    a_outcomes = [o for o in outcomes if o.agent_a == "A"]
    assert len(a_outcomes) == 3
    assert all(o.score_a == 1.0 for o in a_outcomes)


def test_outcomes_from_rank_handles_ties() -> None:
    outcomes = outcomes_from_rank(["A", "B"], [1, 1])
    assert len(outcomes) == 1
    assert outcomes[0].score_a == 0.5


def test_anchor_is_pinned_to_zero() -> None:
    # Simple 3-agent round robin where A > B > C.
    outcomes: list[GameOutcome] = []
    for _ in range(20):
        outcomes.extend(outcomes_from_rank(["A", "B", "C"], [1, 2, 3]))
    ratings = compute_elo(outcomes, anchor="B")
    assert ratings["B"] == pytest.approx(0.0, abs=1e-9)
    assert ratings["A"] > 0.0
    assert ratings["C"] < 0.0


def test_stronger_agent_gets_higher_rating() -> None:
    # A wins 70% of games against B; B wins 70% of games against C.
    outcomes: list[GameOutcome] = []
    for _ in range(70):
        outcomes.append(GameOutcome("A", "B", 1.0))
        outcomes.append(GameOutcome("B", "C", 1.0))
    for _ in range(30):
        outcomes.append(GameOutcome("A", "B", 0.0))
        outcomes.append(GameOutcome("B", "C", 0.0))

    ratings = compute_elo(outcomes, anchor="C")
    assert ratings["C"] == pytest.approx(0.0, abs=1e-9)
    assert ratings["A"] > ratings["B"] > ratings["C"]


def test_bt_approximates_expected_elo_gap() -> None:
    # Canonical BT check: agent X beats agent Y exactly 75% of the time in
    # a lot of games. Expected Elo gap is 400 * log10(0.75 / 0.25) ~= 190.8.
    outcomes: list[GameOutcome] = []
    for _ in range(750):
        outcomes.append(GameOutcome("X", "Y", 1.0))
    for _ in range(250):
        outcomes.append(GameOutcome("X", "Y", 0.0))

    ratings = compute_elo(outcomes, anchor="Y")
    expected_gap = 400.0 * math.log10(0.75 / 0.25)
    assert ratings["X"] == pytest.approx(expected_gap, rel=0.02)


def test_ties_count_as_half_win_each_side() -> None:
    # All games are ties: A and B should have identical ratings.
    outcomes = [GameOutcome("A", "B", 0.5) for _ in range(100)]
    ratings = compute_elo(outcomes, anchor="A")
    assert ratings["A"] == pytest.approx(0.0, abs=1e-9)
    assert ratings["B"] == pytest.approx(0.0, abs=1e-6)


def test_compute_elo_rejects_unknown_anchor() -> None:
    outcomes = [GameOutcome("A", "B", 1.0)]
    with pytest.raises(ValueError):
        compute_elo(outcomes, anchor="Z")


def test_compute_elo_rejects_empty() -> None:
    with pytest.raises(ValueError):
        compute_elo([], anchor="A")


def test_four_player_round_robin() -> None:
    # 4-player outcomes derived from a strict ranking; iterate several
    # games to stabilize.
    agents = ("A", "B", "C", "D")
    outcomes: list[GameOutcome] = []
    for _ in range(30):
        outcomes.extend(outcomes_from_rank(agents, [1, 2, 3, 4]))
    ratings = compute_elo(outcomes, anchor="A")
    # A is always best → all others should have negative Elo vs A.
    assert ratings["A"] == pytest.approx(0.0, abs=1e-9)
    assert ratings["B"] < 0 and ratings["C"] < ratings["B"] and ratings["D"] < ratings["C"]
