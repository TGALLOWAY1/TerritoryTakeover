"""Tests for non-uniform MCTS rollout policies.

Unit tests cover termination, return-value shape/range, the
``make_rollout`` factory dispatch, and ``UCTAgent``'s ``rollout_kind``
plumbing. They all complete in a few seconds.

Two matchup tests validate that the heuristics actually help:

- :func:`test_informed_rollout_beats_uniform_at_same_iters` — 100 games,
  10x10, 200 iterations per move. Informed must win >=60% of decided
  games. Expected wall time ~5 minutes. This is the real validation for
  the ``informed`` policy; if it fails, the heuristic or its per-move
  budget needs diagnosing before the change is worth shipping.
- :func:`test_voronoi_guided_beats_informed_long_horizon` — 200 games,
  15x15, 2000 iterations per move. Very expensive (several hours) and
  gated behind the ``TERRITORY_RUN_LONG_ROLLOUT_MATCHUP`` env var.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from territory_takeover.engine import new_game
from territory_takeover.search import (
    UCTAgent,
    play_game,
    tournament,
)
from territory_takeover.search.agent import Agent
from territory_takeover.search.mcts.rollout import (
    informed_rollout,
    make_rollout,
    uniform_rollout,
    voronoi_guided_rollout,
)


def test_informed_rollout_terminates_and_returns_normalized_vector() -> None:
    for i in range(5):
        rng = np.random.default_rng(1000 + i)
        state = new_game(board_size=10, num_players=2, seed=1000 + i)
        value = informed_rollout(state, rng)
        assert state.done, f"trial={i} rollout did not terminate"
        assert value.shape == (2,), f"trial={i} wrong shape {value.shape}"
        assert value.dtype == np.float64, f"trial={i} wrong dtype {value.dtype}"
        for p in range(2):
            assert 0.0 <= value[p] <= 1.0, (
                f"trial={i} player={p} value out of range: {value[p]}"
            )


def test_voronoi_guided_rollout_terminates_and_returns_normalized_vector() -> None:
    for i in range(3):
        rng = np.random.default_rng(2000 + i)
        state = new_game(board_size=10, num_players=2, seed=2000 + i)
        value = voronoi_guided_rollout(state, rng)
        assert state.done, f"trial={i} rollout did not terminate"
        assert value.shape == (2,), f"trial={i} wrong shape {value.shape}"
        assert value.dtype == np.float64, f"trial={i} wrong dtype {value.dtype}"
        for p in range(2):
            assert 0.0 <= value[p] <= 1.0, (
                f"trial={i} player={p} value out of range: {value[p]}"
            )


def test_informed_rollout_respects_epsilon_one_is_uniform_in_distribution() -> None:
    """With epsilon=1.0 the policy degenerates to uniform-random.

    Not a statistical equivalence test — just a sanity check that the
    ``epsilon=1.0`` branch runs end-to-end and produces a valid terminal
    value (i.e. the code path with no softmax scoring works).
    """
    rng = np.random.default_rng(3001)
    state = new_game(board_size=8, num_players=2, seed=3001)
    value = informed_rollout(state, rng, epsilon=1.0)
    assert state.done
    assert value.shape == (2,)


def test_make_rollout_dispatch_returns_expected_callables() -> None:
    assert make_rollout("uniform") is uniform_rollout
    assert make_rollout("informed") is informed_rollout
    assert make_rollout("voronoi_guided") is voronoi_guided_rollout


def test_make_rollout_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown rollout kind"):
        make_rollout("not-a-real-policy")


def test_voronoi_guided_rollout_rejects_bad_k() -> None:
    rng = np.random.default_rng(0)
    state = new_game(board_size=8, num_players=2, seed=0)
    with pytest.raises(ValueError, match="k must be"):
        voronoi_guided_rollout(state, rng, k=0)


def test_uct_agent_accepts_rollout_kind() -> None:
    """UCTAgent(rollout_kind="informed") plays a short 8x8 game to completion."""
    agents: list[Agent] = [
        UCTAgent(
            iterations=16,
            rollout_kind="informed",
            rng=np.random.default_rng(4000),
        ),
        UCTAgent(
            iterations=16,
            rollout_kind="informed",
            rng=np.random.default_rng(4001),
        ),
    ]
    terminal = play_game(agents, board_size=8, num_players=2, seed=4000, max_turns=5_000)
    assert terminal.done


def test_uct_agent_rejects_both_rollout_fn_and_rollout_kind() -> None:
    with pytest.raises(ValueError, match="rollout_fn or rollout_kind"):
        UCTAgent(
            iterations=16,
            rollout_fn=uniform_rollout,
            rollout_kind="informed",
        )


def test_uct_agent_rollout_kind_matches_explicit_fn() -> None:
    """``rollout_kind="uniform"`` must wire up the same callable as the default."""
    agent = UCTAgent(
        iterations=8,
        rollout_kind="uniform",
        rng=np.random.default_rng(5000),
    )
    assert agent._rollout_fn is uniform_rollout


def test_informed_rollout_beats_uniform_at_same_iters() -> None:
    """Informed rollout beats uniform at the same iteration budget.

    Spec acceptance criterion: with both agents at 200 iterations per move
    on a 10x10 board over 100 games (alternating seats for fairness),
    the informed-rollout UCT must win >=60% of the decided games.
    100 games is enough that a 60-40 split rejects the null hypothesis
    p=0.5 at p<0.03 (binomial). Expected wall time ~5 minutes.
    """
    informed = UCTAgent(
        iterations=200,
        rollout_kind="informed",
        rng=np.random.default_rng(7001),
        name="uct-informed",
    )
    uniform = UCTAgent(
        iterations=200,
        rollout_kind="uniform",
        rng=np.random.default_rng(7002),
        name="uct-uniform",
    )
    results = tournament(
        agent_a=informed,
        agent_b=uniform,
        num_games=100,
        board_size=10,
        seed=2026,
    )
    decided = results["wins_a"] + results["wins_b"]
    assert decided > 0, "no decided games"
    informed_share = results["wins_a"] / decided
    assert informed_share >= 0.60, (
        f"informed rollout failed to beat uniform at matched iteration "
        f"budget: {results}, informed_share={informed_share:.2%}. "
        "Either the heuristic is not actually helping, or per-move scoring "
        "is too slow to fit the 10us budget."
    )


def test_voronoi_guided_beats_informed_long_horizon() -> None:
    """Voronoi-guided rollout outperforms informed on long horizons.

    Very expensive (~hours). Gated behind
    ``TERRITORY_RUN_LONG_ROLLOUT_MATCHUP=1`` so routine ``pytest`` runs
    skip it. If this assertion fails, per the spec the outcome is
    interesting (not a regression): keep ``informed`` as the default and
    document that the extra Voronoi cost didn't pay off at this budget.
    """
    if not os.environ.get("TERRITORY_RUN_LONG_ROLLOUT_MATCHUP"):
        pytest.skip(
            "set TERRITORY_RUN_LONG_ROLLOUT_MATCHUP=1 to run "
            "(expected wall time: hours)"
        )
    voronoi = UCTAgent(
        iterations=2000,
        rollout_kind="voronoi_guided",
        rng=np.random.default_rng(8001),
        name="uct-voronoi",
    )
    informed = UCTAgent(
        iterations=2000,
        rollout_kind="informed",
        rng=np.random.default_rng(8002),
        name="uct-informed",
    )
    results = tournament(
        agent_a=voronoi,
        agent_b=informed,
        num_games=200,
        board_size=15,
        seed=2026,
    )
    decided = results["wins_a"] + results["wins_b"]
    assert decided > 0, "no decided games"
    voronoi_share = results["wins_a"] / decided
    assert voronoi_share > 0.50, (
        f"voronoi_guided did not outperform informed at long horizon: "
        f"{results}, voronoi_share={voronoi_share:.2%}. Per the spec this "
        "is a characterization result, not a regression: keep 'informed' "
        "as the default rollout and document the outcome."
    )
