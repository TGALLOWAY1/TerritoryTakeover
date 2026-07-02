"""Tests for the LinearEvaluator in territory_takeover.eval.heuristic."""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray

from territory_takeover import eval as tt_eval
from territory_takeover.actions import legal_actions
from territory_takeover.constants import EMPTY, OWNED_CODES
from territory_takeover.engine import new_game, step
from territory_takeover.eval import heuristic as heuristic_mod
from territory_takeover.eval.features import choke_pressure as shared_choke_pressure
from territory_takeover.eval.heuristic import LinearEvaluator, default_evaluator
from territory_takeover.eval.voronoi import voronoi_partition as _real_voronoi_partition
from territory_takeover.state import GameState, PlayerState


def _make_state(
    grid: NDArray[np.int8], heads: list[tuple[int, int]]
) -> GameState:
    players = [
        PlayerState(
            player_id=i,
            head=h,
            territory_count=int(np.count_nonzero(grid == OWNED_CODES[i])),
            alive=True,
        )
        for i, h in enumerate(heads)
    ]
    state = GameState(grid=grid, players=players)
    state.alive_count = len(players)
    state.empty_count = int(np.count_nonzero(grid == EMPTY))
    return state


def _build_midgame_40x40(n_plies: int = 120) -> GameState:
    """Seeded random rollout to produce a realistic mid-game 40x40 state."""
    state = new_game(board_size=40, seed=0)
    rng = np.random.default_rng(42)
    for _ in range(n_plies):
        if state.done:
            break
        legal = legal_actions(state, state.current_player)
        if not legal:
            break
        action = int(rng.choice(legal))
        step(state, action)
    return state


def test_evaluate_empty_start_state_symmetric() -> None:
    # Symmetric corner spawns on a blank 40x40 board: every feature should be
    # identical across the four players, so the score vector must be flat.
    state = new_game(board_size=40)
    scores = default_evaluator().evaluate(state)

    assert scores.shape == (4,)
    for i in range(1, 4):
        assert abs(scores[i] - scores[0]) < 1e-9, (
            f"scores[{i}]={scores[i]} differs from scores[0]={scores[0]}"
        )


def test_evaluate_shape_and_dtype() -> None:
    state = new_game(board_size=40)
    scores = default_evaluator().evaluate(state)

    assert scores.shape == (4,)
    assert scores.dtype == np.float64


def test_evaluate_deterministic() -> None:
    state = new_game(board_size=40, seed=7)
    evaluator = default_evaluator()
    a = evaluator.evaluate(state)
    b = evaluator.evaluate(state)

    assert np.array_equal(a, b)


def test_evaluate_for_matches_evaluate_slice() -> None:
    state = new_game(board_size=40, seed=3)
    evaluator = default_evaluator()
    vec = evaluator.evaluate(state)

    for pid in range(4):
        assert float(vec[pid]) == evaluator.evaluate_for(state, pid), f"pid={pid}"


def test_dead_player_scored_via_features_not_sentinel() -> None:
    # Under the corrected rules death keeps territory, so a dead player is
    # scored with the same features as everyone else — no sentinel. Their
    # reachable_area collapses to territory_count and choke_pressure
    # saturates at 1.0.
    grid = np.zeros((6, 6), dtype=np.int8)
    for c in range(5):
        grid[0, c] = OWNED_CODES[0]
    grid[5, 5] = OWNED_CODES[1]
    state = _make_state(grid, [(0, 0), (5, 5)])
    state.players[0].alive = False
    state.alive_count = 1

    reach = LinearEvaluator({"reachable_area": 1.0}).evaluate(state)
    assert reach[0] == 5.0, "dead player's reachable_area must equal territory_count"
    assert reach[1] > reach[0], "living player owns the rest of the board"

    choke = LinearEvaluator({"choke_pressure": 1.0}).evaluate(state)
    assert choke[0] == 1.0

    # A dead player with a big territory still outscores a small living one
    # under a territory-weighted evaluator — death is not a catastrophe.
    territory = LinearEvaluator({"territory_total": 1.0}).evaluate(state)
    assert territory[0] == 5.0
    assert territory[1] == 1.0
    assert territory[0] > territory[1]


def test_unknown_weight_key_raises() -> None:
    with pytest.raises(ValueError, match="not_a_feature"):
        LinearEvaluator({"not_a_feature": 1.0})


def test_zero_weights_evaluator_returns_zero_for_all_players() -> None:
    state = new_game(board_size=40)
    state.players[3].alive = False
    state.alive_count = 3
    scores = LinearEvaluator({}).evaluate(state)

    for pid in range(4):
        assert scores[pid] == 0.0, f"pid={pid}"


def test_mobility_zero_dominated_by_mobility_four() -> None:
    # P0 head surrounded on all four sides by P1 territory -> mobility 0.
    # P1 head placed at (6, 6) with four empty neighbours -> mobility 4.
    # Under a mobility-only evaluator, P1's score must strictly exceed P0's.
    grid = np.zeros((10, 10), dtype=np.int8)
    grid[3, 3] = OWNED_CODES[0]  # P0 head
    grid[2, 3] = OWNED_CODES[1]  # wall above P0
    grid[4, 3] = OWNED_CODES[1]  # wall below P0
    grid[3, 2] = OWNED_CODES[1]  # wall left of P0
    grid[3, 4] = OWNED_CODES[1]  # wall right of P0
    grid[6, 6] = OWNED_CODES[1]  # P1 head
    state = _make_state(grid, [(3, 3), (6, 6)])

    evaluator = LinearEvaluator({"mobility": 1.0})
    scores = evaluator.evaluate(state)

    assert scores[0] == 0.0
    assert scores[1] == 4.0
    assert scores[0] < scores[1]


def test_mobility_feature_counts_traversal_moves() -> None:
    # P0's head has one own-cell neighbour (traversal) and two empty
    # neighbours; the mobility feature counts all three, while
    # claiming_mobility counts only the two claims.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = OWNED_CODES[0]
    grid[2, 1] = OWNED_CODES[0]
    grid[1, 2] = OWNED_CODES[1]
    state = _make_state(grid, [(2, 2), (1, 2)])

    mob = LinearEvaluator({"mobility": 1.0}).evaluate(state)
    claim = LinearEvaluator({"claiming_mobility": 1.0}).evaluate(state)

    assert mob[0] == 3.0
    assert claim[0] == 2.0


def test_opponent_distance_handles_solo_survivor() -> None:
    # Only P0 is alive; head_opponent_distance returns math.inf, but the
    # evaluator must substitute 0.0 so the weighted sum stays finite.
    state = new_game(board_size=40)
    for pid in range(1, 4):
        state.players[pid].alive = False
    state.alive_count = 1

    evaluator = LinearEvaluator({"opponent_distance": 0.05})
    score = evaluator.evaluate_for(state, 0)

    assert np.isfinite(score)
    assert score == 0.0


def test_voronoi_computed_once_per_evaluate() -> None:
    # Wrap the voronoi_partition symbol imported into heuristic.py with a
    # call counter. default_evaluator().evaluate() must hit it exactly once
    # even though several features (reachable_area, choke_pressure) would
    # otherwise each recompute the partition.
    state = new_game(board_size=40)
    calls = 0
    original = _real_voronoi_partition

    def counting(st: GameState) -> NDArray[np.int8]:
        nonlocal calls
        calls += 1
        return original(st)

    heuristic_mod.__dict__["voronoi_partition"] = counting
    try:
        default_evaluator().evaluate(state)
    finally:
        heuristic_mod.__dict__["voronoi_partition"] = original

    assert calls == 1, f"voronoi_partition called {calls} times, expected 1"


def test_choke_pressure_feature_matches_shared_impl() -> None:
    # Pin the inlined solo-BFS in LinearEvaluator._feat_choke_pressure to the
    # canonical features.choke_pressure output so the two cannot silently drift.
    state = _build_midgame_40x40()
    evaluator = LinearEvaluator({"choke_pressure": 1.0})
    scores = evaluator.evaluate(state)

    for pid in range(len(state.players)):
        expected = shared_choke_pressure(state, pid)
        assert scores[pid] == pytest.approx(expected, abs=1e-12), (
            f"pid={pid} evaluator={scores[pid]} shared={expected}"
        )


def test_default_evaluator_uses_documented_feature_set() -> None:
    # Guard against feature-set drift between the evaluator defaults and the
    # tuner's search space.
    from territory_takeover.eval.tuning import FEATURE_KEYS

    assert set(default_evaluator().weights) == set(FEATURE_KEYS)


def test_evaluator_exported_from_eval_package() -> None:
    # Guard against forgetting to re-export from the subpackage __init__.
    assert tt_eval.LinearEvaluator is LinearEvaluator
    assert tt_eval.default_evaluator is default_evaluator
    # The old-rules DEAD_SENTINEL constant must stay gone.
    assert not hasattr(heuristic_mod, "DEAD_SENTINEL")


def test_evaluate_40x40_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    # Mid-game 40x40 state: measure mean evaluate() wall time over 100 trials
    # and print the result. The spec's < 1 ms target is aspirational — per
    # CLAUDE.md, performance targets in this repo are "checked manually, not
    # in CI". The assertion here is deliberately loose (50 ms) so slow or
    # noisy shared runners don't flake; regressions an order of magnitude
    # worse than expected still fail.
    state = _build_midgame_40x40()
    evaluator = default_evaluator()
    evaluator.evaluate(state)  # warmup

    n_trials = 100
    t0 = time.perf_counter()
    for _ in range(n_trials):
        evaluator.evaluate(state)
    elapsed = time.perf_counter() - t0
    mean_sec = elapsed / n_trials

    with capsys.disabled():
        print(f"\nevaluate() mean: {mean_sec * 1e6:.1f} us over {n_trials} trials")
    assert mean_sec < 50e-3, (
        f"benchmark mean {mean_sec * 1e6:.1f} us far exceeds any sensible budget"
    )
