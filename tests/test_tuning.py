"""Tests for the (1+lambda)-ES weight tuner in ``territory_takeover.eval.tuning``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from territory_takeover.eval.heuristic import (
    LinearEvaluator,
    default_evaluator,
)
from territory_takeover.eval.tuning import FEATURE_KEYS, tune_weights
from territory_takeover.search import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
    run_match,
)


def _greedy_factory(weights: dict) -> HeuristicGreedyAgent:
    # Fresh LinearEvaluator each call so no feature cache is shared.
    return HeuristicGreedyAgent(evaluator=LinearEvaluator(weights), name="cand")


def test_tune_weights_smoke_beats_default(tmp_path: Path) -> None:
    """3 gen x 4 pop x 4 games/eval finishes fast and beats default in 100-game match.

    The opponent pool is the default-weighted greedy itself, so fitness is
    a direct signal of "beat default". Tuning against UniformRandomAgent
    (as the CLI's Stage A does) is also valid but gives much noisier
    gradient information at this budget; we cover that path via the CLI
    smoke in the stage-A round-trip, not this unit test.
    """
    log_path = tmp_path / "tune_log.jsonl"

    t0 = time.perf_counter()
    tuned = tune_weights(
        base_agent_factory=_greedy_factory,
        opponent_agents=[
            HeuristicGreedyAgent(evaluator=default_evaluator(), name="default"),
        ],
        num_generations=3,
        population_size=4,
        games_per_eval=4,
        board_size=10,
        num_players=2,
        seed=7,
        parallel=False,
        log_path=log_path,
    )
    tune_elapsed = time.perf_counter() - t0
    assert tune_elapsed < 300.0, f"tune_weights smoke took {tune_elapsed:.1f}s, expected < 300s"

    # Every tuned key is in bounds.
    for key in FEATURE_KEYS:
        assert key in tuned, f"missing key {key!r} in tuned weights"
        assert -2.0 <= tuned[key] <= 2.0, f"{key}={tuned[key]} out of bounds [-2, 2]"

    # LinearEvaluator can consume the tuned dict without error.
    _ = LinearEvaluator(tuned)

    # Validation match: tuned greedy vs default greedy, 100 games on 10x10.
    result = run_match(
        agents=[
            HeuristicGreedyAgent(evaluator=LinearEvaluator(tuned), name="tuned"),
            HeuristicGreedyAgent(evaluator=default_evaluator(), name="default"),
        ],
        num_games=100,
        board_size=10,
        swap_seats=True,
        seed=99,
        parallel=False,
        num_players=2,
    )
    tuned_wins = result.per_agent[0].wins
    assert tuned_wins > 50, (
        f"tuned weights did not beat default: {tuned_wins}/100. "
        "Fitness may be too noisy — try games_per_eval=8 or widen sigma0."
    )

    # JSONL log: one record per evaluated candidate, all required keys, monotone gen.
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 * 4, f"expected 12 records, got {len(lines)}"
    required_keys = {
        "generation",
        "candidate_index",
        "weights",
        "wins",
        "games",
        "win_rate",
        "seed",
        "elapsed_s",
    }
    last_gen = -1
    for line in lines:
        rec = json.loads(line)
        assert required_keys <= set(rec), f"record missing keys: {required_keys - set(rec)}"
        assert rec["generation"] >= last_gen, f"non-monotone generation: {rec}"
        last_gen = rec["generation"]


def test_tuned_weights_replay_reproducible(tmp_path: Path) -> None:
    """Every JSONL record can be replayed from its logged seed to the same wins count.

    This is the strongest determinism guarantee the tuner can give: logs
    are not just a post-hoc record but a complete specification of each
    evaluation. A caller can pick any line and rerun the exact match.
    """
    log_path = tmp_path / "tune_log.jsonl"
    opponent = UniformRandomAgent(name="r")

    _ = tune_weights(
        base_agent_factory=_greedy_factory,
        opponent_agents=[opponent],
        num_generations=2,
        population_size=3,
        games_per_eval=4,
        board_size=10,
        num_players=2,
        seed=123,
        parallel=False,
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines, "tuner produced no log records"

    # Replay the last record (most-mutated weights) and the first (default parent).
    for rec in (json.loads(lines[0]), json.loads(lines[-1])):
        candidate = HeuristicGreedyAgent(
            evaluator=LinearEvaluator(rec["weights"]), name="cand"
        )
        # Multi-opponent replay: sub-spawn the same per-opponent seeds the
        # tuner used. With a single opponent this reduces to one spawn.
        sub_seqs = np.random.SeedSequence(rec["seed"]).spawn(1)
        sub_seed = int(sub_seqs[0].generate_state(1, dtype=np.uint32)[0])
        replay = run_match(
            agents=[candidate, UniformRandomAgent(name="r")],
            num_games=rec["games"],
            board_size=10,
            swap_seats=True,
            seed=sub_seed,
            parallel=False,
            num_players=2,
        )
        assert replay.per_agent[0].wins == rec["wins"], (
            f"replay mismatch on gen={rec['generation']} "
            f"cand={rec['candidate_index']}: logged={rec['wins']} "
            f"replay={replay.per_agent[0].wins}"
        )
