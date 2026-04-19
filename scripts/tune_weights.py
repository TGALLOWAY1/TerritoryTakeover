"""CLI entry point for the two-stage evaluator-weight tuner.

Stage A tunes :class:`HeuristicGreedyAgent` weights against
:class:`UniformRandomAgent` (cheap smoke — greedy-vs-random games on a
small board finish in tens of ms). Stage B tunes
:class:`ParanoidAgent` (depth=2) weights against a frozen
:class:`HeuristicGreedyAgent` built with Stage A's tuned weights; this
catches the failure mode where weights that look good at 1-ply greedy
are bad once the search looks further ahead.

Final weights are written to ``configs/tuned_weights.yaml`` under a
nested mapping (``stage_a_greedy``, ``stage_b_paranoid_d2``). Per-stage
JSONL logs are dropped into the output directory for provenance.

Install ``pyyaml`` via ``pip install -e ".[tournament]"`` — the JSONL
logs are produced even without it; the helpful error only fires at the
final YAML save.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from territory_takeover.eval.heuristic import LinearEvaluator, default_evaluator
from territory_takeover.eval.tuning import FEATURE_KEYS, tune_weights
from territory_takeover.search import (
    HeuristicGreedyAgent,
    ParanoidAgent,
    UniformRandomAgent,
    run_match,
)

if TYPE_CHECKING:
    from territory_takeover.search.agent import Agent


def _require_yaml() -> ModuleType:
    """Import PyYAML with a helpful error if the extra is not installed."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise SystemExit(
            "pyyaml is required for scripts/tune_weights.py. "
            'Install with: pip install -e ".[tournament]"'
        ) from exc
    return yaml


def _greedy_factory(weights: dict[str, float]) -> Agent:
    return HeuristicGreedyAgent(evaluator=LinearEvaluator(weights), name="greedy-cand")


def _paranoid_d2_factory(weights: dict[str, float]) -> Agent:
    return ParanoidAgent(
        depth=2, evaluator=LinearEvaluator(weights), name="paranoid-d2-cand"
    )


def _validate_vs_default(
    kind: str,
    candidate_weights: dict[str, float],
    board_size: int,
    num_games: int,
    seed: int,
) -> tuple[int, int]:
    """Run a 100-ish-game head-to-head of tuned vs default and return (wins, games).

    ``kind`` is either ``"greedy"`` or ``"paranoid-d2"``; it selects the
    agent type and gives tuned/default opponents distinct names so
    :func:`run_match` aggregation keys the two sides apart.
    """
    default_weights = default_evaluator().weights
    if kind == "greedy":
        candidate = HeuristicGreedyAgent(
            evaluator=LinearEvaluator(candidate_weights), name="tuned"
        )
        opponent = HeuristicGreedyAgent(
            evaluator=LinearEvaluator(default_weights), name="default"
        )
    elif kind == "paranoid-d2":
        candidate = ParanoidAgent(
            depth=2, evaluator=LinearEvaluator(candidate_weights), name="tuned"
        )
        opponent = ParanoidAgent(
            depth=2, evaluator=LinearEvaluator(default_weights), name="default"
        )
    else:
        raise ValueError(f"unknown kind {kind!r}; expected 'greedy' or 'paranoid-d2'")
    result = run_match(
        agents=[candidate, opponent],
        num_games=num_games,
        board_size=board_size,
        swap_seats=True,
        seed=seed,
        parallel=False,
        num_players=2,
    )
    return result.per_agent[0].wins, result.num_games


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune LinearEvaluator weights (two stages).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSONL logs (default: results/tuning_<timestamp>/).",
    )
    p.add_argument(
        "--config-out",
        type=Path,
        default=Path("configs/tuned_weights.yaml"),
        help="Path to write the final YAML weights artifact.",
    )
    p.add_argument("--board-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--stage-a-generations", type=int, default=8)
    p.add_argument("--stage-a-pop", type=int, default=8)
    p.add_argument("--stage-a-games", type=int, default=8)

    p.add_argument("--stage-b-generations", type=int, default=6)
    p.add_argument("--stage-b-pop", type=int, default=6)
    p.add_argument("--stage-b-games", type=int, default=8)

    p.add_argument(
        "--validation-games",
        type=int,
        default=100,
        help="Games per stage for the tuned-vs-default sanity print-out.",
    )

    parallel = p.add_mutually_exclusive_group()
    parallel.add_argument("--parallel", action="store_true")
    parallel.add_argument("--no-parallel", action="store_true")

    p.add_argument(
        "--skip-stage-b",
        action="store_true",
        help="Only run Stage A (useful for quick smoke runs).",
    )
    return p.parse_args(argv)


def _default_output_dir() -> Path:
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / f"tuning_{ts}"


def _write_weights_yaml(
    path: Path,
    stage_a_weights: dict[str, float],
    stage_b_weights: dict[str, float] | None,
) -> None:
    yaml = _require_yaml()
    payload: dict[str, dict[str, float]] = {"stage_a_greedy": dict(stage_a_weights)}
    if stage_b_weights is not None:
        payload["stage_b_paranoid_d2"] = dict(stage_b_weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def _print_weights(label: str, weights: dict[str, float]) -> None:
    print(f"\n{label}")
    for k in FEATURE_KEYS:
        print(f"  {k:30s} {weights[k]:+.4f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.parallel:
        parallel = True
    elif args.no_parallel:
        parallel = False
    else:
        parallel = False

    out_dir = args.output_dir or _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: greedy vs UniformRandom.
    print(
        f"Stage A: tuning HeuristicGreedyAgent weights vs UniformRandomAgent on "
        f"{args.board_size}x{args.board_size} "
        f"({args.stage_a_generations} gen x {args.stage_a_pop} pop x "
        f"{args.stage_a_games} games)"
    )
    stage_a_weights = tune_weights(
        base_agent_factory=_greedy_factory,
        opponent_agents=[UniformRandomAgent(name="random")],
        num_generations=args.stage_a_generations,
        population_size=args.stage_a_pop,
        games_per_eval=args.stage_a_games,
        board_size=args.board_size,
        num_players=2,
        seed=args.seed,
        parallel=parallel,
        log_path=out_dir / "stage_a.jsonl",
    )
    _print_weights("Stage A best weights:", stage_a_weights)
    a_wins, a_total = _validate_vs_default(
        "greedy",
        stage_a_weights,
        board_size=args.board_size,
        num_games=args.validation_games,
        seed=args.seed ^ 0x5A5A5A5A,
    )
    print(
        f"Stage A validation — tuned greedy vs default greedy: "
        f"{a_wins}/{a_total} wins ({100 * a_wins / a_total:.1f}%)"
    )

    stage_b_weights: dict[str, float] | None = None
    if not args.skip_stage_b:
        # Stage B: paranoid d=2 vs frozen Stage-A greedy.
        stage_a_opponent = HeuristicGreedyAgent(
            evaluator=LinearEvaluator(stage_a_weights), name="greedy-tuned"
        )
        print(
            f"\nStage B: tuning ParanoidAgent(depth=2) weights vs Stage-A-tuned greedy "
            f"on {args.board_size}x{args.board_size} "
            f"({args.stage_b_generations} gen x {args.stage_b_pop} pop x "
            f"{args.stage_b_games} games)"
        )
        stage_b_weights = tune_weights(
            base_agent_factory=_paranoid_d2_factory,
            opponent_agents=[stage_a_opponent],
            num_generations=args.stage_b_generations,
            population_size=args.stage_b_pop,
            games_per_eval=args.stage_b_games,
            board_size=args.board_size,
            num_players=2,
            seed=args.seed ^ 0xBEEF,
            parallel=parallel,
            log_path=out_dir / "stage_b.jsonl",
        )
        _print_weights("Stage B best weights:", stage_b_weights)
        b_wins, b_total = _validate_vs_default(
            "paranoid-d2",
            stage_b_weights,
            board_size=args.board_size,
            num_games=args.validation_games,
            seed=args.seed ^ 0xC0FFEE,
        )
        print(
            f"Stage B validation — tuned paranoid-d2 vs default paranoid-d2: "
            f"{b_wins}/{b_total} wins ({100 * b_wins / b_total:.1f}%)"
        )

    _write_weights_yaml(args.config_out, stage_a_weights, stage_b_weights)
    print(f"\nLogs: {out_dir}\nWeights: {args.config_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
