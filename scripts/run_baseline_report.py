"""Generate the canonical Territory Takeover baseline report.

Builds a fixed roster — Random, Greedy, UCT, RAVE, and the curriculum
reference checkpoint — and runs a pairwise head-to-head tournament on a
fixed 10x10 / 2p board config (the same config the curriculum reference
was trained on). Aggregates per-pair win rates with Wilson 95% CIs and
per-agent decision-latency stats into two artifacts:

- ``docs/baseline_report.md``  — recruiter-readable leaderboard.
- ``docs/baseline_report.csv`` — flat per-pair rows.

Usage::

    python scripts/run_baseline_report.py --games-per-pair 40 --seed 0

The curriculum entry requires ``torch`` and the reference checkpoint at
``docs/phase3d/net_reference.pt``. Pass ``--skip-curriculum`` to run
with the four classical agents only.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.search.harness import (
    AgentStats,
    GameLog,
    _aggregate,
    _wilson_ci,
    run_match,
)
from territory_takeover.search.mcts.rave import RaveAgent
from territory_takeover.search.mcts.uct import UCTAgent
from territory_takeover.search.random_agent import HeuristicGreedyAgent, UniformRandomAgent

if TYPE_CHECKING:
    from territory_takeover.search.agent import Agent


DEFAULT_CHECKPOINT = Path("docs/phase3d/net_reference.pt")


@dataclass(frozen=True, slots=True)
class RosterConfig:
    """Knobs for building the canonical roster.

    Defaults are tuned so UCT/RAVE are strong enough to beat Random/Greedy
    convincingly on 10x10 2p while keeping wall-clock per pair in seconds,
    not minutes. AZ iterations mirror ``configs/phase3d_elo_pool.yaml``.
    """

    seed: int = 0
    uct_iterations: int = 200
    rave_iterations: int = 200
    az_iterations: int = 4
    az_c_puct: float = 1.25
    skip_curriculum: bool = False
    checkpoint_path: Path = field(default_factory=lambda: DEFAULT_CHECKPOINT)


def _spawn_seeds(seed: int, n: int) -> list[int]:
    """Derive ``n`` 32-bit sub-seeds deterministically from one integer."""
    ss = np.random.SeedSequence(seed)
    return [int(s) for s in ss.generate_state(n, dtype=np.uint32)]


def _build_curriculum_agent(cfg: RosterConfig, seed: int) -> Agent:
    """Load the reference curriculum checkpoint as an AlphaZero agent.

    Network architecture mirrors ``docs/phase3d/reference_config.yaml``.
    Import of ``torch``-dependent modules is deferred so the script stays
    importable (and ``--dry-run`` works) without the ``rl_deep`` extra.
    """
    from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent
    from territory_takeover.rl.alphazero.network import AZNetConfig

    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Curriculum checkpoint not found at {cfg.checkpoint_path}. "
            "Pass --skip-curriculum to run without it."
        )
    net_cfg = AZNetConfig(
        board_size=10,
        num_players=2,
        num_res_blocks=2,
        channels=32,
        value_hidden=32,
        head_type="conv",
    )
    return AlphaZeroAgent.from_checkpoint(
        path=str(cfg.checkpoint_path),
        cfg=net_cfg,
        iterations=cfg.az_iterations,
        c_puct=cfg.az_c_puct,
        device="cpu",
        name="curriculum_ref",
        seed=seed,
    )


def build_roster(cfg: RosterConfig) -> list[Agent]:
    """Construct the canonical agent roster.

    Seeds are spawned from ``cfg.seed`` via :class:`numpy.random.SeedSequence`
    so the roster is bit-identical across runs with the same seed.
    """
    seeds = _spawn_seeds(cfg.seed, 5)
    agents: list[Agent] = [
        UniformRandomAgent(name="random", rng=np.random.default_rng(seeds[0])),
        HeuristicGreedyAgent(name="greedy", rng=np.random.default_rng(seeds[1])),
        UCTAgent(
            iterations=cfg.uct_iterations,
            name="uct",
            rng=np.random.default_rng(seeds[2]),
        ),
        RaveAgent(
            iterations=cfg.rave_iterations,
            name="rave",
            rng=np.random.default_rng(seeds[3]),
        ),
    ]
    if not cfg.skip_curriculum:
        agents.append(_build_curriculum_agent(cfg, seed=seeds[4]))
    return agents


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Knobs for the tournament run proper."""

    games_per_pair: int = 40
    board_size: int = 10
    num_players: int = 2
    seed: int = 0
    parallel: bool = False


@dataclass(frozen=True, slots=True)
class PairAggregate:
    """One row of the per-pair leaderboard.

    Wins are counted from ``agent_a``'s perspective; ``win_rate_a`` is
    ``wins_a / games`` (ties count against the win rate as the audit
    calls for). ``ci_low``/``ci_high`` are the Wilson 95% interval.
    """

    agent_a: str
    agent_b: str
    games: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Full result of a :func:`run_all_pairs` run — what the emitters consume."""

    board_size: int
    num_players: int
    games_per_pair: int
    seed: int
    pairs: list[PairAggregate]
    per_agent: list[AgentStats]


def run_all_pairs(roster: list[Agent], cfg: RunConfig) -> BaselineResult:
    """Play every unordered pair in the roster head-to-head via :func:`run_match`.

    Each pair gets its own child :class:`SeedSequence` derived from
    ``cfg.seed``, so any single pair can be re-run independently with the
    same seed and produce bit-identical games.

    ``cfg.games_per_pair`` must be a multiple of ``2`` so that
    ``swap_seats=True`` can rotate seats evenly between the two agents.
    """
    if cfg.games_per_pair < 0:
        raise ValueError(
            f"games_per_pair must be >= 0; got {cfg.games_per_pair}"
        )
    if cfg.games_per_pair % 2 != 0:
        raise ValueError(
            "games_per_pair must be a multiple of 2 so seat rotation is "
            f"balanced across the two seats; got {cfg.games_per_pair}"
        )
    if len(roster) < 2:
        raise ValueError("roster must contain at least 2 agents")
    if cfg.num_players != 2:
        raise ValueError(
            f"run_all_pairs currently supports num_players=2 only; "
            f"got {cfg.num_players}"
        )

    pairs_idx = list(itertools.combinations(range(len(roster)), 2))
    pair_ss = np.random.SeedSequence(cfg.seed)
    pair_seqs = pair_ss.spawn(len(pairs_idx))

    aggregates: list[PairAggregate] = []
    all_games: list[GameLog] = []

    for pi, (ai, bi) in enumerate(pairs_idx):
        agent_a = roster[ai]
        agent_b = roster[bi]
        pair_seed = int(pair_seqs[pi].generate_state(1, dtype=np.uint32)[0])

        result = run_match(
            agents=[agent_a, agent_b],
            num_games=cfg.games_per_pair,
            board_size=cfg.board_size,
            swap_seats=True,
            seed=pair_seed,
            parallel=cfg.parallel,
            num_players=cfg.num_players,
        )

        wins_a = 0
        wins_b = 0
        ties = 0
        for g in result.games:
            a_seat = g.seat_assignment.index(agent_a.name)
            if g.winner_seat is None:
                ties += 1
            elif g.winner_seat == a_seat:
                wins_a += 1
            else:
                wins_b += 1

        win_rate_a = wins_a / cfg.games_per_pair if cfg.games_per_pair > 0 else 0.0
        ci_low, ci_high = _wilson_ci(wins_a, cfg.games_per_pair)
        aggregates.append(
            PairAggregate(
                agent_a=agent_a.name,
                agent_b=agent_b.name,
                games=cfg.games_per_pair,
                wins_a=wins_a,
                wins_b=wins_b,
                ties=ties,
                win_rate_a=win_rate_a,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
        all_games.extend(result.games)

    per_agent = _aggregate([a.name for a in roster], all_games)
    return BaselineResult(
        board_size=cfg.board_size,
        num_players=cfg.num_players,
        games_per_pair=cfg.games_per_pair,
        seed=cfg.seed,
        pairs=aggregates,
        per_agent=per_agent,
    )


def write_pair_csv(path: Path, result: BaselineResult) -> None:
    """Flatten ``result.pairs`` to a per-pair CSV with a stable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "agent_a",
                "agent_b",
                "games",
                "wins_a",
                "wins_b",
                "ties",
                "win_rate_a",
                "ci_low",
                "ci_high",
            ]
        )
        for r in result.pairs:
            w.writerow(
                [
                    r.agent_a,
                    r.agent_b,
                    r.games,
                    r.wins_a,
                    r.wins_b,
                    r.ties,
                    f"{r.win_rate_a:.6f}",
                    f"{r.ci_low:.6f}",
                    f"{r.ci_high:.6f}",
                ]
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the canonical Territory Takeover baseline report."
    )
    p.add_argument(
        "--games-per-pair",
        type=int,
        default=40,
        help="Games per pairwise head-to-head (must be even for seat rotation).",
    )
    p.add_argument("--seed", type=int, default=0, help="Root seed.")
    p.add_argument(
        "--board-size", type=int, default=10, help="Square board edge length."
    )
    p.add_argument(
        "--uct-iterations",
        type=int,
        default=200,
        help="Per-move simulation budget for UCT.",
    )
    p.add_argument(
        "--rave-iterations",
        type=int,
        default=200,
        help="Per-move simulation budget for RAVE.",
    )
    p.add_argument(
        "--az-iterations",
        type=int,
        default=4,
        help="Per-move PUCT budget for the curriculum checkpoint.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the curriculum reference checkpoint.",
    )
    p.add_argument(
        "--skip-curriculum",
        action="store_true",
        help="Omit the curriculum AlphaZero agent (no torch required).",
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=Path("docs/baseline_report.md"),
        help="Output path for the markdown report.",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=Path("docs/baseline_report.csv"),
        help="Output path for the flat per-pair CSV.",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Dispatch games across a multiprocessing pool.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the roster and print names/params, then exit.",
    )
    return p.parse_args(argv)


def _roster_config_from_args(args: argparse.Namespace) -> RosterConfig:
    return RosterConfig(
        seed=args.seed,
        uct_iterations=args.uct_iterations,
        rave_iterations=args.rave_iterations,
        az_iterations=args.az_iterations,
        skip_curriculum=args.skip_curriculum,
        checkpoint_path=args.checkpoint,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _roster_config_from_args(args)

    if args.dry_run:
        print(f"Roster config: seed={cfg.seed}, board={args.board_size}x{args.board_size}, 2p")
        print(f"  skip_curriculum={cfg.skip_curriculum}")
        if not cfg.skip_curriculum:
            print(f"  checkpoint={cfg.checkpoint_path}")
        agents = build_roster(cfg)
        print(f"Roster ({len(agents)} agents):")
        for a in agents:
            print(f"  - {a.name}  ({type(a).__name__})")
        return 0

    run_cfg = RunConfig(
        games_per_pair=args.games_per_pair,
        board_size=args.board_size,
        num_players=2,
        seed=args.seed,
        parallel=args.parallel,
    )
    roster = build_roster(cfg)
    print(
        f"[baseline] roster={[a.name for a in roster]} "
        f"board={run_cfg.board_size}x{run_cfg.board_size} "
        f"games_per_pair={run_cfg.games_per_pair} seed={run_cfg.seed}"
    )
    result = run_all_pairs(roster, run_cfg)
    write_pair_csv(args.csv_out, result)
    print(f"[baseline] wrote {args.csv_out}")

    # Markdown emitter lands in the next commit.
    return 0


if __name__ == "__main__":
    sys.exit(main())
