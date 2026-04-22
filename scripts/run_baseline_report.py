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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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

    # Running the actual tournament is wired up in the next commit.
    print("run: not yet implemented (use --dry-run for now)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
