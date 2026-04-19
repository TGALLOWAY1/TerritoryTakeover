"""CLI entry point for Phase 3a tabular Q-learning training.

Usage:
    python scripts/train_tabular_q.py --config configs/phase3a_tabular_8x8_2p.yaml --seed 0

Every run writes ``results/phase3a/runs/<timestamp>/`` containing:
    - ``config.yaml``: resolved config (overrides applied).
    - ``episode_log.csv``: per-1k-episode training metrics.
    - ``eval_curves.csv``: win rates vs Random / Greedy / UCT at each eval tick.
    - ``q_table_ep<N>.pkl`` + ``q_table.pkl``: Q-table checkpoints and final.
    - ``tb/``: TensorBoard event files (if tensorboardX is installed).
    - ``summary.yaml``: wall-clock + final Q-table size.

Install with ``pip install -e ".[rl]"`` to pull in tensorboardX + matplotlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from territory_takeover.rl.tabular.config import load_config
from territory_takeover.rl.tabular.train import train


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config (see configs/phase3a_tabular_*.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config.seed for reproducibility.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override config.num_episodes (handy for smoke runs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override config.out_dir.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Explicit run-tag directory name (default: UTC timestamp).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_episodes is not None:
        cfg.num_episodes = args.num_episodes
        # Keep the epsilon-decay horizon aligned with the actual episode count
        # if the caller shortened training from the CLI.
        cfg.q.total_episodes = args.num_episodes
    if args.out_dir is not None:
        cfg.out_dir = str(args.out_dir)

    out = train(cfg, run_tag=args.tag)
    print(f"run complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
