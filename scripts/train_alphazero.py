"""CLI entry point for Phase 3c AlphaZero training.

Usage:
    python scripts/train_alphazero.py \
        --config configs/phase3c_alphazero_8x8_2p.yaml --seed 0

Every run writes ``<out_dir>/runs/<timestamp>/`` containing:
    - ``config.yaml``: resolved config.
    - ``iteration_log.csv``: per-iteration losses + buffer size + half-moves.
    - ``net_<iter>.pt`` + ``net_final.pt``: network snapshots.

Install with ``pip install -e ".[rl_deep,dev]"`` to pull in torch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import yaml

from territory_takeover.rl.alphazero.network import AZNetConfig
from territory_takeover.rl.alphazero.selfplay import SelfPlayConfig
from territory_takeover.rl.alphazero.train import TrainConfig, train_loop


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--games-per-iteration", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    with args.config.open() as f:
        raw = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else int(raw.get("seed", 0))
    device = args.device if args.device is not None else str(raw.get("device", "cpu"))
    out_root = Path(args.out_dir if args.out_dir is not None else raw["out_dir"])
    tag = args.tag or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")

    net_cfg = AZNetConfig(
        board_size=int(raw["board_size"]),
        num_players=int(raw["num_players"]),
        **raw.get("net", {}),
    )

    tr = dict(raw.get("train", {}))
    if args.num_iterations is not None:
        tr["num_iterations"] = args.num_iterations
    if args.games_per_iteration is not None:
        tr["games_per_iteration"] = args.games_per_iteration
    train_cfg = TrainConfig(**tr)

    self_play_cfg = SelfPlayConfig(
        board_size=int(raw["board_size"]),
        num_players=int(raw["num_players"]),
        **raw.get("self_play", {}),
    )

    metrics = train_loop(
        net_cfg,
        train_cfg,
        self_play_cfg,
        out_dir=run_dir,
        seed=seed,
        device=device,
    )
    print(f"run complete: {run_dir}")
    print(f"  iterations: {len(metrics.iterations)}")
    if metrics.total_losses:
        print(
            f"  total loss: {metrics.total_losses[0]:.4f} -> "
            f"{metrics.total_losses[-1]:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
