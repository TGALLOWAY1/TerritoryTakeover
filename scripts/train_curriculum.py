#!/usr/bin/env python
"""Phase 3d curriculum trainer CLI.

Single entry point to run a full curriculum end-to-end. Loads a
curriculum YAML (net hyperparams + train hyperparams + stage schedule),
runs :func:`train_curriculum`, and writes artifacts under ``--out-dir``.

Usage::

    python scripts/train_curriculum.py \
        --config configs/phase3d_curriculum.yaml \
        --seed 0 \
        --out-dir results/phase3d/runs/<tag>

Resumption is a future feature — for now the script persists
``curriculum_progress.yaml`` on every evaluation, but restarting reruns
from stage 0. (Reference-agent training is cheap enough at reduced
scope that re-runs are acceptable.)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from territory_takeover.rl.alphazero.network import AZNetConfig
from territory_takeover.rl.curriculum import (
    CurriculumTrainConfig,
    load_schedule_yaml,
    train_curriculum,
)


def _load_config(path: Path) -> tuple[AZNetConfig, CurriculumTrainConfig]:
    raw = yaml.safe_load(path.read_text())
    net_raw = raw.get("net", {})
    net_cfg = AZNetConfig(
        board_size=int(net_raw.get("board_size", 10)),
        num_players=int(net_raw.get("num_players", 2)),
        num_res_blocks=int(net_raw.get("num_res_blocks", 4)),
        channels=int(net_raw.get("channels", 64)),
        value_hidden=int(net_raw.get("value_hidden", 64)),
        scalar_value_head=bool(net_raw.get("scalar_value_head", False)),
        head_type=str(net_raw.get("head_type", "conv")),  # type: ignore[arg-type]
    )

    train_raw = raw.get("train", {})
    train_cfg = CurriculumTrainConfig(
        batch_size=int(train_raw.get("batch_size", 64)),
        learning_rate=float(train_raw.get("learning_rate", 1e-3)),
        l2_weight=float(train_raw.get("l2_weight", 1e-4)),
        value_loss_coef=float(train_raw.get("value_loss_coef", 1.0)),
        buffer_capacity=int(train_raw.get("buffer_capacity", 50_000)),
        eval_games_per_check=int(train_raw.get("eval_games_per_check", 16)),
        iterations_per_eval=int(train_raw.get("iterations_per_eval", 1)),
    )
    return net_cfg, train_cfg


def _write_reproducibility_stamp(out_dir: Path, config_path: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config_path.parent, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    (out_dir / "git_sha.txt").write_text(sha + "\n")
    (out_dir / "invocation.yaml").write_text(
        yaml.safe_dump({"seed": int(seed), "config": str(config_path)})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3d curriculum trainer.")
    parser.add_argument("--config", required=True, help="Phase 3d YAML config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tag",
        default=None,
        help="Subdir under --out-dir. Defaults to UTC timestamp.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_path = Path(args.config)
    net_cfg, train_cfg = _load_config(config_path)
    schedule = load_schedule_yaml(config_path)

    out_root = Path(args.out_dir) if args.out_dir else Path("results/phase3d/runs")
    tag = args.tag or _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / tag

    _write_reproducibility_stamp(out_dir, config_path, args.seed)

    print(f"[curriculum] out_dir={out_dir}")
    print(f"[curriculum] stages={[s.name for s in schedule.stages]}")
    results = train_curriculum(
        schedule=schedule,
        train_cfg=train_cfg,
        template=net_cfg,
        out_dir=out_dir,
        seed=args.seed,
        device=args.device,
    )
    for r in results:
        print(
            f"  {r.name:20s} steps={r.self_play_steps:6d} "
            f"iters={r.num_iterations:4d} "
            f"win_rate_vs_random={r.final_win_rate_vs_random:.3f} "
            f"first_enc_step={r.first_enclosure_step}"
        )
    print(f"[curriculum] final checkpoint: {out_dir / 'net_final.pt'}")


if __name__ == "__main__":
    main()
