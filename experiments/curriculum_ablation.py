#!/usr/bin/env python
"""Phase 3d headline ablation: curriculum vs no-curriculum.

Runs both arms at three seeds each, measures:

1. time-to-first-enclosure (self-play steps in the final stage),
2. final win rate vs a fixed :class:`UniformRandomAgent`,
3. total self-play step count consumed.

Writes one CSV per seed x arm under ``--out-dir`` plus a summary with
Welch's t and Mann-Whitney U across seeds. With N=3 seeds these tests
are underpowered; the script emphasizes effect sizes (Cohen's d,
rank-biserial) and 95% bootstrap CIs as the primary evidence.

Usage::

    python experiments/curriculum_ablation.py \
        --curriculum-config configs/phase3d_curriculum.yaml \
        --direct-config configs/phase3d_direct.yaml \
        --seeds 0,1,2 \
        --out-dir results/phase3d/ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from territory_takeover.rl.alphazero.network import AZNetConfig
from territory_takeover.rl.curriculum import (
    CurriculumTrainConfig,
    Schedule,
    StageResult,
    load_schedule_yaml,
    train_curriculum,
)


def _load_config(path: Path) -> tuple[AZNetConfig, CurriculumTrainConfig, Schedule]:
    raw = yaml.safe_load(path.read_text())
    net_raw = raw.get("net", {})
    net_cfg = AZNetConfig(
        board_size=int(net_raw.get("board_size", 10)),
        num_players=int(net_raw.get("num_players", 2)),
        num_res_blocks=int(net_raw.get("num_res_blocks", 2)),
        channels=int(net_raw.get("channels", 32)),
        value_hidden=int(net_raw.get("value_hidden", 32)),
        head_type=str(net_raw.get("head_type", "conv")),  # type: ignore[arg-type]
    )
    train_raw = raw.get("train", {})
    train_cfg = CurriculumTrainConfig(
        batch_size=int(train_raw.get("batch_size", 32)),
        learning_rate=float(train_raw.get("learning_rate", 1e-3)),
        l2_weight=float(train_raw.get("l2_weight", 1e-4)),
        value_loss_coef=float(train_raw.get("value_loss_coef", 1.0)),
        buffer_capacity=int(train_raw.get("buffer_capacity", 20_000)),
        eval_games_per_check=int(train_raw.get("eval_games_per_check", 8)),
        iterations_per_eval=int(train_raw.get("iterations_per_eval", 1)),
    )
    schedule = load_schedule_yaml(path)
    return net_cfg, train_cfg, schedule


@dataclass
class SeedResult:
    arm: str
    seed: int
    total_self_play_steps: int
    final_stage_steps: int
    final_win_rate_vs_random: float
    first_enclosure_step: int | None
    wall_clock_seconds: float
    stage_results: list[dict[str, Any]] = field(default_factory=list)


def _run_one_seed(
    arm: str,
    seed: int,
    config_path: Path,
    out_dir: Path,
) -> SeedResult:
    net_cfg, train_cfg, schedule = _load_config(config_path)
    run_dir = out_dir / f"arm_{arm}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.monotonic()
    results: list[StageResult] = train_curriculum(
        schedule=schedule,
        train_cfg=train_cfg,
        template=net_cfg,
        out_dir=run_dir,
        seed=seed,
    )
    elapsed = time.monotonic() - t0

    total_steps = sum(r.self_play_steps for r in results)
    final = results[-1]
    first_enc_step = final.first_enclosure_step
    if first_enc_step is None:
        for r in results:
            if r.first_enclosure_step is not None:
                first_enc_step = r.first_enclosure_step
                break

    return SeedResult(
        arm=arm,
        seed=seed,
        total_self_play_steps=total_steps,
        final_stage_steps=final.self_play_steps,
        final_win_rate_vs_random=final.final_win_rate_vs_random,
        first_enclosure_step=first_enc_step,
        wall_clock_seconds=elapsed,
        stage_results=[asdict(r) for r in results],
    )


def _cohens_d(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.stdev(xs), statistics.stdev(ys)
    pooled = math.sqrt(((len(xs) - 1) * sx**2 + (len(ys) - 1) * sy**2) / (len(xs) + len(ys) - 2))
    if pooled == 0.0:
        return float("nan")
    return (mx - my) / pooled


def _welch_t(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Welch's t-statistic and two-sided p-value via t-distribution
    approximation. With N=3 this is order-of-magnitude only."""
    if len(xs) < 2 or len(ys) < 2:
        return float("nan"), float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx, vy = statistics.variance(xs), statistics.variance(ys)
    se2 = vx / len(xs) + vy / len(ys)
    if se2 == 0:
        return float("nan"), float("nan")
    t = (mx - my) / math.sqrt(se2)
    # Welch-Satterthwaite degrees of freedom.
    df = (se2**2) / (
        (vx / len(xs)) ** 2 / (len(xs) - 1) + (vy / len(ys)) ** 2 / (len(ys) - 1)
    )
    # Two-sided p via the survival of a t: closed-form requires incomplete beta.
    # Approximate with the normal tail — with N=3 the t vs normal gap is small
    # compared to the sampling noise we already have.
    from math import erf, sqrt

    z = abs(t)
    p_norm = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return t, p_norm


def _mann_whitney_u(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Exact Mann-Whitney U plus rank-biserial effect size.

    For small samples we enumerate rank assignments directly. Returns
    ``(U, rank_biserial)``. P-values are not computed (N=3 is too small
    for them to mean anything).
    """
    combined = sorted(
        [(v, "x") for v in xs] + [(v, "y") for v in ys],
        key=lambda t: t[0],
    )
    ranks: dict[int, float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_x = sum(ranks[i] for i, (_, tag) in enumerate(combined) if tag == "x")
    nx, ny = len(xs), len(ys)
    u = rank_x - nx * (nx + 1) / 2
    rank_biserial = 1.0 - 2.0 * u / (nx * ny)
    return u, rank_biserial


def _bootstrap_ci(xs: list[float], iters: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if len(xs) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(12345)
    means = []
    arr = np.array(xs)
    for _ in range(iters):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        means.append(float(sample.mean()))
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def summarize(results: list[SeedResult]) -> dict[str, Any]:
    by_arm: dict[str, list[SeedResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)

    summary: dict[str, Any] = {"arms": {}}
    for arm, rs in by_arm.items():
        win_rates = [r.final_win_rate_vs_random for r in rs]
        first_encs = [r.first_enclosure_step for r in rs if r.first_enclosure_step is not None]
        steps = [r.total_self_play_steps for r in rs]
        wins_lo, wins_hi = _bootstrap_ci(win_rates)
        summary["arms"][arm] = {
            "n_seeds": len(rs),
            "n_seeds_with_enclosure": len(first_encs),
            "final_win_rate_vs_random_mean": statistics.mean(win_rates),
            "final_win_rate_vs_random_95ci": [wins_lo, wins_hi],
            "first_enclosure_step_mean": statistics.mean(first_encs) if first_encs else None,
            "total_steps_mean": statistics.mean(steps),
        }

    arm_names = list(by_arm.keys())
    if len(arm_names) == 2:
        a, b = arm_names
        wr_a = [r.final_win_rate_vs_random for r in by_arm[a]]
        wr_b = [r.final_win_rate_vs_random for r in by_arm[b]]
        fe_a = [r.first_enclosure_step for r in by_arm[a] if r.first_enclosure_step is not None]
        fe_b = [r.first_enclosure_step for r in by_arm[b] if r.first_enclosure_step is not None]

        t_stat, p_val = _welch_t(wr_a, wr_b)
        u_stat, rb = _mann_whitney_u(wr_a, wr_b)
        summary["test_win_rate"] = {
            "arm_a": a,
            "arm_b": b,
            "welch_t": t_stat,
            "welch_p_approx": p_val,
            "mann_whitney_u": u_stat,
            "rank_biserial": rb,
            "cohens_d": _cohens_d(wr_a, wr_b),
        }

        if len(fe_a) >= 2 and len(fe_b) >= 2:
            fe_a_f = [float(x) for x in fe_a]
            fe_b_f = [float(x) for x in fe_b]
            t_stat2, p_val2 = _welch_t(fe_a_f, fe_b_f)
            u_stat2, rb2 = _mann_whitney_u(fe_a_f, fe_b_f)
            summary["test_first_enclosure"] = {
                "arm_a": a,
                "arm_b": b,
                "welch_t": t_stat2,
                "welch_p_approx": p_val2,
                "mann_whitney_u": u_stat2,
                "rank_biserial": rb2,
                "cohens_d": _cohens_d(fe_a_f, fe_b_f),
            }
    return summary


def _write_raw(results: list[SeedResult], out_dir: Path) -> None:
    path = out_dir / "raw.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "arm",
                "seed",
                "total_self_play_steps",
                "final_stage_steps",
                "final_win_rate_vs_random",
                "first_enclosure_step",
                "wall_clock_seconds",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.arm,
                    r.seed,
                    r.total_self_play_steps,
                    r.final_stage_steps,
                    f"{r.final_win_rate_vs_random:.4f}",
                    r.first_enclosure_step if r.first_enclosure_step is not None else "",
                    f"{r.wall_clock_seconds:.2f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3d curriculum vs direct ablation.")
    parser.add_argument("--curriculum-config", default="configs/phase3d_curriculum.yaml")
    parser.add_argument("--direct-config", default="configs/phase3d_direct.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out-dir", default="results/phase3d/ablation")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[SeedResult] = []
    for seed in seeds:
        for arm, cfg in (("curriculum", args.curriculum_config), ("direct", args.direct_config)):
            print(f"[ablation] arm={arm} seed={seed} — starting")
            r = _run_one_seed(arm, seed, Path(cfg), out_dir)
            print(
                f"[ablation] arm={arm} seed={seed} done: "
                f"wr={r.final_win_rate_vs_random:.3f} "
                f"first_enc={r.first_enclosure_step} "
                f"steps={r.total_self_play_steps} "
                f"elapsed={r.wall_clock_seconds:.1f}s"
            )
            results.append(r)

    _write_raw(results, out_dir)
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
