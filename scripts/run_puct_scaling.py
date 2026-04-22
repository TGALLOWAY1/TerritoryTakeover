"""Sweep curriculum reference's eval-time PUCT iterations against a fixed panel.

Tests whether the Phase-3d curriculum reference checkpoint's playing
strength scales with eval-time compute (``H(b)``) or is compute-insensitive
(``H(a)``). For each PUCT budget in ``--az-iters`` the script rebuilds a
curriculum agent and runs alternating-seat head-to-heads against every
agent in the opponent panel (Random, Greedy, UCT@200 by default),
emitting a markdown table and a flat CSV.

Results are reported as win rate of the curriculum agent (per opponent)
with a Wilson 95% interval, plus the aggregate win rate across all
panel games. A monotone and statistically clear rise with PUCT iters
supports ``H(b)``; flat or noisy curves support ``H(a)``.

Usage::

    python scripts/run_puct_scaling.py \
        --board-size 20 --games-per-opponent 20 \
        --az-iters 4 16 64 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.search.harness import _wilson_ci, run_match
from territory_takeover.search.mcts.uct import UCTAgent
from territory_takeover.search.random_agent import HeuristicGreedyAgent, UniformRandomAgent

if TYPE_CHECKING:
    from territory_takeover.search.agent import Agent


DEFAULT_CHECKPOINT = Path("docs/phase3d/net_reference.pt")


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Knobs for one PUCT-scaling sweep."""

    board_size: int = 20
    games_per_opponent: int = 20
    seed: int = 0
    az_iterations: tuple[int, ...] = (4, 16, 64)
    az_c_puct: float = 1.25
    uct_iterations: int = 200
    checkpoint_path: Path = field(default_factory=lambda: DEFAULT_CHECKPOINT)
    parallel: bool = False


@dataclass(frozen=True, slots=True)
class PanelRow:
    """Curriculum-vs-one-opponent result at one PUCT setting."""

    az_iterations: int
    opponent: str
    games: int
    wins: int
    ties: int
    losses: int
    win_rate: float
    ci_low: float
    ci_high: float
    avg_decision_time_s: float


@dataclass(frozen=True, slots=True)
class SweepResult:
    board_size: int
    games_per_opponent: int
    seed: int
    uct_iterations: int
    az_c_puct: float
    rows: list[PanelRow]


def _spawn_seeds(seed: int, n: int) -> list[int]:
    ss = np.random.SeedSequence(seed)
    return [int(s) for s in ss.generate_state(n, dtype=np.uint32)]


def _build_curriculum_agent(
    iterations: int,
    c_puct: float,
    checkpoint_path: Path,
    seed: int,
    board_size: int,
) -> Agent:
    """Load the reference curriculum checkpoint as an AlphaZero agent.

    The ``head_type="conv"`` architecture accepts arbitrary board sizes,
    so the checkpoint (trained up to 10x10) can be evaluated at 20x20;
    generalization is part of what the sweep measures.
    """
    from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent
    from territory_takeover.rl.alphazero.network import AZNetConfig

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Curriculum checkpoint not found at {checkpoint_path}."
        )
    net_cfg = AZNetConfig(
        board_size=board_size,
        num_players=2,
        num_res_blocks=2,
        channels=32,
        value_hidden=32,
        head_type="conv",
    )
    return AlphaZeroAgent.from_checkpoint(
        path=str(checkpoint_path),
        cfg=net_cfg,
        iterations=iterations,
        c_puct=c_puct,
        device="cpu",
        name=f"curriculum_ref_puct{iterations}",
        seed=seed,
    )


def build_opponent_panel(cfg: SweepConfig, base_seed: int) -> list[Agent]:
    """Construct the fixed opponent panel the curriculum plays against.

    Random and Greedy are trivial; UCT gives a non-trivial strong
    reference at fixed compute so the reader can compare curriculum's
    win rate vs. a compute-comparable classical agent.
    """
    seeds = _spawn_seeds(base_seed, 3)
    return [
        UniformRandomAgent(name="random", rng=np.random.default_rng(seeds[0])),
        HeuristicGreedyAgent(name="greedy", rng=np.random.default_rng(seeds[1])),
        UCTAgent(
            iterations=cfg.uct_iterations,
            name=f"uct{cfg.uct_iterations}",
            rng=np.random.default_rng(seeds[2]),
        ),
    ]


def run_one_pairing(
    az_agent: Agent,
    opponent: Agent,
    cfg: SweepConfig,
    seed: int,
) -> PanelRow:
    """Play ``cfg.games_per_opponent`` games, alternating seats, 2p."""
    if cfg.games_per_opponent % 2 != 0:
        raise ValueError(
            f"games_per_opponent must be even; got {cfg.games_per_opponent}"
        )
    result = run_match(
        agents=[az_agent, opponent],
        num_games=cfg.games_per_opponent,
        board_size=cfg.board_size,
        swap_seats=True,
        seed=seed,
        parallel=cfg.parallel,
        num_players=2,
    )

    wins = 0
    losses = 0
    ties = 0
    total_decision_time = 0.0
    n_decisions = 0
    for g in result.games:
        az_seat = g.seat_assignment.index(az_agent.name)
        total_decision_time += sum(g.decision_times_s[az_seat])
        n_decisions += len(g.decision_times_s[az_seat])
        if g.winner_seat is None:
            ties += 1
        elif g.winner_seat == az_seat:
            wins += 1
        else:
            losses += 1

    games = cfg.games_per_opponent
    win_rate = wins / games if games > 0 else 0.0
    ci_low, ci_high = _wilson_ci(wins, games)
    avg_decision = total_decision_time / n_decisions if n_decisions > 0 else 0.0

    return PanelRow(
        az_iterations=az_agent.iterations,  # type: ignore[attr-defined]
        opponent=opponent.name,
        games=games,
        wins=wins,
        ties=ties,
        losses=losses,
        win_rate=win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        avg_decision_time_s=avg_decision,
    )


def run_sweep(cfg: SweepConfig) -> SweepResult:
    """Run the full sweep: every ``az_iters`` x every panel opponent.

    Seeds are spawned deterministically from ``cfg.seed`` so the full
    sweep is reproducible from one integer.
    """
    if not cfg.az_iterations:
        raise ValueError("az_iterations must be non-empty")

    # Seed tree: one child per (iterations, opponent) pair, plus one root
    # for building the opponent panel.
    num_pairs = len(cfg.az_iterations) * 3  # 3 panel opponents
    pair_seeds = _spawn_seeds(cfg.seed, num_pairs + 1)
    panel_base_seed = pair_seeds[0]
    pair_seeds = pair_seeds[1:]

    rows: list[PanelRow] = []
    for pair_idx, (iters, opp_idx) in enumerate(
        itertools.product(cfg.az_iterations, range(3))
    ):
        panel = build_opponent_panel(cfg, panel_base_seed)
        opponent = panel[opp_idx]
        az = _build_curriculum_agent(
            iterations=iters,
            c_puct=cfg.az_c_puct,
            checkpoint_path=cfg.checkpoint_path,
            seed=pair_seeds[pair_idx],
            board_size=cfg.board_size,
        )
        row = run_one_pairing(az, opponent, cfg, seed=pair_seeds[pair_idx])
        rows.append(row)

    return SweepResult(
        board_size=cfg.board_size,
        games_per_opponent=cfg.games_per_opponent,
        seed=cfg.seed,
        uct_iterations=cfg.uct_iterations,
        az_c_puct=cfg.az_c_puct,
        rows=rows,
    )


def _current_commit_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
    return out.strip() or "unknown"


def _wilson_fmt(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def _group_rows_by_iters(
    rows: list[PanelRow],
) -> dict[int, list[PanelRow]]:
    grouped: dict[int, list[PanelRow]] = {}
    for r in rows:
        grouped.setdefault(r.az_iterations, []).append(r)
    return grouped


def format_report_md(result: SweepResult, checkpoint_path: Path, command: str) -> str:
    """Render the curriculum PUCT-scaling report."""
    grouped = _group_rows_by_iters(result.rows)
    opp_order: list[str] = []
    for r in result.rows:
        if r.opponent not in opp_order:
            opp_order.append(r.opponent)

    header = "| PUCT iters | Opponent | Games | Wins | Ties | Losses | Win rate | 95% CI |"
    sep = "|---:|---|---:|---:|---:|---:|---:|---|"
    table_lines = [header, sep]
    for iters in sorted(grouped):
        for opp in opp_order:
            r = next(x for x in grouped[iters] if x.opponent == opp)
            table_lines.append(
                f"| {r.az_iterations} | {r.opponent} | {r.games} | {r.wins} | "
                f"{r.ties} | {r.losses} | {r.win_rate:.3f} | "
                f"{_wilson_fmt(r.ci_low, r.ci_high)} |"
            )

    agg_header = (
        "| PUCT iters | Aggregate games | Wins | Ties | Losses | "
        "Aggregate win rate | 95% CI | Avg decision (s) |"
    )
    agg_sep = "|---:|---:|---:|---:|---:|---:|---|---:|"
    agg_lines = [agg_header, agg_sep]
    for iters in sorted(grouped):
        rs = grouped[iters]
        tot_games = sum(r.games for r in rs)
        tot_wins = sum(r.wins for r in rs)
        tot_ties = sum(r.ties for r in rs)
        tot_losses = sum(r.losses for r in rs)
        wr = tot_wins / tot_games if tot_games > 0 else 0.0
        lo, hi = _wilson_ci(tot_wins, tot_games)
        times = [r.avg_decision_time_s for r in rs if not math.isnan(r.avg_decision_time_s)]
        avg_time = sum(times) / len(times) if times else 0.0
        agg_lines.append(
            f"| {iters} | {tot_games} | {tot_wins} | {tot_ties} | {tot_losses} | "
            f"{wr:.3f} | {_wilson_fmt(lo, hi)} | {avg_time:.4f} |"
        )

    lines = [
        "# Curriculum Reference — PUCT Scaling Sweep",
        "",
        (
            f"Curriculum reference vs. a fixed panel (Random, Greedy, "
            f"UCT@{result.uct_iterations}) at {result.board_size}x"
            f"{result.board_size}, 2 players, "
            f"{result.games_per_opponent} games per (PUCT, opponent) pairing."
        ),
        "",
        (
            "Tests whether the curriculum checkpoint's playing strength "
            "scales with eval-time PUCT compute. Note: the checkpoint "
            "was trained only up to 10x10 — its 20x20 behaviour is "
            "out-of-distribution."
        ),
        "",
        "## Per-opponent breakdown",
        "",
        "\n".join(table_lines),
        "",
        "## Aggregate win rate by PUCT budget",
        "",
        "\n".join(agg_lines),
        "",
        "## Reproducibility",
        "",
        f"- Board: {result.board_size}x{result.board_size}, 2 players",
        f"- Seed: {result.seed}",
        f"- Games per (PUCT, opponent): {result.games_per_opponent}",
        f"- UCT reference iterations: {result.uct_iterations}",
        f"- AlphaZero c_puct: {result.az_c_puct}",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Commit: `{_current_commit_sha()}`",
        f"- Command: `{command}`",
        "",
        (
            "Wilson 95% intervals are computed on curriculum win rate "
            "(ties count against the win rate)."
        ),
        "",
    ]
    return "\n".join(lines)


def write_rows_csv(path: Path, result: SweepResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "az_iterations",
                "opponent",
                "games",
                "wins",
                "ties",
                "losses",
                "win_rate",
                "ci_low",
                "ci_high",
                "avg_decision_time_s",
            ]
        )
        for r in result.rows:
            w.writerow(
                [
                    r.az_iterations,
                    r.opponent,
                    r.games,
                    r.wins,
                    r.ties,
                    r.losses,
                    f"{r.win_rate:.6f}",
                    f"{r.ci_low:.6f}",
                    f"{r.ci_high:.6f}",
                    f"{r.avg_decision_time_s:.6f}",
                ]
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep curriculum reference's PUCT iterations at fixed budgets."
    )
    p.add_argument(
        "--board-size",
        type=int,
        default=20,
        help="Square board edge length (20 or larger for representative play).",
    )
    p.add_argument(
        "--games-per-opponent",
        type=int,
        default=20,
        help="Games per (PUCT, opponent) pairing. Must be even.",
    )
    p.add_argument("--seed", type=int, default=0, help="Root seed.")
    p.add_argument(
        "--az-iters",
        type=int,
        nargs="+",
        default=[4, 16, 64],
        help="PUCT iteration counts to sweep.",
    )
    p.add_argument(
        "--uct-iterations",
        type=int,
        default=200,
        help="UCT reference budget.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the curriculum reference checkpoint.",
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=Path("docs/curriculum_puct_scaling.md"),
        help="Output path for the markdown report.",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=Path("docs/curriculum_puct_scaling.csv"),
        help="Output path for the flat CSV.",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Dispatch games across a multiprocessing pool.",
    )
    return p.parse_args(argv)


def _sweep_config_from_args(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        board_size=args.board_size,
        games_per_opponent=args.games_per_opponent,
        seed=args.seed,
        az_iterations=tuple(args.az_iters),
        uct_iterations=args.uct_iterations,
        checkpoint_path=args.checkpoint,
        parallel=args.parallel,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _sweep_config_from_args(args)

    print(
        f"[puct-sweep] board={cfg.board_size}x{cfg.board_size} 2p "
        f"games_per_opponent={cfg.games_per_opponent} "
        f"az_iters={cfg.az_iterations} seed={cfg.seed}"
    )
    result = run_sweep(cfg)
    write_rows_csv(args.csv_out, result)
    print(f"[puct-sweep] wrote {args.csv_out}")

    md = format_report_md(result, cfg.checkpoint_path, " ".join(sys.argv))
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(md, encoding="utf-8")
    print(f"[puct-sweep] wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
