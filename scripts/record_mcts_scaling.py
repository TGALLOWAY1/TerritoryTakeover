"""Record a UCT-vs-random win-rate-vs-compute scaling plot.

Runs UCT at each of a sweep of simulation budgets against a
uniform-random opponent, and plots win rate (with Wilson 95% CI)
as a function of log-scale iterations. Visible proof that the
MCTS implementation scales: more compute -> more wins, monotonic
within CI noise.

Output is checked into ``docs/assets/mcts_scaling.png`` and
embedded in the project README. Deterministic on a fixed root seed.

Defaults: 10x10 board, 2 players, 10 games per budget, budgets
[10, 50, 200, 1000]. At 10x10 the full sweep takes roughly 5-10
minutes on a single core; use ``--games 5`` or ``--board-size 8``
to go faster.

Usage::

    python scripts/record_mcts_scaling.py --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from territory_takeover.search.harness import tournament
from territory_takeover.search.mcts.uct import UCTAgent
from territory_takeover.search.random_agent import UniformRandomAgent


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% CI for k wins out of n trials. Matches harness convention."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True, slots=True)
class ScalingConfig:
    board_size: int = 10
    num_players: int = 2
    seed: int = 0
    games: int = 10
    iterations: tuple[int, ...] = (10, 50, 200, 1000)


@dataclass(frozen=True, slots=True)
class Point:
    iterations: int
    wins: int
    ties: int
    losses: int
    win_rate: float
    ci_low: float
    ci_high: float


def run_sweep(cfg: ScalingConfig) -> list[Point]:
    """Run UCT@iters vs Random for each budget. Ties count against UCT."""
    points: list[Point] = []
    root = np.random.SeedSequence(cfg.seed)
    child_seqs = root.spawn(len(cfg.iterations))
    for iters, child_ss in zip(cfg.iterations, child_seqs, strict=True):
        agent_seeds = child_ss.generate_state(3, dtype=np.uint32)
        uct = UCTAgent(
            iterations=iters,
            rng=np.random.default_rng(int(agent_seeds[0])),
            name=f"uct@{iters}",
        )
        rand = UniformRandomAgent(
            rng=np.random.default_rng(int(agent_seeds[1])),
            name="random",
        )
        print(
            f"[scaling] UCT@{iters} vs random, {cfg.games} games on "
            f"{cfg.board_size}x{cfg.board_size}...",
            flush=True,
        )
        results = tournament(
            agent_a=uct,
            agent_b=rand,
            num_games=cfg.games,
            board_size=cfg.board_size,
            seed=int(agent_seeds[2]),
        )
        wins = int(results["wins_a"])
        losses = int(results["wins_b"])
        ties = int(results["ties"])
        total = wins + losses + ties
        assert total == cfg.games
        win_rate = wins / total
        ci_low, ci_high = _wilson_ci(wins, total)
        points.append(
            Point(
                iterations=iters,
                wins=wins,
                ties=ties,
                losses=losses,
                win_rate=win_rate,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
        print(
            f"[scaling]   -> wins={wins} ties={ties} losses={losses} "
            f"win_rate={win_rate:.2f} CI=[{ci_low:.2f}, {ci_high:.2f}]",
            flush=True,
        )
    return points


def render_plot(points: list[Point], cfg: ScalingConfig, out: Path) -> None:
    import matplotlib.pyplot as plt

    iters = np.array([p.iterations for p in points], dtype=np.float64)
    wr = np.array([p.win_rate for p in points], dtype=np.float64)
    lo = np.array([p.ci_low for p in points], dtype=np.float64)
    hi = np.array([p.ci_high for p in points], dtype=np.float64)
    err_low = wr - lo
    err_high = hi - wr

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        iters,
        wr,
        yerr=[err_low, err_high],
        marker="o",
        linestyle="-",
        linewidth=2.0,
        capsize=4,
        color="#1f77b4",
        label="UCT vs uniform random",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="chance (0.5)")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("UCT simulations per move (log scale)")
    ax.set_ylabel(f"Win rate over {cfg.games} games (Wilson 95% CI)")
    ax.set_title(
        f"MCTS scaling - UCT vs random on "
        f"{cfg.board_size}x{cfg.board_size}, {cfg.num_players}p, seed {cfg.seed}"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render a UCT-vs-random win-rate-vs-compute scaling plot to PNG."
        )
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--board-size", type=int, default=10)
    p.add_argument("--games", type=int, default=10)
    p.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[10, 50, 200, 1000],
        help="UCT simulation budgets to sweep (one point per value).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/mcts_scaling.png"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> ScalingConfig:
    return ScalingConfig(
        board_size=args.board_size,
        seed=args.seed,
        games=args.games,
        iterations=tuple(args.iterations),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    if args.dry_run:
        print(f"Scaling config: {cfg}, out={args.out}")
        return 0
    points = run_sweep(cfg)
    print(f"[scaling] writing PNG to {args.out}")
    render_plot(points, cfg, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
