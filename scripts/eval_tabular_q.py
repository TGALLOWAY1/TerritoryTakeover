"""Evaluate a trained TabularQAgent checkpoint against baseline agents.

Usage:
    python scripts/eval_tabular_q.py \
        --checkpoint results/phase3a/runs/<stamp>/q_table.pkl \
        --board-size 8 --num-players 2 --games 500

Prints a markdown-formatted summary table to stdout. When --plot is passed,
also renders win-rate curves from ``eval_curves.csv`` in the checkpoint's run
directory to ``<run>/winrate_vs_<opp>.png``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from territory_takeover.rl.tabular.eval import evaluate_vs, evaluate_vs_4p
from territory_takeover.rl.tabular.q_agent import TabularQAgent
from territory_takeover.search.random_agent import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--num-players", type=int, default=2, choices=(2, 4))
    parser.add_argument(
        "--spawn",
        type=str,
        default="",
        help=(
            "Spawn positions as 'r,c;r,c;...' (default: corners for the board). "
            "Empty string means use engine defaults."
        ),
    )
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--uct-iters", type=int, default=32)
    parser.add_argument("--uct-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def _default_spawns(
    board_size: int, num_players: int
) -> list[tuple[int, int]] | None:
    if num_players == 2:
        return [(0, 0), (board_size - 1, board_size - 1)]
    if num_players == 4:
        return [
            (0, 0),
            (0, board_size - 1),
            (board_size - 1, 0),
            (board_size - 1, board_size - 1),
        ]
    return None


def _parse_spawns(raw: str) -> list[tuple[int, int]] | None:
    if not raw:
        return None
    out: list[tuple[int, int]] = []
    for token in raw.split(";"):
        r, c = token.split(",")
        out.append((int(r), int(c)))
    return out


def _plot_curves(run_dir: Path) -> None:
    csv_path = run_dir / "eval_curves.csv"
    if not csv_path.exists():
        print(f"  (no eval_curves.csv at {csv_path}; skipping plots)")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (matplotlib unavailable: {exc})")
        return

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"  (eval_curves.csv empty at {csv_path})")
        return

    episodes = [float(r["episode"]) for r in rows]
    for column in rows[0]:
        if not column.startswith("win_rate_vs_"):
            continue
        ys = [float(r[column]) if r[column] else float("nan") for r in rows]
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(episodes, ys, marker="o")
        ax.set_xlabel("episode")
        ax.set_ylabel(column)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.set_title(column.replace("_", " "))
        png = run_dir / f"{column}.png"
        fig.tight_layout()
        fig.savefig(png)
        plt.close(fig)
        print(f"  wrote {png}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    agent = TabularQAgent.load(args.checkpoint, rng=np.random.default_rng(args.seed))
    agent.set_greedy(True)

    spawn = _parse_spawns(args.spawn)
    if spawn is None:
        spawn = _default_spawns(args.board_size, args.num_players)

    rng = np.random.default_rng(args.seed)
    rand = UniformRandomAgent(rng=np.random.default_rng(rng.integers(1 << 32)))
    greedy = HeuristicGreedyAgent(rng=np.random.default_rng(rng.integers(1 << 32)))

    eval_fn = evaluate_vs if args.num_players == 2 else evaluate_vs_4p

    rows: list[tuple[str, dict[str, float]]] = []
    rows.append(
        (
            "random",
            eval_fn(
                agent,
                rand,
                args.games,
                args.board_size,
                spawn,
                int(rng.integers(1 << 32)),
            ),
        )
    )
    rows.append(
        (
            "greedy",
            eval_fn(
                agent,
                greedy,
                args.games,
                args.board_size,
                spawn,
                int(rng.integers(1 << 32)),
            ),
        )
    )
    if args.num_players == 2 and args.uct_games > 0:
        from territory_takeover.search.mcts.uct import UCTAgent

        uct = UCTAgent(
            iterations=args.uct_iters,
            rng=np.random.default_rng(rng.integers(1 << 32)),
        )
        rows.append(
            (
                f"uct-{args.uct_iters}",
                evaluate_vs(
                    agent,
                    uct,
                    args.uct_games,
                    args.board_size,
                    spawn,
                    int(rng.integers(1 << 32)),
                ),
            )
        )

    print(f"## Evaluation of {args.checkpoint}")
    print()
    print(
        f"- board: {args.board_size}x{args.board_size} | players: {args.num_players} "
        f"| spawns: {spawn}"
    )
    print(f"- Q-table size: {len(agent.q_table)}")
    print()
    print("| opponent | games | win | loss | tie | win_rate | 95% CI |")
    print("|---|---|---|---|---|---|---|")
    for name, r in rows:
        print(
            f"| {name} | {int(r['games'])} | {int(r['wins'])} | "
            f"{int(r['losses'])} | {int(r['ties'])} | "
            f"{r['win_rate']:.3f} | "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] |"
        )

    if args.plot:
        run_dir = args.checkpoint.parent
        print(f"\nPlotting curves from {run_dir / 'eval_curves.csv'}:")
        _plot_curves(run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
