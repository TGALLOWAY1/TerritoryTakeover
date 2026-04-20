"""Evaluate a trained AlphaZero checkpoint against baseline agents.

Usage:
    python scripts/eval_alphazero.py \\
        --checkpoint results/phase3c/runs/<stamp>/net_final.pt \\
        --config configs/phase3c_alphazero_8x8_2p.yaml \\
        --games 200

Prints a markdown table of win rates with Wilson 95% CIs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent
from territory_takeover.rl.alphazero.network import AZNetConfig
from territory_takeover.search.harness import tournament
from territory_takeover.search.mcts.uct import UCTAgent
from territory_takeover.search.random_agent import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
)


def _wilson_ci(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games == 0:
        return (float("nan"), float("nan"))
    p = wins / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return (center - half, center + half)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--uct-iters", type=int, default=32)
    parser.add_argument("--uct-games", type=int, default=100)
    parser.add_argument("--mcts-iters", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    with args.config.open() as f:
        raw = yaml.safe_load(f)

    board_size = int(raw["board_size"])
    num_players = int(raw["num_players"])
    if num_players != 2:
        raise SystemExit("Eval script currently supports 2-player boards only.")

    net_cfg = AZNetConfig(
        board_size=board_size,
        num_players=num_players,
        **raw.get("net", {}),
    )
    az = AlphaZeroAgent.from_checkpoint(
        str(args.checkpoint),
        net_cfg,
        iterations=args.mcts_iters,
        device=args.device,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    rand = UniformRandomAgent(rng=np.random.default_rng(rng.integers(1 << 32)))
    greedy = HeuristicGreedyAgent(rng=np.random.default_rng(rng.integers(1 << 32)))
    uct = UCTAgent(
        iterations=args.uct_iters,
        rng=np.random.default_rng(rng.integers(1 << 32)),
    )

    matchups: list[tuple[str, object, int]] = [
        ("random", rand, args.games),
        ("greedy", greedy, args.games),
        (f"uct-{args.uct_iters}", uct, args.uct_games),
    ]

    print(f"## Eval of {args.checkpoint}")
    print()
    print(
        f"- board: {board_size}x{board_size} | players: {num_players} | "
        f"mcts_iters: {args.mcts_iters}"
    )
    print()
    print("| opponent | games | win | loss | tie | win_rate | 95% CI |")
    print("|---|---|---|---|---|---|---|")
    for name, opp, g in matchups:
        az.reset()
        result = tournament(
            az, opp, g, board_size=board_size, seed=int(rng.integers(1 << 32))
        )
        wins = result["wins_a"]
        losses = result["wins_b"]
        ties = result["ties"]
        rate = wins / g if g else float("nan")
        lo, hi = _wilson_ci(wins, g)
        print(
            f"| {name} | {g} | {wins} | {losses} | {ties} | "
            f"{rate:.3f} | [{lo:.3f}, {hi:.3f}] |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
