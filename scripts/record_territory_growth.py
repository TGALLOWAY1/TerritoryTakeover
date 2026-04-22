"""Record a territory-growth plot for one deterministic game.

Plays one 20x20 / 2-player game (RAVE@200 vs Greedy, seed 0 by
default) and plots each player's total territory (path length +
claimed cells) as a function of turn number. This makes the
dynamics of a game legible: who gains ground when, when enclosures
fire, who pulls ahead.

The output is checked into ``docs/assets/territory_growth.png`` and
embedded in the project README. Deterministic on a fixed seed.

Usage::

    python scripts/record_territory_growth.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.engine import new_game, step
from territory_takeover.search.mcts.rave import RaveAgent
from territory_takeover.search.random_agent import HeuristicGreedyAgent

if TYPE_CHECKING:
    from territory_takeover.search.agent import Agent
    from territory_takeover.state import GameState


@dataclass(frozen=True, slots=True)
class GrowthConfig:
    board_size: int = 20
    num_players: int = 2
    seed: int = 0
    rave_iterations: int = 200


def build_roster(cfg: GrowthConfig) -> list[Agent]:
    """Seat 0: RAVE@200. Seat 1: Greedy. Deterministic from ``cfg.seed``."""
    ss = np.random.SeedSequence(cfg.seed)
    agent_seeds = ss.generate_state(2, dtype=np.uint32)
    rave = RaveAgent(
        iterations=cfg.rave_iterations,
        name="RAVE@200",
        rng=np.random.default_rng(int(agent_seeds[0])),
    )
    greedy = HeuristicGreedyAgent(
        rng=np.random.default_rng(int(agent_seeds[1])),
        name="Greedy",
    )
    return [rave, greedy]


def play_game(roster: list[Agent], cfg: GrowthConfig) -> list[GameState]:
    """Play to termination and return the per-half-move trajectory."""
    for agent in roster:
        agent.reset()

    root = np.random.SeedSequence(cfg.seed)
    game_seed = int(root.generate_state(1, dtype=np.uint32)[0])

    state = new_game(
        board_size=cfg.board_size,
        num_players=cfg.num_players,
        seed=game_seed,
    )
    trajectory: list[GameState] = [state.copy()]
    while not state.done:
        seat = state.current_player
        action = roster[seat].select_action(state, seat)
        step(state, action, strict=True)
        trajectory.append(state.copy())
    return trajectory


def render_plot(
    trajectory: list[GameState],
    roster: list[Agent],
    cfg: GrowthConfig,
    out: Path,
) -> None:
    """Draw total-territory-over-turn lines for each player."""
    import matplotlib.pyplot as plt

    turns = [s.turn_number for s in trajectory]
    # Total territory = path length + claimed count (the scoring formula
    # used by `compute_terminal_reward("relative")` and by the baseline
    # leaderboard's tie-break).
    territory_per_player = [
        [len(s.players[p].path) + s.players[p].claimed_count for s in trajectory]
        for p in range(cfg.num_players)
    ]
    claimed_per_player = [
        [s.players[p].claimed_count for s in trajectory]
        for p in range(cfg.num_players)
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    colors = ["#1f77b4", "#d62728"]  # blue (seat 0 = RAVE), red (seat 1 = Greedy)
    for p in range(cfg.num_players):
        ax_top.plot(
            turns,
            territory_per_player[p],
            color=colors[p],
            linewidth=2.0,
            label=f"{roster[p].name} (seat {p})",
        )
        ax_bot.plot(
            turns,
            claimed_per_player[p],
            color=colors[p],
            linewidth=2.0,
            label=f"{roster[p].name} (seat {p})",
        )

    ax_top.set_ylabel("Total territory (path + claimed)")
    ax_top.set_title(
        f"Territory growth - {cfg.board_size}x{cfg.board_size}, "
        f"{cfg.num_players}p, seed {cfg.seed}"
    )
    ax_top.legend(loc="upper left")
    ax_top.grid(alpha=0.3)

    ax_bot.set_xlabel("Turn number")
    ax_bot.set_ylabel("Enclosed (claimed) cells")
    ax_bot.grid(alpha=0.3)
    # Visually separate the two sub-plots with a tight layout.

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render a territory-growth plot (RAVE@200 vs Greedy, self-enclosure "
            "dynamics over one game) to a PNG."
        )
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--board-size", type=int, default=20)
    p.add_argument("--rave-iterations", type=int, default=200)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/territory_growth.png"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> GrowthConfig:
    return GrowthConfig(
        board_size=args.board_size,
        seed=args.seed,
        rave_iterations=args.rave_iterations,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    if args.dry_run:
        print(f"Growth config: {cfg}, out={args.out}")
        return 0
    roster = build_roster(cfg)
    print(
        f"[growth] playing {roster[0].name} (seat 0) vs {roster[1].name} "
        f"(seat 1) on {cfg.board_size}x{cfg.board_size}, seed={cfg.seed}"
    )
    trajectory = play_game(roster, cfg)
    final = trajectory[-1]
    territory = [
        len(final.players[p].path) + final.players[p].claimed_count
        for p in range(cfg.num_players)
    ]
    print(
        f"[growth] final: turn={final.turn_number} winner={final.winner} "
        f"territory={territory}"
    )
    print(f"[growth] writing PNG to {args.out}")
    render_plot(trajectory, roster, cfg, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
