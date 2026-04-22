"""Record a 4-panel deterministic "agent behavior" gallery PNG.

Plays four self-play games on the same board size / seed — one per
agent (Random, Greedy, UCT@200, RAVE@200) — and renders the state at a
fixed target turn (falling back to the terminal state if the game ends
earlier) into a 2x2 matplotlib grid.

The output is checked into ``docs/assets/agent_gallery.png`` and
embedded in the project README. The script is deterministic on a fixed
root seed — re-running with the same seed produces a visually identical
PNG.

Usage::

    python scripts/record_agent_gallery.py --seed 0
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
from territory_takeover.search.mcts.uct import UCTAgent
from territory_takeover.search.random_agent import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
)
from territory_takeover.viz import render_matplotlib

if TYPE_CHECKING:
    from territory_takeover.search.agent import Agent
    from territory_takeover.state import GameState


@dataclass(frozen=True, slots=True)
class GalleryConfig:
    """Knobs for the agent gallery."""

    board_size: int = 20
    num_players: int = 2
    seed: int = 0
    uct_iterations: int = 200
    rave_iterations: int = 200
    target_turn: int = 100


# Order matters — this is the order panels appear in the 2x2 grid
# (row-major: top-left, top-right, bottom-left, bottom-right).
PANEL_LABELS: tuple[str, ...] = (
    "Random",
    "Greedy",
    "UCT",
    "RAVE",
)


def _build_roster_for_panel(
    panel_index: int, cfg: GalleryConfig, panel_ss: np.random.SeedSequence
) -> list[Agent]:
    """Build two identical agents (self-play) for one panel.

    Seat seeds are spawned from ``panel_ss`` so each panel is
    deterministic from ``cfg.seed`` alone.
    """
    seat_seeds = panel_ss.generate_state(cfg.num_players, dtype=np.uint32)
    rngs = [np.random.default_rng(int(s)) for s in seat_seeds]

    if panel_index == 0:
        return [UniformRandomAgent(rng=r, name="random") for r in rngs]
    if panel_index == 1:
        return [HeuristicGreedyAgent(rng=r, name="greedy") for r in rngs]
    if panel_index == 2:
        return [
            UCTAgent(iterations=cfg.uct_iterations, rng=r, name="uct")
            for r in rngs
        ]
    if panel_index == 3:
        return [
            RaveAgent(iterations=cfg.rave_iterations, rng=r, name="rave")
            for r in rngs
        ]
    raise ValueError(f"panel_index must be in [0, 4); got {panel_index}")


def _play_until(
    roster: list[Agent], cfg: GalleryConfig, game_seed: int
) -> GameState:
    """Play a self-play game and return the state at ``cfg.target_turn``.

    If the game ends before ``target_turn``, returns the terminal state.
    """
    for agent in roster:
        agent.reset()

    state = new_game(
        board_size=cfg.board_size,
        num_players=cfg.num_players,
        seed=game_seed,
    )

    while not state.done and state.turn_number < cfg.target_turn:
        seat = state.current_player
        action = roster[seat].select_action(state, seat)
        step(state, action, strict=True)

    return state.copy()


def build_panels(cfg: GalleryConfig) -> list[tuple[str, GameState]]:
    """Play all four self-play games and return (label, state) per panel.

    Each panel gets an independent child SeedSequence so panels are
    deterministic individually. All four games share the same
    ``game_seed`` derived from ``cfg.seed``, so the starting board
    (spawn positions) is identical across panels — only play style
    differs.
    """
    root = np.random.SeedSequence(cfg.seed)
    game_seed = int(root.generate_state(1, dtype=np.uint32)[0])
    panel_ss_list = root.spawn(4)

    panels: list[tuple[str, GameState]] = []
    for i, label in enumerate(PANEL_LABELS):
        roster = _build_roster_for_panel(i, cfg, panel_ss_list[i])
        if i == 2:
            full_label = f"{label}@{cfg.uct_iterations} self-play"
        elif i == 3:
            full_label = f"{label}@{cfg.rave_iterations} self-play"
        else:
            full_label = f"{label} self-play"
        print(
            f"[gallery] panel {i} ({full_label}): playing up to "
            f"turn {cfg.target_turn}..."
        )
        state = _play_until(roster, cfg, game_seed)
        print(
            f"[gallery] panel {i} done: turn={state.turn_number} "
            f"done={state.done} winner={state.winner}"
        )
        panels.append((full_label, state))
    return panels


def render_gallery(
    panels: list[tuple[str, GameState]], cfg: GalleryConfig, out: Path
) -> None:
    """Render the four panels into a 2x2 figure and save to ``out``."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (label, state) in zip(axes.flat, panels, strict=True):
        render_matplotlib(state, ax=ax, show_heads=True)
        ax.set_title(label, fontsize=14)
    fig.suptitle(
        f"Agent behavior at turn {cfg.target_turn} - "
        f"{cfg.board_size}x{cfg.board_size}, {cfg.num_players}p self-play, "
        f"seed {cfg.seed}",
        fontsize=15,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render a 4-panel agent-behavior gallery (Random / Greedy / "
            "UCT / RAVE self-play) to a PNG."
        )
    )
    p.add_argument("--seed", type=int, default=0, help="Root seed.")
    p.add_argument("--board-size", type=int, default=20)
    p.add_argument("--target-turn", type=int, default=100)
    p.add_argument("--uct-iterations", type=int, default=200)
    p.add_argument("--rave-iterations", type=int, default=200)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/agent_gallery.png"),
        help="Output PNG path.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running games.",
    )
    return p.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> GalleryConfig:
    return GalleryConfig(
        board_size=args.board_size,
        seed=args.seed,
        uct_iterations=args.uct_iterations,
        rave_iterations=args.rave_iterations,
        target_turn=args.target_turn,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)

    if args.dry_run:
        print(
            f"Gallery config: seed={cfg.seed}, "
            f"board={cfg.board_size}x{cfg.board_size}, {cfg.num_players}p, "
            f"target_turn={cfg.target_turn}"
        )
        print(f"  uct_iterations={cfg.uct_iterations}")
        print(f"  rave_iterations={cfg.rave_iterations}")
        print(f"  out={args.out}")
        return 0

    panels = build_panels(cfg)
    print(f"[gallery] writing 2x2 PNG to {args.out}")
    render_gallery(panels, cfg, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
