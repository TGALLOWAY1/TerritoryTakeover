"""Record a deterministic demo game to an animated GIF.

Plays one 20x20 / 2p game between the RAVE@200 (winner of the
canonical leaderboard) and curriculum_ref@4 (neural net) agents, with
every move captured and the trajectory rendered via
``territory_takeover.viz.save_game_gif``.

The output is checked into ``docs/assets/demo.gif`` and embedded in
the project README. The script is deterministic on a fixed root seed —
re-running it with the same seed produces a visually identical GIF.

Usage::

    python scripts/record_demo.py --seed 0

``--frame-stride 2`` (default) drops every other frame to keep GIF
size bounded; use ``--frame-stride 1`` for every-frame playback.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.engine import new_game, step
from territory_takeover.search.mcts.rave import RaveAgent
from territory_takeover.viz import save_game_gif

if TYPE_CHECKING:
    from territory_takeover.search.agent import Agent
    from territory_takeover.state import GameState


DEFAULT_CHECKPOINT = Path("docs/phase3d/net_reference.pt")


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Knobs for the demo recording."""

    board_size: int = 20
    num_players: int = 2
    seed: int = 0
    rave_iterations: int = 200
    az_iterations: int = 4
    az_c_puct: float = 1.25
    checkpoint_path: Path = field(default_factory=lambda: DEFAULT_CHECKPOINT)
    frame_stride: int = 2
    fps: int = 4


def _build_curriculum_agent(cfg: DemoConfig, seed: int) -> Agent:
    """Load the reference curriculum checkpoint as an AlphaZero agent.

    Deferred torch import so ``--help`` and tests work without the
    ``rl_deep`` extra installed.
    """
    from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent
    from territory_takeover.rl.alphazero.network import AZNetConfig

    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Curriculum checkpoint not found at {cfg.checkpoint_path}."
        )
    net_cfg = AZNetConfig(
        board_size=cfg.board_size,
        num_players=cfg.num_players,
        num_res_blocks=2,
        channels=32,
        value_hidden=32,
        head_type="conv",
    )
    return AlphaZeroAgent.from_checkpoint(
        path=str(cfg.checkpoint_path),
        cfg=net_cfg,
        iterations=cfg.az_iterations,
        c_puct=cfg.az_c_puct,
        device="cpu",
        name="curriculum_ref",
        seed=seed,
    )


def build_roster(cfg: DemoConfig) -> list[Agent]:
    """Build the two-agent demo roster: RAVE@200 (seat 0), curriculum (seat 1).

    Seeds are spawned deterministically from ``cfg.seed`` so the full
    demo is reproducible from one integer.
    """
    ss = np.random.SeedSequence(cfg.seed)
    agent_seeds = ss.generate_state(2, dtype=np.uint32)
    rave = RaveAgent(
        iterations=cfg.rave_iterations,
        name="rave",
        rng=np.random.default_rng(int(agent_seeds[0])),
    )
    az = _build_curriculum_agent(cfg, seed=int(agent_seeds[1]))
    return [rave, az]


def play_demo_game(roster: list[Agent], cfg: DemoConfig) -> list[GameState]:
    """Play one full game and return the per-half-move trajectory.

    The initial state is captured so the trajectory starts from an
    empty board (minus spawns); each subsequent entry is a fresh
    ``state.copy()`` after ``step`` is applied. Callers downsample by
    ``cfg.frame_stride`` before rendering.
    """
    if len(roster) != cfg.num_players:
        raise ValueError(
            f"roster length {len(roster)} != num_players {cfg.num_players}"
        )

    ss = np.random.SeedSequence(cfg.seed)
    game_seed_arr = ss.generate_state(1, dtype=np.uint32)
    game_seed = int(game_seed_arr[0])

    for agent in roster:
        agent.reset()

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


def _downsample(trajectory: list[GameState], stride: int) -> list[GameState]:
    """Keep every ``stride``-th frame plus the last frame (terminal state).

    We always include the final frame so the GIF ends on the terminal
    board, not one move short.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")
    if stride == 1:
        return list(trajectory)
    kept = trajectory[::stride]
    if trajectory and trajectory[-1] is not kept[-1]:
        kept.append(trajectory[-1])
    return kept


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a deterministic demo game to an animated GIF."
    )
    p.add_argument("--seed", type=int, default=0, help="Root seed.")
    p.add_argument("--board-size", type=int, default=20)
    p.add_argument("--rave-iterations", type=int, default=200)
    p.add_argument("--az-iterations", type=int, default=4)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the curriculum reference checkpoint.",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Render every Nth state (1 = every frame).",
    )
    p.add_argument("--fps", type=int, default=4)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/demo.gif"),
        help="Output GIF path.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the roster and print configuration, then exit.",
    )
    return p.parse_args(argv)


def _demo_config_from_args(args: argparse.Namespace) -> DemoConfig:
    return DemoConfig(
        board_size=args.board_size,
        seed=args.seed,
        rave_iterations=args.rave_iterations,
        az_iterations=args.az_iterations,
        checkpoint_path=args.checkpoint,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _demo_config_from_args(args)

    if args.dry_run:
        print(f"Demo config: seed={cfg.seed}, "
              f"board={cfg.board_size}x{cfg.board_size}, {cfg.num_players}p")
        print(f"  rave_iterations={cfg.rave_iterations}")
        print(f"  az_iterations={cfg.az_iterations}")
        print(f"  frame_stride={cfg.frame_stride}, fps={cfg.fps}")
        print(f"  out={args.out}")
        agents = build_roster(cfg)
        print("Roster:")
        for a in agents:
            print(f"  - {a.name}  ({type(a).__name__})")
        return 0

    roster = build_roster(cfg)
    print(
        f"[demo] roster={[a.name for a in roster]} "
        f"board={cfg.board_size}x{cfg.board_size} seed={cfg.seed}"
    )
    trajectory = play_demo_game(roster, cfg)
    print(
        f"[demo] captured {len(trajectory)} frames "
        f"(final winner_seat={trajectory[-1].winner})"
    )
    frames = _downsample(trajectory, cfg.frame_stride)
    print(f"[demo] writing {len(frames)} frames at {cfg.fps} fps to {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_game_gif(frames, str(args.out), fps=cfg.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
