"""Record a game to an interactive HTML viewer with live win-probability bars.

Plays a full game between two agents, captures every half-move, and writes a
single self-contained HTML file that can be opened directly in a browser.
The viewer shows the grid, each seat's agent name / strategy label / Elo, and
a live-updating win-probability bar per player.

See :mod:`territory_takeover.viz_html` for the renderer.

Usage::

    # Fast default: RAVE vs HeuristicGreedy on a 20x20 board.
    python scripts/record_html_demo.py --out /tmp/tt_game.html

    # Custom matchup using the agent shorthand keys (random/greedy/rave/alphazero).
    python scripts/record_html_demo.py \\
        --seat0 rave --seat1 random --out /tmp/tt_game.html

    # Use an AlphaZero checkpoint for seat 1 (requires the ``rl_deep`` extra).
    python scripts/record_html_demo.py --seat1 alphazero \\
        --checkpoint docs/phase3d/net_reference.pt --out /tmp/tt_game.html

The ``--win-prob`` flag selects the estimator used to fill the probability bars:
``heuristic`` (default, no torch required) or ``alphazero`` (requires a loaded
AlphaZero checkpoint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.engine import new_game, step
from territory_takeover.eval.heuristic import default_evaluator
from territory_takeover.search.mcts.rave import RaveAgent
from territory_takeover.search.random_agent import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
)
from territory_takeover.search.registry import STRATEGY_LABELS
from territory_takeover.viz_html import (
    AgentCard,
    alphazero_win_probs,
    heuristic_win_probs,
    load_elo_csv,
    save_game_html,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.rl.alphazero.evaluator import NNEvaluator
    from territory_takeover.search.agent import Agent
    from territory_takeover.state import GameState


DEFAULT_CHECKPOINT = Path("docs/phase3d/net_reference.pt")
DEFAULT_ELO_CSV = Path("docs/phase3d/elo_final.csv")

AGENT_KEYS: tuple[str, ...] = ("random", "greedy", "rave", "alphazero")
WIN_PROB_KEYS: tuple[str, ...] = ("heuristic", "alphazero")


def _build_alphazero_agent(
    checkpoint_path: Path,
    board_size: int,
    num_players: int,
    iterations: int,
    seed: int,
    name: str,
) -> Agent:
    """Load the reference AlphaZero checkpoint. Deferred torch import."""
    from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent
    from territory_takeover.rl.alphazero.network import AZNetConfig

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"AlphaZero checkpoint not found at {checkpoint_path}."
        )
    net_cfg = AZNetConfig(
        board_size=board_size,
        num_players=num_players,
        num_res_blocks=2,
        channels=32,
        value_hidden=32,
        head_type="conv",
    )
    return AlphaZeroAgent.from_checkpoint(
        path=str(checkpoint_path),
        cfg=net_cfg,
        iterations=iterations,
        c_puct=1.25,
        device="cpu",
        name=name,
        seed=seed,
    )


def _build_agent(
    key: str,
    seed: int,
    args: argparse.Namespace,
    name_override: str | None = None,
) -> Agent:
    """Build one agent. ``name_override`` wins over the default ``key`` name so
    callers can align the rendered label with an external table (e.g. Elo CSV
    rows like ``curriculum_seed0``).
    """
    name = name_override if name_override else key
    if key == "random":
        return UniformRandomAgent(rng=np.random.default_rng(seed), name=name)
    if key == "greedy":
        return HeuristicGreedyAgent(rng=np.random.default_rng(seed), name=name)
    if key == "rave":
        return RaveAgent(
            iterations=args.rave_iterations,
            name=name,
            rng=np.random.default_rng(seed),
        )
    if key == "alphazero":
        return _build_alphazero_agent(
            checkpoint_path=args.checkpoint,
            board_size=args.board_size,
            num_players=args.num_players,
            iterations=args.az_iterations,
            seed=seed,
            name=name,
        )
    raise ValueError(f"Unknown agent key {key!r}; choose from {AGENT_KEYS}")


def _build_nn_evaluator(args: argparse.Namespace) -> NNEvaluator:
    """Load a standalone :class:`NNEvaluator` for the alphazero win-prob path."""
    from territory_takeover.rl.alphazero.evaluator import NNEvaluator
    from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"--win-prob alphazero requires --checkpoint; "
            f"not found at {args.checkpoint}"
        )
    import torch

    cfg = AZNetConfig(
        board_size=args.board_size,
        num_players=args.num_players,
        num_res_blocks=2,
        channels=32,
        value_hidden=32,
        head_type="conv",
    )
    net = AlphaZeroNet(cfg)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    # Checkpoints in-tree are saved as bare state_dicts (see
    # ``AlphaZeroAgent.from_checkpoint``); some external runs wrap them in a
    # ``{"net": ...}`` dict, so handle both shapes.
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()
    return NNEvaluator(net, device="cpu", batch_size=1)


def _compute_win_probs(
    state: GameState,
    args: argparse.Namespace,
    nn_evaluator: NNEvaluator | None,
) -> NDArray[np.float64]:
    if args.win_prob == "heuristic":
        return heuristic_win_probs(
            state,
            default_evaluator(),
            temperature=args.temperature,
        )
    if args.win_prob == "alphazero":
        if nn_evaluator is None:
            raise RuntimeError("nn_evaluator is None but --win-prob alphazero was set")
        return alphazero_win_probs(state, nn_evaluator, state.current_player)
    raise ValueError(f"Unknown win-prob estimator {args.win_prob!r}")


def _strategy_label_for(agent: Agent) -> str:
    return STRATEGY_LABELS.get(type(agent).__name__, type(agent).__name__)


def _build_agent_cards(
    agents: list[Agent],
    elo_csv: Path | None,
) -> list[AgentCard]:
    ratings: dict[str, float] = {}
    if elo_csv is not None and elo_csv.exists():
        ratings = load_elo_csv(elo_csv)
    cards: list[AgentCard] = []
    for seat, agent in enumerate(agents):
        cards.append(
            AgentCard(
                seat=seat,
                name=agent.name,
                strategy=_strategy_label_for(agent),
                elo=ratings.get(agent.name),
            )
        )
    return cards


def _play_game(
    agents: list[Agent],
    args: argparse.Namespace,
    nn_evaluator: NNEvaluator | None,
) -> tuple[list[GameState], list[NDArray[np.float64]]]:
    ss = np.random.SeedSequence(args.seed)
    game_seed = int(ss.generate_state(1, dtype=np.uint32)[0])
    for a in agents:
        a.reset()

    state = new_game(
        board_size=args.board_size,
        num_players=args.num_players,
        seed=game_seed,
    )
    trajectory: list[GameState] = [state.copy()]
    probs: list[NDArray[np.float64]] = [_compute_win_probs(state, args, nn_evaluator)]

    while not state.done:
        seat = state.current_player
        action = agents[seat].select_action(state, seat)
        step(state, action, strict=True)
        trajectory.append(state.copy())
        probs.append(_compute_win_probs(state, args, nn_evaluator))

    return trajectory, probs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a game to an interactive HTML viewer."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--board-size", type=int, default=20)
    p.add_argument("--num-players", type=int, default=2, choices=[2, 4])
    p.add_argument("--seat0", choices=AGENT_KEYS, default="rave")
    p.add_argument("--seat1", choices=AGENT_KEYS, default="greedy")
    p.add_argument("--seat2", choices=AGENT_KEYS, default="greedy")
    p.add_argument("--seat3", choices=AGENT_KEYS, default="random")
    for seat in range(4):
        p.add_argument(
            f"--name{seat}",
            type=str,
            default=None,
            help=f"Display name for seat {seat} (defaults to the agent key). "
            f"Use the canonical Elo-CSV name to pull in the rating, "
            f"e.g. 'curriculum_seed0'.",
        )
    p.add_argument("--rave-iterations", type=int, default=200)
    p.add_argument("--az-iterations", type=int, default=16)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="AlphaZero checkpoint path (used when any seat is 'alphazero' "
        "or --win-prob is 'alphazero').",
    )
    p.add_argument(
        "--win-prob",
        choices=WIN_PROB_KEYS,
        default="heuristic",
        help="Source of per-player win probabilities.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=5.0,
        help="Softmax temperature for the heuristic win-prob estimator.",
    )
    p.add_argument(
        "--elo-csv",
        type=Path,
        default=DEFAULT_ELO_CSV,
        help="CSV with agent,elo_vs_anchor,anchor columns. Missing rows render as '—'.",
    )
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--title", type=str, default="TerritoryTakeover")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/demo.html"),
        help="Output HTML path.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    seat_keys = [args.seat0, args.seat1, args.seat2, args.seat3][: args.num_players]
    seat_names = [args.name0, args.name1, args.name2, args.name3][: args.num_players]
    ss = np.random.SeedSequence(args.seed)
    agent_seeds = ss.generate_state(args.num_players + 1, dtype=np.uint32)
    agents = [
        _build_agent(
            key,
            seed=int(agent_seeds[i]),
            args=args,
            name_override=seat_names[i],
        )
        for i, key in enumerate(seat_keys)
    ]

    nn_evaluator: NNEvaluator | None = None
    if args.win_prob == "alphazero":
        nn_evaluator = _build_nn_evaluator(args)

    print(
        f"[html-demo] roster={[a.name for a in agents]} "
        f"board={args.board_size}x{args.board_size} seed={args.seed}"
    )
    trajectory, probs = _play_game(agents, args, nn_evaluator)
    print(
        f"[html-demo] captured {len(trajectory)} frames "
        f"(winner_seat={trajectory[-1].winner})"
    )

    cards = _build_agent_cards(agents, elo_csv=args.elo_csv)
    save_game_html(
        trajectory=trajectory,
        agent_cards=cards,
        win_probs_per_frame=probs,
        path=args.out,
        title=args.title,
        fps=args.fps,
    )
    print(f"[html-demo] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
