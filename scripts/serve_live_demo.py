"""Serve a live HTML viewer that streams gameplay between two agents.

Starts a small stdlib HTTP server and plays games continuously, pushing each
new frame to the browser as it is produced. Open the printed URL to watch
agents play in real time. Click "Reset" in the UI to restart with a fresh
seed; interrupt the process to stop.

See :mod:`territory_takeover.viz_live` for the server module.

Usage::

    # Default: HeuristicGreedy vs HeuristicGreedy on a 20x20 board.
    python scripts/serve_live_demo.py

    # RAVE vs HeuristicGreedy, 4 frames per second.
    python scripts/serve_live_demo.py --seat0 rave --seat1 greedy --fps 4

    # Pick a port and disable the auto-open browser.
    python scripts/serve_live_demo.py --port 8765 --no-browser
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

import numpy as np

from territory_takeover.search.mcts.rave import RaveAgent
from territory_takeover.search.random_agent import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
)
from territory_takeover.search.registry import STRATEGY_LABELS
from territory_takeover.viz_html import AgentCard
from territory_takeover.viz_live import LiveConfig, LiveServer, play_and_serve

AGENT_KEYS: tuple[str, ...] = ("random", "greedy", "rave")


def _build_agent(key: str, seed: int, name: str, rave_iterations: int) -> object:
    """Construct one agent. Returns ``Agent`` (typed loosely to keep imports flat)."""
    if key == "random":
        return UniformRandomAgent(rng=np.random.default_rng(seed), name=name)
    if key == "greedy":
        return HeuristicGreedyAgent(rng=np.random.default_rng(seed), name=name)
    if key == "rave":
        return RaveAgent(
            iterations=rave_iterations,
            name=name,
            rng=np.random.default_rng(seed),
        )
    raise ValueError(f"Unknown agent key {key!r}; choose from {AGENT_KEYS}")


def _strategy_label_for(agent: object) -> str:
    cls_name = type(agent).__name__
    return STRATEGY_LABELS.get(cls_name, cls_name)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve a live HTML viewer that streams gameplay."
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--board-size", type=int, default=20)
    p.add_argument("--num-players", type=int, default=2, choices=[2, 4])
    p.add_argument("--seat0", choices=AGENT_KEYS, default="greedy")
    p.add_argument("--seat1", choices=AGENT_KEYS, default="greedy")
    p.add_argument("--seat2", choices=AGENT_KEYS, default="greedy")
    p.add_argument("--seat3", choices=AGENT_KEYS, default="random")
    p.add_argument("--rave-iterations", type=int, default=200)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--title", type=str, default="TerritoryTakeover Live")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the default browser.",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Play one game then exit (default: loop forever).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    seat_keys = [args.seat0, args.seat1, args.seat2, args.seat3][: args.num_players]

    def agent_factory(episode_index: int) -> list[object]:
        ss = np.random.SeedSequence([args.seed, episode_index])
        agent_seeds = ss.generate_state(args.num_players, dtype=np.uint32)
        return [
            _build_agent(
                key=key,
                seed=int(agent_seeds[i]),
                name=f"{key}_{i}",
                rave_iterations=args.rave_iterations,
            )
            for i, key in enumerate(seat_keys)
        ]

    def card_factory(agents: list[object]) -> list[AgentCard]:
        cards: list[AgentCard] = []
        for seat, agent in enumerate(agents):
            name_attr = getattr(agent, "name", type(agent).__name__)
            cards.append(
                AgentCard(
                    seat=seat,
                    name=str(name_attr),
                    strategy=_strategy_label_for(agent),
                    elo=None,
                )
            )
        return cards

    server = LiveServer(host=args.host, port=args.port, title=args.title)
    server.start()
    print(f"[live-demo] serving at {server.url}")
    print("[live-demo] press Ctrl-C to stop")

    if not args.no_browser:
        # Defer the open slightly so the server is fully ready.
        threading.Timer(0.4, lambda: webbrowser.open(server.url)).start()

    config = LiveConfig(
        board_size=args.board_size,
        num_players=args.num_players,
        fps=args.fps,
        seed=args.seed,
    )
    try:
        play_and_serve(
            server=server,
            agent_factory=agent_factory,  # type: ignore[arg-type]
            agent_card_factory=card_factory,  # type: ignore[arg-type]
            config=config,
            loop=not args.once,
        )
    except KeyboardInterrupt:
        print("\n[live-demo] stopping…")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
