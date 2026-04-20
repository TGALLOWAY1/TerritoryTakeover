#!/usr/bin/env python
"""Round-robin Elo evaluator for Phase 3d.

Reads a YAML listing a pool of agents (``random``, ``greedy``, ``uct``,
``alphazero``), plays a head-to-head tournament at a fixed
``(board_size, num_players)``, computes Bradley-Terry Elo pinned to a
named anchor, and writes ``elo_final.csv``.

Usage::

    python scripts/compute_elo.py \
        --config configs/phase3d_elo_pool.yaml \
        --games-per-pair 40 \
        --seed 0 \
        --out results/elo_final.csv

Multi-player games are supported by filling all seats with agents from
the pool in rotating order; each configured permutation is played
``games_per_pair`` times. Head-to-head is the common case and happens
when ``num_players == 2``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from territory_takeover.engine import new_game, step
from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent
from territory_takeover.rl.alphazero.network import AZNetConfig
from territory_takeover.rl.eval.elo import (
    GameOutcome,
    compute_elo,
    outcomes_from_rank,
)
from territory_takeover.search.random_agent import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    kind: str
    params: dict[str, Any]


def _load_agent_specs(path: Path) -> list[AgentSpec]:
    raw = yaml.safe_load(path.read_text())
    specs = []
    for entry in raw["agents"]:
        specs.append(
            AgentSpec(
                name=str(entry["name"]),
                kind=str(entry["kind"]),
                params=dict(entry.get("params", {})),
            )
        )
    if not specs:
        raise ValueError(f"{path}: no agents in pool")
    return specs


def _build_agent(spec: AgentSpec, rng: np.random.Generator) -> Any:
    if spec.kind == "random":
        return UniformRandomAgent(rng=rng, name=spec.name)
    if spec.kind == "greedy":
        return HeuristicGreedyAgent(rng=rng, name=spec.name)
    if spec.kind == "alphazero":
        cfg = AZNetConfig(
            board_size=int(spec.params["board_size"]),
            num_players=int(spec.params["num_players"]),
            num_res_blocks=int(spec.params.get("num_res_blocks", 4)),
            channels=int(spec.params.get("channels", 64)),
            value_hidden=int(spec.params.get("value_hidden", 64)),
            scalar_value_head=bool(spec.params.get("scalar_value_head", False)),
            head_type=str(spec.params.get("head_type", "conv")),  # type: ignore[arg-type]
        )
        agent = AlphaZeroAgent.from_checkpoint(
            path=str(spec.params["checkpoint"]),
            cfg=cfg,
            iterations=int(spec.params.get("iterations", 32)),
            c_puct=float(spec.params.get("c_puct", 1.25)),
            device=str(spec.params.get("device", "cpu")),
            name=spec.name,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        return agent
    raise ValueError(f"unknown agent kind: {spec.kind}")


def _rank_from_final_claims(claims: list[int]) -> list[int]:
    """Convert raw claim counts into ranks (1 = best, ties share rank)."""
    order = sorted(range(len(claims)), key=lambda i: -claims[i])
    ranks = [0] * len(claims)
    current_rank = 1
    for idx_in_order, agent_idx in enumerate(order):
        if idx_in_order > 0 and claims[agent_idx] < claims[order[idx_in_order - 1]]:
            current_rank = idx_in_order + 1
        ranks[agent_idx] = current_rank
    return ranks


def _play_one_game(
    agents: list[Any],
    board_size: int,
    spawn_positions: list[tuple[int, int]] | None,
    seed: int | None,
) -> list[int]:
    num_players = len(agents)
    state = new_game(
        board_size=board_size,
        num_players=num_players,
        spawn_positions=spawn_positions,
        seed=seed,
    )
    for a in agents:
        if hasattr(a, "reset"):
            a.reset()
    while not state.done:
        player = state.current_player
        action = agents[player].select_action(state, player)
        step(state, action, strict=False)
    return [p.claimed_count for p in state.players]


def run_round_robin(
    agents: list[AgentSpec],
    board_size: int,
    num_players: int,
    games_per_pair: int,
    seed: int,
    spawn_positions: list[tuple[int, int]] | None = None,
) -> list[GameOutcome]:
    """Play every assignment of ``num_players`` agents from the pool.

    For ``num_players == 2`` this is a pure head-to-head tournament. For
    larger ``num_players`` we iterate :func:`itertools.permutations`
    (without replacement) — each seating is played ``games_per_pair``
    times.
    """
    rng = np.random.default_rng(seed)
    outcomes: list[GameOutcome] = []

    for seating in itertools.permutations(range(len(agents)), num_players):
        seat_specs = [agents[i] for i in seating]
        seat_names = [s.name for s in seat_specs]

        for game_idx in range(games_per_pair):
            built = [
                _build_agent(
                    spec, np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                )
                for spec in seat_specs
            ]
            game_seed = int(rng.integers(0, 2**31 - 1))
            claims = _play_one_game(built, board_size, spawn_positions, game_seed)
            ranks = _rank_from_final_claims(claims)
            outcomes.extend(outcomes_from_rank(seat_names, ranks))
            del game_idx  # silence unused-var lints under -Wall linters

    return outcomes


def _write_csv(ratings: dict[str, float], out_path: Path, anchor: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(ratings.items(), key=lambda kv: -kv[1])
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "elo_vs_anchor", "anchor"])
        for name, elo in rows:
            w.writerow([name, f"{elo:.2f}", anchor])


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin Elo evaluator.")
    parser.add_argument("--config", required=True, help="YAML listing the agent pool.")
    parser.add_argument("--board-size", type=int, required=True)
    parser.add_argument("--num-players", type=int, required=True)
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument(
        "--anchor",
        default="random",
        help="Name of the agent to pin at Elo = 0. Must appear in the pool.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    specs = _load_agent_specs(Path(args.config))
    print(f"[elo] agents ({len(specs)}): {[s.name for s in specs]}")
    outcomes = run_round_robin(
        specs,
        board_size=args.board_size,
        num_players=args.num_players,
        games_per_pair=args.games_per_pair,
        seed=args.seed,
    )
    print(f"[elo] {len(outcomes)} pairwise outcomes collected")
    ratings = compute_elo(outcomes, anchor=args.anchor)
    _write_csv(ratings, Path(args.out), args.anchor)
    print(f"[elo] wrote {args.out}")
    for name, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
        print(f"  {name:30s} {elo:+8.2f}")


if __name__ == "__main__":
    main()
