#!/usr/bin/env python
"""Evaluate the corrected-rules retrained curriculum agent vs the previous best agents.

Runs focused 2-player head-to-head matches at a configurable board size
(default 10x10, the top of the curriculum schedule) between:

- ``az_new``  — the checkpoint retrained from scratch on the corrected rules.
- ``az_old``  — the legacy reference checkpoint (``docs/phase3d/net_reference.pt``),
  trained on the OLD rules, run through a legacy observation adapter that
  reproduces the old encoder layout (path planes = owned cells, claimed
  planes = zeros) so its weights load and see exactly what the old encoder
  would produce on a corrected-rules state.
- The classical roster: ``random``, ``greedy``, ``uct@N``, ``rave@N``.

Matchups: each AZ agent vs each classical baseline, plus az_new vs az_old.
Every matchup uses the seed-locked harness (`run_match`) with seat swapping.

Output: a markdown report (win rates + Wilson 95% CIs) written to
``docs/experiments/corrected_rules_eval.md`` and echoed to stdout.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

REPORT_PATH = Path("docs/experiments/corrected_rules_eval.md")
OLD_CHECKPOINT = Path("docs/phase3d/net_reference.pt")


def _wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _build_legacy_agent(iterations: int, c_puct: float, seed: int):  # noqa: ANN202
    """Load the old-rules reference checkpoint behind a legacy obs adapter."""
    import torch

    from territory_takeover.rl.alphazero.evaluator import NNEvaluator, state_hash
    from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent
    from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig

    num_players = 2

    def legacy_encode(state, active_player):  # noqa: ANN001, ANN202
        """Reproduce the OLD (3N+2)-channel encoder on a corrected-rules state."""
        n = len(state.players)
        h, w = state.grid.shape
        grid = state.grid
        planes = np.zeros((3 * n + 2, h, w), dtype=np.float32)
        for pid in range(n):
            planes[pid] = (grid == pid + 1).astype(np.float32)  # old PATH planes
            # old CLAIMED planes [n .. 2n-1] stay zero: no claimed codes exist.
        planes[2 * n] = (grid == 0).astype(np.float32)
        head = state.players[active_player].head
        if state.players[active_player].alive and head != (-1, -1):
            planes[2 * n + 1, head[0], head[1]] = 1.0
        planes[2 * n + 2 + active_player] = 1.0
        total = float(h * w)
        scalars = np.zeros(3 + n, dtype=np.float32)
        scalars[0] = min(state.turn_number / total, 1.0)
        # old scalars[1..n] were claimed_count fractions -> exactly 0 now.
        empty = float(np.count_nonzero(grid == 0))
        scalars[1 + n] = 1.0 - empty / total
        scalars[2 + n] = state.players[active_player].territory_count / total
        return planes, scalars

    class _LegacyCfg(AZNetConfig):
        @property
        def grid_in_channels(self) -> int:
            return 3 * self.num_players + 2

    class _LegacyEvaluator(NNEvaluator):
        def evaluate_batch(self, requests):  # noqa: ANN001, ANN202
            results = [None] * len(requests)
            miss_idxs, miss_grids, miss_scalars, miss_masks = [], [], [], []
            for i, (state, active_player, legal_mask) in enumerate(requests):
                key = state_hash(state, active_player)
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    prior, value = cached
                    results[i] = (prior.copy(), value.copy())
                    continue
                grid, scalars = legacy_encode(state, active_player)
                miss_idxs.append(i)
                miss_grids.append(grid)
                miss_scalars.append(scalars)
                miss_masks.append(legal_mask)
            if miss_idxs:
                self._forward_misses(
                    requests, results, miss_idxs, miss_grids, miss_scalars, miss_masks
                )
            return [r for r in results if r is not None]

    cfg = _LegacyCfg(
        board_size=10,
        num_players=num_players,
        num_res_blocks=2,
        channels=32,
        value_hidden=32,
        head_type="conv",
    )
    net = AlphaZeroNet(cfg)
    state_dict = torch.load(str(OLD_CHECKPOINT), map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    agent = AlphaZeroAgent(
        net, iterations=iterations, c_puct=c_puct, name="az_old", seed=seed
    )
    agent.evaluator = _LegacyEvaluator(net)
    return agent


def _build_new_agent(path: Path, iterations: int, c_puct: float, seed: int):  # noqa: ANN202
    from territory_takeover.rl.alphazero.mcts import AlphaZeroAgent
    from territory_takeover.rl.alphazero.network import AZNetConfig

    cfg = AZNetConfig(
        board_size=10,
        num_players=2,
        num_res_blocks=2,
        channels=32,
        value_hidden=32,
        head_type="conv",
    )
    return AlphaZeroAgent.from_checkpoint(
        path=str(path),
        cfg=cfg,
        iterations=iterations,
        c_puct=c_puct,
        device="cpu",
        name="az_new",
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=10)
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--uct-iterations", type=int, default=100)
    parser.add_argument("--rave-iterations", type=int, default=100)
    parser.add_argument("--rave-games", type=int, default=None,
                        help="override games for RAVE pairs (they are slow)")
    parser.add_argument("--az-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-old", action="store_true")
    parser.add_argument("--skip-rave", action="store_true")
    args = parser.parse_args()

    from territory_takeover.search.harness import run_match
    from territory_takeover.search.mcts.rave import RaveAgent
    from territory_takeover.search.mcts.uct import UCTAgent
    from territory_takeover.search.random_agent import (
        HeuristicGreedyAgent,
        UniformRandomAgent,
    )

    ss = np.random.SeedSequence(args.seed)
    seeds = [int(s) for s in ss.generate_state(8, dtype=np.uint32)]

    def classical_roster() -> list:  # noqa: ANN202
        roster = [
            UniformRandomAgent(name="random", rng=np.random.default_rng(seeds[0])),
            HeuristicGreedyAgent(name="greedy", rng=np.random.default_rng(seeds[1])),
            UCTAgent(
                iterations=args.uct_iterations,
                rollout_kind="uniform",
                name=f"uct@{args.uct_iterations}",
                rng=np.random.default_rng(seeds[2]),
            ),
        ]
        if not args.skip_rave:
            roster.append(
                RaveAgent(
                    iterations=args.rave_iterations,
                    name=f"rave@{args.rave_iterations}",
                    rng=np.random.default_rng(seeds[3]),
                )
            )
        return roster

    az_agents = [
        _build_new_agent(
            args.new_checkpoint, args.az_iterations, 1.25, seeds[4]
        )
    ]
    if not args.skip_old:
        az_agents.append(_build_legacy_agent(args.az_iterations, 1.25, seeds[5]))

    rows: list[tuple[str, str, float, int, int]] = []

    def play_pair(a, b, games: int, pair_seed: int) -> None:  # noqa: ANN001
        result = run_match(
            [a, b],
            num_games=games,
            board_size=args.board_size,
            swap_seats=True,
            seed=pair_seed,
            parallel=False,
            num_players=2,
        )
        wins = {a.name: 0.0, b.name: 0.0}
        ties = 0
        for log in result.games:
            if log.winner_seat is None:
                ties += 1
                continue
            wins[log.seat_assignment[log.winner_seat]] += 1
        rows.append((a.name, b.name, wins[a.name], ties, games))
        lo, hi = _wilson(wins[a.name], games)
        print(
            f"{a.name} vs {b.name}: {wins[a.name]:.0f}/{games} wins "
            f"({ties} ties)  wr={wins[a.name] / games:.3f} CI=[{lo:.3f},{hi:.3f}]",
            flush=True,
        )

    pair_idx = 0
    for az in az_agents:
        for baseline in classical_roster():
            games = args.games_per_pair
            if baseline.name.startswith("rave") and args.rave_games is not None:
                games = args.rave_games
            play_pair(az, baseline, games, seeds[6] + pair_idx)
            pair_idx += 1
    if len(az_agents) == 2:
        play_pair(az_agents[0], az_agents[1], args.games_per_pair, seeds[7])

    lines = [
        "# Corrected-rules evaluation: retrained vs previous best agents",
        "",
        f"Board {args.board_size}x{args.board_size}, 2 players, seat-swapped, "
        f"seed {args.seed}. az iterations={args.az_iterations}, "
        f"uct/rave iterations={args.uct_iterations}/{args.rave_iterations}.",
        "",
        "| Agent A | Agent B | A wins | Ties | Games | A win rate | Wilson 95% CI |",
        "|---------|---------|-------:|-----:|------:|-----------:|---------------|",
    ]
    for a_name, b_name, wins, ties, games in rows:
        lo, hi = _wilson(wins, games)
        lines.append(
            f"| {a_name} | {b_name} | {wins:.0f} | {ties} | {games} "
            f"| {wins / games:.3f} | [{lo:.3f}, {hi:.3f}] |"
        )
    lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nreport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
