"""Smoke test for the rl.tabular self-play training loop.

Runs a short 200-episode training job on 8x8 / 2p, verifies that artifacts
are written (config.yaml, episode_log.csv, eval_curves.csv, q_table.pkl),
the Q-table is non-empty, and the epsilon schedule has actually decayed
from eps_start by run-end. Runtime target < 5 s.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from territory_takeover.rl.tabular.config import TrainConfig
from territory_takeover.rl.tabular.q_agent import QConfig, TabularQAgent
from territory_takeover.rl.tabular.reward import RewardConfig
from territory_takeover.rl.tabular.train import train


def test_training_smoke_writes_expected_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(
            board_size=8,
            num_players=2,
            spawn_positions=[[0, 0], [7, 7]],
            num_episodes=200,
            eval_every=100,
            eval_games_vs_random=20,
            eval_games_vs_greedy=20,
            eval_games_vs_uct=0,
            uct_iters=4,
            checkpoint_every=100,
            q=QConfig(
                alpha=0.2,
                gamma=0.95,
                eps_start=1.0,
                eps_end=0.1,
                eps_decay_fraction=0.5,
                total_episodes=200,
            ),
            reward=RewardConfig(),
            seed=0,
            out_dir=tmp,
        )
        out = train(cfg, run_tag="smoke")

        assert (out / "config.yaml").exists()
        assert (out / "episode_log.csv").exists()
        assert (out / "eval_curves.csv").exists()
        assert (out / "q_table.pkl").exists()
        assert (out / "summary.yaml").exists()

        # Reload the final Q-table and confirm it is non-empty.
        agent = TabularQAgent.load(out / "q_table.pkl")
        assert len(agent.q_table) > 0, "Q-table did not accumulate any entries"

        # eval_curves.csv should have at least one row written during training.
        with (out / "eval_curves.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1, f"expected >=1 eval row, got {len(rows)}"
        for row in rows:
            # sanity: win rates are in [0, 1]
            for key in row:
                if key.startswith("win_rate_vs_"):
                    val = float(row[key])
                    assert 0.0 <= val <= 1.0, f"win rate {val} out of range"


def test_training_smoke_epsilon_decays() -> None:
    # Confirm the epsilon schedule actually drops from eps_start to near eps_end
    # over the course of the run.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(
            board_size=8,
            num_players=2,
            spawn_positions=[[0, 0], [7, 7]],
            num_episodes=200,
            eval_every=0,  # skip eval entirely so the test is fast
            eval_games_vs_random=0,
            eval_games_vs_greedy=0,
            eval_games_vs_uct=0,
            checkpoint_every=0,  # no intermediate ckpts
            q=QConfig(
                alpha=0.2,
                gamma=0.95,
                eps_start=1.0,
                eps_end=0.1,
                eps_decay_fraction=0.5,
                total_episodes=200,
            ),
            reward=RewardConfig(),
            seed=1,
            out_dir=tmp,
        )
        out = train(cfg, run_tag="decay")
        # Parse episode_log.csv and check epsilon decayed.
        epsilons: list[float] = []
        with (out / "episode_log.csv").open() as f:
            for row in csv.DictReader(f):
                epsilons.append(float(row["epsilon"]))
        assert len(epsilons) >= 2
        assert epsilons[0] >= 0.9, f"eps_start={epsilons[0]} should be near 1.0"
        # eps_decay_fraction=0.5 over 200 episodes -> decay complete at ep 100.
        assert abs(epsilons[-1] - 0.1) < 1e-6 or epsilons[-1] <= 0.15, (
            f"eps_end={epsilons[-1]} should be near 0.1"
        )


def _run_directory_contents(path: Path) -> set[str]:
    return {p.name for p in path.iterdir()}
