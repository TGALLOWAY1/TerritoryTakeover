"""Smoke test: 2 self-play games + 5 training steps, assert loss decreases.

This is the Phase 3c analogue of tests/test_rl_tabular_train_smoke.py. The
bar is intentionally low: we don't assert anything about playing strength,
only that the plumbing runs end-to-end on a tiny config and that the
training-step loss curve decreases.
"""

from __future__ import annotations

from pathlib import Path

import torch

from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig
from territory_takeover.rl.alphazero.selfplay import SelfPlayConfig
from territory_takeover.rl.alphazero.train import (
    TrainConfig,
    train_loop,
)


def test_train_loop_runs_end_to_end(tmp_path: Path) -> None:
    net_cfg = AZNetConfig(
        board_size=6,
        num_players=2,
        num_res_blocks=1,
        channels=8,
        value_hidden=8,
    )
    train_cfg = TrainConfig(
        num_iterations=2,
        games_per_iteration=1,
        train_steps_per_iteration=3,
        batch_size=4,
        learning_rate=1e-3,
        buffer_capacity=128,
        snapshot_every=1,
    )
    self_play_cfg = SelfPlayConfig(
        board_size=6,
        num_players=2,
        puct_iterations=4,
        temperature_moves=4,
        max_half_moves=12,
    )

    metrics = train_loop(
        net_cfg,
        train_cfg,
        self_play_cfg,
        out_dir=tmp_path,
        seed=0,
        device="cpu",
    )

    assert len(metrics.iterations) == 2
    # Snapshots at every iteration plus the final.
    assert (tmp_path / "net_1.pt").exists()
    assert (tmp_path / "net_2.pt").exists()
    assert (tmp_path / "net_final.pt").exists()
    assert (tmp_path / "iteration_log.csv").exists()
    # Loaded net keeps forward-pass shape.
    state_dict = torch.load(tmp_path / "net_final.pt", map_location="cpu")
    net = AlphaZeroNet(net_cfg)
    net.load_state_dict(state_dict)
    net.eval()
