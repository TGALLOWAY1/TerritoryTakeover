"""End-to-end smoke test for the curriculum trainer.

Drives a two-stage curriculum (6x6/2p -> 8x8/2p) for a handful of steps
and verifies:

- both stages run to promotion,
- weight transfer happens between stages,
- per-stage artifacts land on disk,
- the final checkpoint loads into a fresh net at the last stage's shape.
"""

from __future__ import annotations

from pathlib import Path

import torch

from territory_takeover.rl.alphazero.network import AlphaZeroNet, AZNetConfig
from territory_takeover.rl.curriculum import (
    CurriculumTrainConfig,
    PromotionCriterion,
    Schedule,
    Stage,
    train_curriculum,
)


def _tiny_template() -> AZNetConfig:
    return AZNetConfig(
        board_size=6,
        num_players=2,
        num_res_blocks=1,
        channels=8,
        value_hidden=8,
        head_type="conv",
    )


def _tiny_stage(name: str, board_size: int) -> Stage:
    return Stage(
        name=name,
        board_size=board_size,
        num_players=2,
        puct_iterations=4,
        games_per_iteration=1,
        train_steps_per_iteration=2,
        temperature_moves=2,
        max_half_moves=20,
        promotion=PromotionCriterion(
            max_self_play_steps=10,
            elo_gain_window=5,
            elo_gain_threshold=100.0,
            min_evaluations=5,
        ),
    )


def test_two_stage_curriculum_runs_end_to_end(tmp_path: Path) -> None:
    schedule = Schedule(stages=(_tiny_stage("s0_6x6", 6), _tiny_stage("s1_8x8", 8)))
    results = train_curriculum(
        schedule=schedule,
        train_cfg=CurriculumTrainConfig(
            batch_size=4,
            eval_games_per_check=2,
            iterations_per_eval=1,
        ),
        template=_tiny_template(),
        out_dir=tmp_path,
        seed=0,
    )

    assert len(results) == 2
    assert [r.name for r in results] == ["s0_6x6", "s1_8x8"]
    assert all(r.promoted for r in results)
    assert (tmp_path / "stage_00_s0_6x6" / "net_final.pt").exists()
    assert (tmp_path / "stage_01_s1_8x8" / "net_final.pt").exists()
    assert (tmp_path / "stage_01_s1_8x8" / "transfer_report.json").exists()
    assert (tmp_path / "net_final.pt").exists()
    assert (tmp_path / "first_enclosure.csv").exists()
    assert (tmp_path / "stage_results.json").exists()
    assert (tmp_path / "curriculum_progress.yaml").exists()

    # Final checkpoint should load into a fresh 8x8 net without shape errors.
    final_cfg = AZNetConfig(
        board_size=8,
        num_players=2,
        num_res_blocks=1,
        channels=8,
        value_hidden=8,
        head_type="conv",
    )
    net = AlphaZeroNet(final_cfg)
    state_dict = torch.load(tmp_path / "net_final.pt", map_location="cpu")
    net.load_state_dict(state_dict)


def test_single_stage_promotes_on_step_budget(tmp_path: Path) -> None:
    schedule = Schedule(stages=(_tiny_stage("only", 6),))
    results = train_curriculum(
        schedule=schedule,
        train_cfg=CurriculumTrainConfig(
            batch_size=4,
            eval_games_per_check=2,
            iterations_per_eval=1,
        ),
        template=_tiny_template(),
        out_dir=tmp_path,
        seed=1,
    )
    assert len(results) == 1
    assert results[0].self_play_steps >= 10
