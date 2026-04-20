"""Tests for the curriculum schedule and promotion-decision logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from territory_takeover.rl.curriculum import (
    PromotionCriterion,
    PromotionState,
    Schedule,
    Stage,
    load_schedule_yaml,
)


def _default_criterion(**overrides: object) -> PromotionCriterion:
    params = {
        "max_self_play_steps": 1_000_000,
        "elo_gain_window": 5,
        "elo_gain_threshold": 10.0,
        "min_evaluations": 5,
    }
    params.update(overrides)
    return PromotionCriterion(**params)  # type: ignore[arg-type]


def test_promotion_criterion_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        PromotionCriterion(max_self_play_steps=0)
    with pytest.raises(ValueError):
        PromotionCriterion(max_self_play_steps=1, elo_gain_window=0)
    with pytest.raises(ValueError):
        PromotionCriterion(max_self_play_steps=1, elo_gain_threshold=-1.0)
    with pytest.raises(ValueError):
        PromotionCriterion(max_self_play_steps=1, elo_gain_window=5, min_evaluations=2)


def test_schedule_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Schedule(stages=())


def test_promotion_state_does_not_promote_before_min_evaluations() -> None:
    crit = _default_criterion()
    state = PromotionState(criterion=crit)

    # Four evaluations with identical Elo: still below min_evaluations.
    for _ in range(4):
        state.record_evaluation(100.0)
    assert state.should_promote() is False


def test_promotion_state_fires_on_plateau() -> None:
    crit = _default_criterion(elo_gain_window=5, elo_gain_threshold=10.0, min_evaluations=5)
    state = PromotionState(criterion=crit)

    for elo in (100.0, 102.0, 101.0, 103.0, 100.5):
        state.record_evaluation(elo)
    assert state.should_promote() is True


def test_promotion_state_does_not_fire_on_improving_window() -> None:
    crit = _default_criterion(elo_gain_window=5, elo_gain_threshold=10.0, min_evaluations=5)
    state = PromotionState(criterion=crit)

    for elo in (100.0, 110.0, 120.0, 130.0, 140.0):
        state.record_evaluation(elo)
    assert state.should_promote() is False


def test_promotion_state_fires_on_step_budget_regardless_of_elo() -> None:
    crit = _default_criterion(max_self_play_steps=1000, min_evaluations=5)
    state = PromotionState(criterion=crit)

    # Elo still climbing — would NOT plateau.
    for elo in (100.0, 110.0):
        state.record_evaluation(elo)
    state.record_steps(1001)
    assert state.should_promote() is True


def test_promotion_state_roundtrips_through_dict() -> None:
    crit = _default_criterion()
    state = PromotionState(criterion=crit)
    state.record_steps(500)
    for elo in (10.0, 20.0, 30.0):
        state.record_evaluation(elo)

    payload = state.to_dict()
    restored = PromotionState.from_dict(crit, payload)

    assert restored.stage_steps == 500
    assert restored.num_evaluations == 3
    assert list(restored.elo_window) == [10.0, 20.0, 30.0]


def test_load_schedule_yaml_round_trip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "curriculum.yaml"
    yaml_path.write_text(
        """
curriculum:
  stages:
    - name: s1_10x10_2p
      board_size: 10
      num_players: 2
      puct_iterations: 32
      games_per_iteration: 10
      train_steps_per_iteration: 100
      promotion:
        max_self_play_steps: 200000
        elo_gain_window: 5
        elo_gain_threshold: 10
        min_evaluations: 5
    - name: s2_15x15_4p
      board_size: 15
      num_players: 4
      puct_iterations: 32
      games_per_iteration: 8
      train_steps_per_iteration: 100
      promotion:
        max_self_play_steps: 200000
        elo_gain_window: 5
        elo_gain_threshold: 15
        min_evaluations: 5
"""
    )

    schedule = load_schedule_yaml(yaml_path)
    assert len(schedule.stages) == 2

    s1, s2 = schedule.stages
    assert isinstance(s1, Stage)
    assert s1.name == "s1_10x10_2p"
    assert s1.board_size == 10
    assert s1.num_players == 2
    assert s1.promotion.max_self_play_steps == 200_000
    assert s2.num_players == 4
    assert s2.promotion.elo_gain_threshold == 15.0


def test_load_schedule_yaml_rejects_empty_stages(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("curriculum:\n  stages: []\n")
    with pytest.raises(ValueError):
        load_schedule_yaml(yaml_path)


def test_load_schedule_yaml_rejects_missing_top_key(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("something_else: {}\n")
    with pytest.raises(ValueError):
        load_schedule_yaml(yaml_path)
