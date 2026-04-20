"""Curriculum schedule: stages, promotion logic, YAML loader.

A :class:`Schedule` is a list of :class:`Stage`s. Each stage pins a
``(board_size, num_players)`` and a per-stage budget of self-play steps.
After each evaluation the trainer calls :meth:`PromotionState.update` with
the latest Elo estimate and current step count; :meth:`should_promote`
returns ``True`` when either:

1. the Elo rolling window has plateaued
   (``max(last K) - min(last K) < elo_gain_threshold``), or
2. the stage's ``max_self_play_steps`` budget is hit.

A minimum number of evaluations is required before plateau can trigger
(``min_evaluations``) so a stage doesn't promote on the first two
identical Elo samples.

This module is intentionally agnostic to the underlying trainer — it only
owns the declarative schedule and the promotion decision.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class PromotionCriterion:
    """When to promote from a stage to the next one."""

    max_self_play_steps: int
    """Hard per-stage budget on self-play steps. Promotes unconditionally
    once this is hit."""

    elo_gain_window: int = 5
    """Number of recent evaluations to inspect for plateau."""

    elo_gain_threshold: float = 10.0
    """Promote when ``max(window) - min(window) < elo_gain_threshold`` and
    the window is full."""

    min_evaluations: int = 5
    """Don't evaluate plateau before this many evals. Guards against
    promoting on the first few identical-by-chance Elo samples."""

    def __post_init__(self) -> None:
        if self.max_self_play_steps <= 0:
            raise ValueError("max_self_play_steps must be positive")
        if self.elo_gain_window <= 0:
            raise ValueError("elo_gain_window must be positive")
        if self.elo_gain_threshold < 0:
            raise ValueError("elo_gain_threshold must be non-negative")
        if self.min_evaluations < self.elo_gain_window:
            # Require at least a full window before plateau can fire.
            raise ValueError("min_evaluations must be >= elo_gain_window")


@dataclass(frozen=True, slots=True)
class Stage:
    """One curriculum stage."""

    name: str
    board_size: int
    num_players: int
    puct_iterations: int
    games_per_iteration: int
    train_steps_per_iteration: int
    promotion: PromotionCriterion
    spawn_positions: list[tuple[int, int]] | None = None
    max_half_moves: int | None = None
    temperature_moves: int = 16
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25


@dataclass(frozen=True, slots=True)
class Schedule:
    """Ordered list of curriculum stages."""

    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        if len(self.stages) == 0:
            raise ValueError("Schedule must contain at least one stage")


@dataclass(slots=True)
class PromotionState:
    """Mutable in-stage state used to decide when to promote.

    Owned by the trainer, not by the schedule. Serialized into
    ``curriculum_progress.yaml`` for resumption.
    """

    criterion: PromotionCriterion
    stage_steps: int = 0
    elo_window: deque[float] = field(init=False)
    num_evaluations: int = 0

    def __post_init__(self) -> None:
        self.elo_window = deque(maxlen=self.criterion.elo_gain_window)

    def record_steps(self, delta: int) -> None:
        if delta < 0:
            raise ValueError("delta must be non-negative")
        self.stage_steps += delta

    def record_evaluation(self, elo: float) -> None:
        self.elo_window.append(float(elo))
        self.num_evaluations += 1

    def should_promote(self) -> bool:
        if self.stage_steps >= self.criterion.max_self_play_steps:
            return True
        if self.num_evaluations < self.criterion.min_evaluations:
            return False
        if len(self.elo_window) < self.criterion.elo_gain_window:
            return False
        spread = max(self.elo_window) - min(self.elo_window)
        return spread < self.criterion.elo_gain_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_steps": int(self.stage_steps),
            "num_evaluations": int(self.num_evaluations),
            "elo_window": list(self.elo_window),
        }

    @classmethod
    def from_dict(cls, criterion: PromotionCriterion, data: dict[str, Any]) -> PromotionState:
        state = cls(criterion=criterion)
        state.stage_steps = int(data.get("stage_steps", 0))
        state.num_evaluations = int(data.get("num_evaluations", 0))
        for elo in data.get("elo_window", []):
            state.elo_window.append(float(elo))
        return state


def _parse_spawns(raw: Any) -> list[tuple[int, int]] | None:
    if raw is None:
        return None
    out: list[tuple[int, int]] = []
    for pair in raw:
        if len(pair) != 2:
            raise ValueError(f"spawn position must be (row, col); got {pair!r}")
        out.append((int(pair[0]), int(pair[1])))
    return out


def _parse_promotion(raw: dict[str, Any]) -> PromotionCriterion:
    return PromotionCriterion(
        max_self_play_steps=int(raw["max_self_play_steps"]),
        elo_gain_window=int(raw.get("elo_gain_window", 5)),
        elo_gain_threshold=float(raw.get("elo_gain_threshold", 10.0)),
        min_evaluations=int(raw.get("min_evaluations", 5)),
    )


def _parse_stage(raw: dict[str, Any]) -> Stage:
    return Stage(
        name=str(raw["name"]),
        board_size=int(raw["board_size"]),
        num_players=int(raw["num_players"]),
        puct_iterations=int(raw["puct_iterations"]),
        games_per_iteration=int(raw["games_per_iteration"]),
        train_steps_per_iteration=int(raw["train_steps_per_iteration"]),
        promotion=_parse_promotion(raw["promotion"]),
        spawn_positions=_parse_spawns(raw.get("spawn_positions")),
        max_half_moves=(
            int(raw["max_half_moves"]) if raw.get("max_half_moves") is not None else None
        ),
        temperature_moves=int(raw.get("temperature_moves", 16)),
        c_puct=float(raw.get("c_puct", 1.25)),
        dirichlet_alpha=float(raw.get("dirichlet_alpha", 0.3)),
        dirichlet_eps=float(raw.get("dirichlet_eps", 0.25)),
    )


def load_schedule_yaml(path: str | Path) -> Schedule:
    """Parse a curriculum YAML file.

    Expected top-level shape::

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
              ...
    """
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "curriculum" not in raw:
        raise ValueError(f"{path}: expected top-level 'curriculum' key")
    stages_raw = raw["curriculum"].get("stages")
    if not stages_raw:
        raise ValueError(f"{path}: curriculum.stages must be non-empty")
    stages = tuple(_parse_stage(s) for s in stages_raw)
    return Schedule(stages=stages)


__all__ = [
    "PromotionCriterion",
    "PromotionState",
    "Schedule",
    "Stage",
    "load_schedule_yaml",
]
