"""Phase 3d curriculum-learning package.

Public surface:

- :class:`Stage`, :class:`Schedule`, :class:`PromotionCriterion`,
  :class:`PromotionState` — declarative schedule types (``schedule.py``).
- :func:`load_schedule_yaml` — parse a curriculum YAML into a
  :class:`Schedule`.
- :func:`transfer_weights` — copy a state-dict across stages under a
  ``strict=False`` load, returning the set of mismatched/skipped keys.
- :class:`CurriculumTrainer` — drives per-stage training, checkpointing,
  and promotion decisions on top of ``rl.alphazero.train.train_loop``.
"""

from territory_takeover.rl.curriculum.schedule import (
    PromotionCriterion,
    PromotionState,
    Schedule,
    Stage,
    load_schedule_yaml,
)
from territory_takeover.rl.curriculum.transfer import transfer_weights

__all__ = [
    "PromotionCriterion",
    "PromotionState",
    "Schedule",
    "Stage",
    "load_schedule_yaml",
    "transfer_weights",
]
