"""RL evaluation utilities shared across phases.

Currently exposes only the Phase 3d Elo module; Phase 3b/3c evaluation
lives in their respective ``eval_*.py`` scripts.
"""

from territory_takeover.rl.eval.elo import (
    GameOutcome,
    PairwiseResult,
    compute_elo,
    outcomes_from_rank,
)

__all__ = [
    "GameOutcome",
    "PairwiseResult",
    "compute_elo",
    "outcomes_from_rank",
]
