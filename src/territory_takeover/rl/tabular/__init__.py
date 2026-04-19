"""Tabular Q-learning baseline (Phase 3a).

Submodules:
  - :mod:`state_encoder` — :class:`GameState` → compact discrete key.
  - :mod:`reward` — per-step, death, and terminal-rank reward shaping.
  - :mod:`q_agent` — :class:`TabularQAgent`, implementing the
    :class:`territory_takeover.search.agent.Agent` protocol.
  - :mod:`eval` — local 2-player tournament driver that plumbs
    ``spawn_positions`` through to :func:`new_game` (harness.tournament does
    not).
  - :mod:`train` — self-play training loop with TensorBoard + CSV logging.
  - :mod:`config` — dataclass configs + YAML (de)serialization.
"""

from .state_encoder import (
    NBR_EMPTY,
    NBR_OOB,
    NBR_OPP_CLAIM,
    NBR_OPP_PATH,
    NBR_OWN_CLAIM,
    NBR_OWN_PATH,
    PHASE_EARLY,
    PHASE_END,
    PHASE_LATE,
    PHASE_MID,
    StateKey,
    encode_state,
)

__all__ = [
    "NBR_EMPTY",
    "NBR_OOB",
    "NBR_OPP_CLAIM",
    "NBR_OPP_PATH",
    "NBR_OWN_CLAIM",
    "NBR_OWN_PATH",
    "PHASE_EARLY",
    "PHASE_END",
    "PHASE_LATE",
    "PHASE_MID",
    "StateKey",
    "encode_state",
]
