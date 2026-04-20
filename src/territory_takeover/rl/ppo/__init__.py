"""Phase 3b: PPO with action masking and self-play.

Submodules:

- :mod:`spaces` — grid + scalar observation encoder and logit-level action
  masking helper.
- :mod:`network` — actor-critic CNN with masked policy head.
- :mod:`ppo_core` — rollout buffer, GAE, and the clipped PPO update step.
- :mod:`vec_env` — synchronous vectorized env wrapper that runs ``N`` games
  in lockstep and emits the batched shapes ``RolloutBuffer.add`` consumes.

Self-play, training, and Elo tracking live in later-session modules
(``self_play``, ``train``, ``elo``) and are out of scope here.
"""

from __future__ import annotations

__all__: list[str] = []
