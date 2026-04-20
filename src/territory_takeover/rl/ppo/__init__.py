"""Phase 3b: PPO with action masking and self-play.

Submodules:

- :mod:`spaces` — grid + scalar observation encoder and logit-level action
  masking helper.
- :mod:`network` — actor-critic CNN with masked policy head.
- :mod:`ppo_core` — rollout buffer, GAE, and the clipped PPO update step.

Self-play, training, and Elo tracking live in later-session modules
(``vec_env``, ``self_play``, ``train``, ``elo``) and are out of scope for this
package skeleton.
"""

from __future__ import annotations

__all__: list[str] = []
