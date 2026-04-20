"""Phase 3c: AlphaZero-style agent (PUCT MCTS + ResNet policy/value net).

Submodules:

- :mod:`spaces` — Phase 3b observation encoder extended with a fixed
  seat-ordered layout and a turn one-hot channel. The 4-dim value head in
  :mod:`network` requires fixed seat ordering so value[i] is unambiguously
  "expected final score of seat i."
- :mod:`network` — ResNet trunk + masked policy head + per-seat value head.
- :mod:`evaluator` — batched NN inference with an LRU state cache and
  virtual-loss hooks for concurrent PUCT search.
- :mod:`mcts` — PUCT selection, NN-prior expansion, 4-dim value backup,
  Dirichlet noise at the root. Reuses the passive
  :class:`territory_takeover.search.mcts.node.MCTSNode` where possible.

Self-play generation, replay storage, training loop, and the tournament
gating stub live in :mod:`selfplay`, :mod:`replay`, and :mod:`train`.
"""

from __future__ import annotations

__all__: list[str] = []
