"""Common agent interface for search-based policies.

:class:`Agent` is a structural :class:`typing.Protocol` so tree-search code,
tournaments, and the gym opponent hooks can accept any object that implements
the interface without inheritance. The two required methods are
:meth:`select_action` (produce a move for ``player_id`` in ``state``) and
:meth:`reset` (clear any per-game caches between games); ``name`` is a short
identifier used in logs and tournament reports.

``time_budget_s`` and ``max_iterations`` are part of the contract so MCTS and
other anytime agents can slot in later. Deterministic agents (random, greedy)
ignore both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from territory_takeover.state import GameState


class Agent(Protocol):
    name: str

    def select_action(
        self,
        state: GameState,
        player_id: int,
        time_budget_s: float | None = None,
        max_iterations: int | None = None,
    ) -> int: ...

    def reset(self) -> None: ...


__all__ = ["Agent"]
