"""Agent factory for YAML-driven tournaments.

:data:`REGISTRY` maps the class names used in tournament configs to the
actual agent constructors. :class:`AgentSpec` carries a name (the
logical label used in result tables), a class name, and a kwargs dict;
:meth:`AgentSpec.build` resolves a small set of string sentinels (e.g.
``evaluator: "default"``) before instantiating the agent.

The registry lives outside :mod:`scripts` so that tests and internal
callers can construct agents from config data without importing
anything in ``scripts/``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.eval.heuristic import default_evaluator

from .maxn import MaxNAgent, ParanoidAgent
from .mcts.rave import RaveAgent
from .mcts.uct import UCTAgent
from .random_agent import HeuristicGreedyAgent, UniformRandomAgent

if TYPE_CHECKING:
    from .agent import Agent


REGISTRY: dict[str, Callable[..., Agent]] = {
    "UniformRandomAgent": UniformRandomAgent,
    "HeuristicGreedyAgent": HeuristicGreedyAgent,
    "MaxNAgent": MaxNAgent,
    "ParanoidAgent": ParanoidAgent,
    "UCTAgent": UCTAgent,
    "RaveAgent": RaveAgent,
}


# Short human-readable strategy label for each agent class. Keyed by
# ``type(agent).__name__`` so callers can look up a live agent instance without
# reinstantiating. ``AlphaZeroAgent`` lives in ``rl.alphazero.mcts`` and is
# listed here even though it isn't in ``REGISTRY`` (which is restricted to
# tournament-configurable classes) so HTML / dashboard callers can render a
# consistent label. Unknown class names should fall back to the class name
# itself at the call site.
STRATEGY_LABELS: dict[str, str] = {
    "UniformRandomAgent": "random",
    "HeuristicGreedyAgent": "heuristic-greedy",
    "MaxNAgent": "maxn",
    "ParanoidAgent": "paranoid",
    "UCTAgent": "uct",
    "RaveAgent": "rave",
    "AlphaZeroAgent": "alphazero",
}


def _resolve_evaluator(value: object) -> object:
    """Resolve ``evaluator: "default"`` sentinel to a live ``LinearEvaluator``."""
    if value == "default":
        return default_evaluator()
    return value


_SENTINEL_RESOLVERS: dict[str, Callable[[object], object]] = {
    "evaluator": _resolve_evaluator,
}


@dataclass(frozen=True)
class AgentSpec:
    """A data-only description of an agent constructor call."""

    name: str
    class_name: str
    kwargs: dict[str, object] = field(default_factory=dict)

    def build(self, rng: np.random.Generator | None = None) -> Agent:
        """Instantiate the agent.

        ``rng`` is injected only when the target class accepts an ``rng``
        parameter and the caller did not set one in ``kwargs``. Unknown
        class names raise :class:`KeyError` with the bad name in the
        message.
        """
        if self.class_name not in REGISTRY:
            raise KeyError(
                f"Unknown agent class {self.class_name!r}; known: {sorted(REGISTRY)}"
            )
        cls = REGISTRY[self.class_name]

        resolved: dict[str, object] = {}
        for k, v in self.kwargs.items():
            resolver = _SENTINEL_RESOLVERS.get(k)
            resolved[k] = resolver(v) if resolver is not None else v

        sig_params = inspect.signature(cls).parameters
        if rng is not None and "rng" in sig_params and "rng" not in resolved:
            resolved["rng"] = rng

        agent: Agent = cls(name=self.name, **resolved)
        return agent


__all__ = ["REGISTRY", "STRATEGY_LABELS", "AgentSpec"]
