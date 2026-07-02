"""Composed position evaluator.

:class:`LinearEvaluator` sums a dict of feature weights into a per-player
score vector. The underlying per-player features live in
:mod:`territory_takeover.eval.features`; this module plays the role of the
composition layer that MCTS leaf scoring and minimax heuristics can call.

Perf invariant: the Voronoi partition is computed **exactly once** per
``evaluate()`` call and shared across features via :class:`_FeatureCache`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.eval.features import (
    _solo_reachable_empties,
    claiming_mobility,
    head_opponent_distance,
    mobility,
    territory_total,
)
from territory_takeover.eval.voronoi import voronoi_partition

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


@dataclass(slots=True)
class _FeatureCache:
    """Shared intermediates computed once per :meth:`LinearEvaluator.evaluate` call.

    ``partition`` is the multi-source BFS Voronoi owner grid (see
    :func:`territory_takeover.eval.voronoi.voronoi_partition`); ``reachable_counts``
    holds ``int(np.count_nonzero(partition == pid))`` for each ``pid``, indexed
    by ``player_id``.
    """

    partition: NDArray[np.int8]
    reachable_counts: list[int]


def _build_cache(state: GameState) -> _FeatureCache:
    partition = voronoi_partition(state)
    num_players = len(state.players)
    counts = [int(np.count_nonzero(partition == pid)) for pid in range(num_players)]
    return _FeatureCache(partition=partition, reachable_counts=counts)


_FeatFn = Callable[["GameState", int, _FeatureCache], float]


class LinearEvaluator:
    """Linear combination of per-player features.

    ``weights`` keys are feature names; each name must resolve to a bound
    method ``_feat_<name>`` on the class. Unknown keys raise ``ValueError``
    at construction. Entries with weight ``0.0`` are dropped from the hot
    scoring loop (``self.weights`` still retains them as the source of
    truth).

    Dead players are scored with the same features as living ones — under
    the corrected rules death (being walled out / finishing your region) is
    inevitable and keeps all territory, so it must not be treated as a
    catastrophe. A dead player's diminished prospects show up naturally:
    their ``reachable_area`` collapses to their territory and their
    ``choke_pressure`` saturates at 1.

    ``evaluate()`` pays for the Voronoi partition once and scores every
    player against the same cache. ``evaluate_for()`` rebuilds the cache
    per call — prefer ``evaluate()`` when you need more than one player's
    score in the same state.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights: dict[str, float] = dict(weights)
        feat_fns: list[tuple[str, float, _FeatFn]] = []
        unknown: list[str] = []
        for name, weight in self.weights.items():
            fn = getattr(self, f"_feat_{name}", None)
            if fn is None:
                unknown.append(name)
                continue
            if weight == 0.0:
                continue
            feat_fns.append((name, weight, fn))
        if unknown:
            available = sorted(
                attr[len("_feat_") :]
                for attr in dir(self)
                if attr.startswith("_feat_")
            )
            raise ValueError(
                f"Unknown feature key(s): {sorted(unknown)}; available: {available}"
            )
        self._feat_fns: list[tuple[str, float, _FeatFn]] = feat_fns

    def evaluate(self, state: GameState) -> NDArray[np.float64]:
        n = len(state.players)
        scores = np.zeros(n, dtype=np.float64)
        cache = _build_cache(state)
        for p in state.players:
            scores[p.player_id] = self._score(state, p.player_id, cache)
        return scores

    def evaluate_for(self, state: GameState, player_id: int) -> float:
        cache = _build_cache(state)
        return self._score(state, player_id, cache)

    def _score(self, state: GameState, player_id: int, cache: _FeatureCache) -> float:
        total = 0.0
        for _name, weight, fn in self._feat_fns:
            total += weight * fn(state, player_id, cache)
        return total

    def _feat_territory_total(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        return float(territory_total(state, player_id))

    def _feat_reachable_area(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        # Dead players keep their territory but can never claim more; the
        # Voronoi partition only seeds living players, so substitute the
        # territory they hold.
        p = state.players[player_id]
        if not p.alive:
            return float(p.territory_count)
        return float(cache.reachable_counts[player_id])

    def _feat_mobility(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        return float(mobility(state, player_id))

    def _feat_claiming_mobility(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        return float(claiming_mobility(state, player_id))

    def _feat_choke_pressure(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        # Mirrors features.choke_pressure exactly, except the contested reach
        # is read from cache.reachable_counts instead of re-running the
        # Voronoi BFS. A dedicated test pins the two implementations together.
        p = state.players[player_id]
        if not p.alive:
            return 1.0
        if p.head == (-1, -1):
            return 1.0

        solo = _solo_reachable_empties(state, player_id)
        if solo == 0:
            return 1.0

        # reachable_counts includes the player's territory (seeded at
        # distance 0); solo counts EMPTY cells only, so subtract it.
        contested = cache.reachable_counts[player_id] - p.territory_count
        if contested < 0:
            contested = 0
        ratio = contested / solo
        if ratio > 1.0:
            ratio = 1.0
        return 1.0 - ratio

    def _feat_opponent_distance(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        d = head_opponent_distance(state, player_id)
        if math.isinf(d):
            return 0.0
        return float(d)


def default_evaluator() -> LinearEvaluator:
    """Factory returning a :class:`LinearEvaluator` with starter weights.

    These weights are a reasonable initial guess — they will be tuned against
    self-play data later. Treat them as defaults, not tuned targets.
    """
    return LinearEvaluator(
        {
            "territory_total": 1.0,
            "reachable_area": 0.3,
            "claiming_mobility": 0.2,
            "mobility": 0.1,
            "choke_pressure": -0.4,
            "opponent_distance": 0.05,
        }
    )


__all__ = ["LinearEvaluator", "default_evaluator"]
