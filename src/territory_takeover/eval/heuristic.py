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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from territory_takeover.constants import DIRECTIONS, EMPTY
from territory_takeover.eval.features import (
    enclosure_potential,
    head_opponent_distance,
    mobility,
    territory_total,
)
from territory_takeover.eval.voronoi import voronoi_partition

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


DEAD_SENTINEL: Final[float] = -1e6


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

    Dead players (``state.players[pid].alive is False``) short-circuit to
    :data:`DEAD_SENTINEL` without evaluating any feature. The sentinel is
    finite so downstream softmax / normalization arithmetic stays well-
    behaved; its magnitude (``-1e6``) is far below any plausible live-player
    score on a 40x40 board.

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
            if not p.alive:
                scores[p.player_id] = DEAD_SENTINEL
                continue
            scores[p.player_id] = self._score(state, p.player_id, cache)
        return scores

    def evaluate_for(self, state: GameState, player_id: int) -> float:
        if not state.players[player_id].alive:
            return DEAD_SENTINEL
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
        return float(cache.reachable_counts[player_id])

    def _feat_mobility(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        return float(mobility(state, player_id))

    def _feat_enclosure_potential_area(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        result = enclosure_potential(state, player_id)
        if result is None:
            return 0.0
        return float(result[1])

    def _feat_choke_pressure(
        self, state: GameState, player_id: int, cache: _FeatureCache
    ) -> float:
        # Mirrors features.choke_pressure exactly, except the contested reach
        # is read from cache.reachable_counts instead of re-running the
        # Voronoi BFS. A dedicated test pins the two implementations together.
        p = state.players[player_id]
        if not p.alive:
            return 1.0
        head = p.head
        if head == (-1, -1):
            return 1.0

        grid = state.grid
        h, w = grid.shape
        hr, hc = head

        visited = np.zeros((h, w), dtype=np.bool_)
        q: deque[tuple[int, int]] = deque()
        for dr, dc in DIRECTIONS:
            nr, nc = hr + dr, hc + dc
            if 0 <= nr < h and 0 <= nc < w and grid.item(nr, nc) == EMPTY:
                visited[nr, nc] = True
                q.append((nr, nc))

        while q:
            r, c = q.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and not visited[nr, nc]
                    and grid.item(nr, nc) == EMPTY
                ):
                    visited[nr, nc] = True
                    q.append((nr, nc))

        solo = int(visited.sum())
        if solo == 0:
            return 1.0

        # reachable_counts includes the head cell; solo counts empty cells
        # only, so subtract one for the comparison.
        contested = cache.reachable_counts[player_id] - 1
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
            "mobility": 0.5,
            "enclosure_potential_area": 0.2,
            "choke_pressure": -0.4,
            "opponent_distance": 0.05,
        }
    )


__all__ = ["DEAD_SENTINEL", "LinearEvaluator", "default_evaluator"]
