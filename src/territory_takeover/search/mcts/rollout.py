"""Rollout / playout policies used by MCTS.

The default :func:`uniform_rollout` simulates claim-biased random play to
the end of the episode and returns the per-player territory vector
normalized to ``[0, 1]`` (see :func:`_terminal_value`). Normalization to
a fixed range is what lets the UCB exploration constant ``c`` stay
meaningful across 20x20, 30x30, and 40x40 boards without retuning.

A rollout function has signature ``(GameState, np.random.Generator) ->
NDArray[np.float64]`` of shape ``(num_players,)``. The caller in
:mod:`territory_takeover.search.mcts.uct` is responsible for passing a
:meth:`GameState.copy` — rollouts mutate their input state in place.

Two non-uniform policies live alongside the default:

- :func:`informed_rollout` — epsilon-greedy soft-scored policy. Each move
  combines a claim bonus (EMPTY targets beat traversal moves), a frontier
  proxy (empty-neighbor count of the target cell), and — in the Voronoi
  variant — a pull toward the centroid of the player's Voronoi region.
- :func:`voronoi_guided_rollout` — heavier policy that recomputes a
  Voronoi partition every ``k`` steps and biases moves toward the
  centroid of the current player's owned region. Slower per step but
  expected to help on long horizons.

:func:`make_rollout` dispatches the three kinds by name so callers
(notably :class:`~territory_takeover.search.mcts.uct.UCTAgent` via its
``rollout_kind`` parameter) don't have to import each function.
"""

from __future__ import annotations

import math
import random as _pyrandom
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.actions import legal_actions
from territory_takeover.constants import DIRECTIONS, EMPTY
from territory_takeover.engine import step
from territory_takeover.eval.voronoi import voronoi_partition
from territory_takeover.rollout import simulate_random_rollout

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


RolloutFn = Callable[["GameState", np.random.Generator], "NDArray[np.float64]"]


# --- informed_rollout hyperparameters ---------------------------------------
# Weights were chosen so the claim bonus dominates (claiming a cell is worth
# a point; traversal is not) while the frontier term gives smooth gradients
# among claiming moves and steers traversal moves back toward open space.
# Divided by `_TAU = 0.5` inside softmax so typical score differences of 1-5
# translate to peaked-but-soft sampling distributions.
_W_CLAIM: float = 3.0
_W_REACH: float = 1.0
_W_VORONOI: float = 0.2
_TAU: float = 0.5
_IDENTICAL_SCORE_EPS: float = 1e-9
_DEFAULT_EPSILON: float = 0.15
_DEFAULT_VORONOI_K: int = 4


def _terminal_value(state: GameState) -> NDArray[np.float64]:
    """Per-player ``territory_count / board_area`` in ``[0, 1]``.

    The grid is square so ``state.grid.size`` is the total number of
    cells (board area). Territory count is the same quantity used by
    :func:`territory_takeover.engine._compute_winner`, so the rollout's
    leaf value is consistent with the engine's notion of "territory".
    """
    area = float(state.grid.size)
    return np.array(
        [p.territory_count / area for p in state.players],
        dtype=np.float64,
    )


def uniform_rollout(
    state: GameState, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Play claim-biased random moves until the game ends; return leaf value.

    Mutates ``state`` in place. Delegates to
    :func:`territory_takeover.rollout.simulate_random_rollout` (uniform
    among claiming moves when any exist, else uniform among traversal
    moves) — a policy that terminates orders of magnitude faster than
    uniform-over-all-legal-moves, which would random-walk over
    already-owned cells.
    """
    seed = int(rng.integers(0, 2**63))
    simulate_random_rollout(state, _pyrandom.Random(seed))
    return _terminal_value(state)


def _score_action(
    state: GameState,
    player_id: int,
    action: int,
    head_r: int,
    head_c: int,
    voronoi_centroid: tuple[float, float] | None,
) -> float:
    """Compute the soft-score of playing ``action`` from the current head.

    Components:

    - a claim bonus when the target cell is EMPTY (claiming scores a point;
      traversal over own territory does not),
    - a frontier proxy: the empty-neighbor count of the target cell, which
      prefers claiming moves that keep options open and steers traversal
      moves back toward open space,
    - an optional Voronoi-centroid pull used only by
      :func:`voronoi_guided_rollout`.

    Under the corrected rules there is no self-trap death — a player can
    always walk back through their own territory — so the steep dead-end
    penalties of the old ruleset are gone: claiming into a one-cell pocket
    is cleanup, not suicide.
    """
    dr, dc = DIRECTIONS[action]
    tr = head_r + dr
    tc = head_c + dc
    grid = state.grid
    h, w = grid.shape

    score = _W_CLAIM if grid.item(tr, tc) == EMPTY else 0.0

    # Count EMPTY 4-neighbors of the candidate target t = (tr, tc).
    empty_neighbors = 0
    if tr > 0 and grid.item(tr - 1, tc) == EMPTY:
        empty_neighbors += 1
    if tr < h - 1 and grid.item(tr + 1, tc) == EMPTY:
        empty_neighbors += 1
    if tc > 0 and grid.item(tr, tc - 1) == EMPTY:
        empty_neighbors += 1
    if tc < w - 1 and grid.item(tr, tc + 1) == EMPTY:
        empty_neighbors += 1
    score += _W_REACH * float(empty_neighbors)

    # Voronoi pull (only when running voronoi_guided_rollout and the player
    # actually owns territory in the current partition).
    if voronoi_centroid is not None:
        cr, cc = voronoi_centroid
        dist_centroid = abs(float(tr) - cr) + abs(float(tc) - cc)
        score -= _W_VORONOI * dist_centroid

    return score


def _sample_softmax(
    rng: np.random.Generator, scores: list[float], tau: float
) -> int:
    """Sample an index from ``scores`` proportional to ``softmax(scores / tau)``.

    Pure-Python implementation. numpy's softmax path was ~15 us per call on
    4-element inputs — overhead dominated by ``np.asarray`` / ``np.exp`` /
    ``rng.choice`` setup. At the rollout call frequency (tens of thousands
    per search) that alone blows the 10 us per-move budget.

    Falls back to uniform sampling when ``max - min`` of the scores is below
    :data:`_IDENTICAL_SCORE_EPS` — this matches the spec's requirement that a
    tied score vector degenerates to uniform instead of pretending to make an
    informed choice.
    """
    n = len(scores)
    if n == 1:
        return 0
    m = max(scores)
    if m - min(scores) < _IDENTICAL_SCORE_EPS:
        return int(rng.integers(n))
    probs = [math.exp((s - m) / tau) for s in scores]
    total = 0.0
    for p in probs:
        total += p
    u = float(rng.random()) * total
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return n - 1


def informed_rollout(
    state: GameState,
    rng: np.random.Generator,
    epsilon: float = _DEFAULT_EPSILON,
) -> NDArray[np.float64]:
    """Epsilon-greedy soft-scored rollout; mutates ``state`` in place.

    With probability ``epsilon`` samples a legal action uniformly; otherwise
    computes :func:`_score_action` per legal move and samples proportional to
    ``softmax(scores / tau)`` with ``tau = 0.5``. On game end returns the
    per-player normalized value vector via :func:`_terminal_value`.
    """
    while not state.done:
        pid = state.current_player
        legal = legal_actions(state, pid)
        if not legal:
            # Defensive: the engine's turn-advance only hands the move to
            # seats that can still claim, so this should be unreachable; a
            # no-op step lets the engine's liveness check clean up.
            step(state, 0, strict=False)
            continue
        if len(legal) == 1:
            step(state, legal[0], strict=False)
            continue
        if float(rng.random()) < epsilon:
            action = legal[int(rng.integers(len(legal)))]
        else:
            head_r, head_c = state.players[pid].head
            scores = [
                _score_action(state, pid, a, head_r, head_c, None) for a in legal
            ]
            action = legal[_sample_softmax(rng, scores, _TAU)]
        step(state, action, strict=False)
    return _terminal_value(state)


def _voronoi_centroid(
    voronoi: NDArray[np.int8], player_id: int
) -> tuple[float, float] | None:
    """Return the mean (row, col) of cells owned by ``player_id`` in ``voronoi``.

    Returns ``None`` when the player owns zero cells. Callers skip the
    Voronoi-pull score component in that case.
    """
    owned = np.argwhere(voronoi == player_id)
    if owned.shape[0] == 0:
        return None
    return float(owned[:, 0].mean()), float(owned[:, 1].mean())


def voronoi_guided_rollout(
    state: GameState,
    rng: np.random.Generator,
    k: int = _DEFAULT_VORONOI_K,
    epsilon: float = _DEFAULT_EPSILON,
) -> NDArray[np.float64]:
    """Softmax rollout with Voronoi-centroid bias refreshed every ``k`` moves.

    Heavier than :func:`informed_rollout` because of the periodic full Voronoi
    partition (~200 us on 40x40); amortized cost with ``k = 4`` is ~50 us per
    move. Exposed so the MCTS harness can compare it against the cheaper
    ``informed`` policy.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    step_counter = 0
    centroids: list[tuple[float, float] | None] = []
    while not state.done:
        pid = state.current_player
        legal = legal_actions(state, pid)
        if not legal:
            step(state, 0, strict=False)
            continue
        if len(legal) == 1:
            step(state, legal[0], strict=False)
            step_counter += 1
            continue
        if not centroids or step_counter % k == 0:
            partition = voronoi_partition(state)
            centroids = [
                _voronoi_centroid(partition, p.player_id) for p in state.players
            ]
        if float(rng.random()) < epsilon:
            action = legal[int(rng.integers(len(legal)))]
        else:
            head_r, head_c = state.players[pid].head
            centroid = centroids[pid] if pid < len(centroids) else None
            scores = [
                _score_action(state, pid, a, head_r, head_c, centroid)
                for a in legal
            ]
            action = legal[_sample_softmax(rng, scores, _TAU)]
        step(state, action, strict=False)
        step_counter += 1
    return _terminal_value(state)


def make_rollout(kind: str) -> RolloutFn:
    """Return the rollout function for ``kind`` (``uniform`` / ``informed`` /
    ``voronoi_guided``).

    Raises :class:`ValueError` on unknown names. The returned callable matches
    the :data:`RolloutFn` signature — :func:`informed_rollout` and
    :func:`voronoi_guided_rollout` have extra keyword defaults that stay at
    their module-level values when invoked via the factory.
    """
    if kind == "uniform":
        return uniform_rollout
    if kind == "informed":
        return informed_rollout
    if kind == "voronoi_guided":
        return voronoi_guided_rollout
    raise ValueError(f"unknown rollout kind: {kind!r}")


__all__ = [
    "RolloutFn",
    "informed_rollout",
    "make_rollout",
    "uniform_rollout",
    "voronoi_guided_rollout",
]
