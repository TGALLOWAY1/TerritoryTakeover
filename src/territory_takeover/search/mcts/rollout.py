"""Rollout / playout policies used by MCTS.

The default :func:`uniform_rollout` simulates uniformly-random play to
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
  combines a 1-step reachable proxy, an enclosure-closure bonus (trigger
  check + full simulation when the candidate touches own-path), a near-
  trap penalty, and a small pull away from the nearest opponent head.
  Baseline per-decision cost is ~5 us; the enclosure simulation branch
  fires on ~25% of scored actions and pushes amortized cost to ~25 us
  per move on mid-size boards. A flat-bonus alternative (no simulation)
  was tried first and measurably *hurt* playing strength — real claimed-
  count delta is needed to distinguish actual loop closures from moves
  that merely touch own-path.
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
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.actions import legal_actions
from territory_takeover.constants import DIRECTIONS, EMPTY, PATH_CODES
from territory_takeover.engine import step
from territory_takeover.eval.voronoi import voronoi_partition

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


RolloutFn = Callable[["GameState", np.random.Generator], "NDArray[np.float64]"]


# --- informed_rollout hyperparameters ---------------------------------------
# Weights were chosen so the enclosure-closure bonus dominates when it fires
# (that's the whole point of the heuristic) while reach / mobility / opponent
# terms give smooth gradients in ordinary play. Divided by `_TAU = 0.5` inside
# softmax so typical score differences of 1-5 translate to peaked-but-soft
# sampling distributions.
_W_REACH: float = 1.0
_W_ENCLOSE: float = 3.0
_W_VORONOI: float = 0.2
_TAU: float = 0.5
_IDENTICAL_SCORE_EPS: float = 1e-9
_DEFAULT_EPSILON: float = 0.15
_DEFAULT_VORONOI_K: int = 4

# Hard dead-end / near-trap penalties. Empirically chosen: gentler values
# (e.g. a flat -5 on near-trap) produced a rollout that was no stronger than
# uniform in matched-iteration tournaments. Making the dead-end case
# unambiguously dominated and the near-trap case strongly discouraged moves
# MCTS's leaf values closer to a sensible "don't voluntarily trap yourself"
# baseline. The magnitudes exceed any other score component so softmax with
# ``tau = 0.5`` assigns trap moves effectively zero probability.
_TRAP_SCORE_DEADEND: float = -100.0
_TRAP_SCORE_SINGLE_EXIT: float = -10.0


def _terminal_value(state: GameState) -> NDArray[np.float64]:
    """Per-player ``(path_length + claimed_count) / board_area`` in ``[0, 1]``.

    The grid is square so ``state.grid.size`` is the total number of
    cells (board area). Path tiles plus claimed tiles is the same
    quantity used by :func:`territory_takeover.engine._compute_winner`,
    so the rollout's leaf value is consistent with the engine's notion
    of "territory".
    """
    area = float(state.grid.size)
    return np.array(
        [(len(p.path) + p.claimed_count) / area for p in state.players],
        dtype=np.float64,
    )


def uniform_rollout(
    state: GameState, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Play uniformly-random legal moves until the game ends; return leaf value.

    Mutates ``state`` in place. The engine's :func:`_advance_turn` skips
    seats with no legal moves (marking them ``alive = False``) so the
    current player at any non-terminal state is guaranteed to have at
    least one legal action; the empty-``legal`` branch is purely
    defensive.
    """
    while not state.done:
        legal = legal_actions(state, state.current_player)
        # Defensive: engine invariant says current_player always has legal
        # moves at a non-terminal state (_advance_turn skips dead seats).
        # If somehow not, fall back to action 0 with strict=False so the
        # engine marks the player dead and advances rather than raising.
        action = legal[int(rng.integers(len(legal)))] if legal else 0
        step(state, action, strict=False)
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

    Three components survived matched-iteration tournament testing:

    - a steep near-trap penalty (hard-avoid dead-ends, strongly avoid
      single-exit cells, otherwise reward by empty-neighbor count),
    - an enclosure-closure bonus that runs a full ``state.copy() + step()``
      when the candidate is adjacent to a same-player path tile other than
      the current head and reads the real ``claimed_count`` delta (a flat
      bonus without simulation was measurably *worse* than uniform — the
      trigger fires on many moves that touch own-path without enclosing
      anything),
    - an optional Voronoi-centroid pull used only by
      :func:`voronoi_guided_rollout`.

    An opponent-distance bonus was tried and dropped; at the 200-iteration
    / 10x10 benchmark it added variance without improving win rate. The
    enclosure branch fires on ~25% of scored actions under random-ish play
    so amortized cost is ~25 us per move, above the original 10 us design
    target but a clear net win at matched iterations (measured).
    """
    dr, dc = DIRECTIONS[action]
    tr = head_r + dr
    tc = head_c + dc
    grid = state.grid
    h, w = grid.shape

    # Count EMPTY 4-neighbors of the candidate target t = (tr, tc). The
    # previous head at (head_r, head_c) is already a PATH tile on the grid,
    # so the grid.item check naturally excludes it from the empty count.
    empty_neighbors = 0
    if tr > 0 and grid.item(tr - 1, tc) == EMPTY:
        empty_neighbors += 1
    if tr < h - 1 and grid.item(tr + 1, tc) == EMPTY:
        empty_neighbors += 1
    if tc > 0 and grid.item(tr, tc - 1) == EMPTY:
        empty_neighbors += 1
    if tc < w - 1 and grid.item(tr, tc + 1) == EMPTY:
        empty_neighbors += 1

    # Steep trap schedule: a move that traps immediately is essentially
    # disqualified; single-exit moves are strongly discouraged; beyond that,
    # more empty neighbors is linearly better.
    if empty_neighbors == 0:
        score = _TRAP_SCORE_DEADEND
    elif empty_neighbors == 1:
        score = _TRAP_SCORE_SINGLE_EXIT
    else:
        score = _W_REACH * float(empty_neighbors)

    # Enclosure-closure bonus. Cheap trigger check (~4 lookups). When it
    # fires we pay a full ``state.copy() + step()`` to read the true
    # ``claimed_count`` delta and reward only real loop closures.
    if len(state.players[player_id].path) >= 4:
        own_code = PATH_CODES[player_id]
        trigger = False
        if tr > 0:
            nr, nc = tr - 1, tc
            if (nr, nc) != (head_r, head_c) and grid.item(nr, nc) == own_code:
                trigger = True
        if not trigger and tr < h - 1:
            nr, nc = tr + 1, tc
            if (nr, nc) != (head_r, head_c) and grid.item(nr, nc) == own_code:
                trigger = True
        if not trigger and tc > 0:
            nr, nc = tr, tc - 1
            if (nr, nc) != (head_r, head_c) and grid.item(nr, nc) == own_code:
                trigger = True
        if not trigger and tc < w - 1:
            nr, nc = tr, tc + 1
            if (nr, nc) != (head_r, head_c) and grid.item(nr, nc) == own_code:
                trigger = True
        if trigger:
            baseline = state.players[player_id].claimed_count
            succ = state.copy()
            step(succ, action, strict=False)
            delta = succ.players[player_id].claimed_count - baseline
            if delta > 0:
                score += _W_ENCLOSE * float(delta)

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
            # Defensive — mirrors uniform_rollout. The engine should have
            # marked this seat dead already; if not, step(strict=False)
            # will.
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

    Returns ``None`` when the player owns zero cells (early game, pre-any
    claim). Callers skip the Voronoi-pull score component in that case.
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
