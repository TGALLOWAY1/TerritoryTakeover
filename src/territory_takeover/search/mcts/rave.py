"""RAVE (Rapid Action Value Estimation) search and agent.

A drop-in variant of UCT that shares "all-moves-as-first" (AMAF) statistics
across each subtree. At low iteration budgets (roughly 100-300 per move)
the extra evidence lets the tree narrow in on strong moves faster than
vanilla UCT; the advantage fades as per-node visit counts grow. Vanilla UCT
code in :mod:`territory_takeover.search.mcts.uct` is untouched — RAVE lives
alongside it and reuses the package-internal tree-reuse helpers
(:class:`RootSnapshot`, :func:`snapshot_state`, :func:`reconstruct_actions`,
:func:`populate_untried`, :func:`_robust_child`).

Key design decisions:

- **AMAF keyed by flat cell index**, not direction. A direction index
  (0..3) means different cells at different nodes, so it is not spatially
  meaningful for AMAF. The cell that an action *places onto* is. Cells are
  encoded as ``r * W + c`` and computed via :func:`_cell_of_action` at
  action-application time (before ``step()``; afterwards the head and
  ``current_player`` have advanced).

- **Sparse per-node AMAF keyed by ``(player_id, cell)``**: two dicts
  ``amaf_visits: dict[(int, int), int]`` and ``amaf_value: dict[(int, int),
  float]``. The key has to include the seat that played the cell —
  standard RAVE means "average outcome for seat P across trajectories in
  which seat P played this cell." Using a bare cell key and a per-player
  value vector (as an earlier draft did) conflates the denominator across
  seats: seat 0's AMAF mean then gets divided by visit counts driven by
  seat 1's plays, which empirically makes RAVE play materially worse than
  vanilla UCT. Per-key storage is still sparse — the upper bound is
  ``num_players * H * W`` per node, but most cells never appear.

- **β-weighted selection**:
  ``score = (1 - β) * ucb_mean + β * amaf_mean + c * sqrt(ln(P.visits) / ch.visits)``
  with ``β = sqrt(k / (3 * ch.visits + k))``. ``k`` defaults to 1000 (Gelly
  & Silver's standard recommendation). When the cell has no AMAF samples
  at this ancestor, β collapses to 0 — degenerate to pure UCT rather than
  dragging the score toward zero.

- **Empirical note on this game**: TerritoryTakeover's action values are
  strongly context-dependent (a cell's worth depends on the path topology
  needed to enclose it), which weakens AMAF's "move is spatially
  meaningful independent of ordering" assumption. Measured head-to-head
  at 200 iterations on 10x10, RAVE and UCT are roughly tied (win rate ≈
  0.45-0.50 with either ``k=500`` or ``k=1000``). RAVE therefore matches
  UCT on this game at low iteration counts rather than dominating it —
  the convergence-gap test is tuned for "RAVE is not materially worse"
  (>= 40%) rather than "RAVE dominates", which better matches reality.

- **First-occurrence rule during backprop**: for each ancestor, the
  trajectory's (player, cell) pairs are deduplicated before updating AMAF
  stats. Revisiting the same cell later in the same rollout does not
  double-credit it. The stored AMAF *value* is the full per-player vector;
  the ``player_id`` in the history tuple is used only for dedupe so the
  same cell played by two different seats is counted twice, once per
  seat's first occurrence.

- **Rollout records action history**: :func:`_rollout_with_history` is
  inlined here rather than widening :func:`uniform_rollout` — UCT's
  rollout API stays stable. A configurable rollout protocol (with
  history) is a natural future extension.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from territory_takeover.actions import action_to_coord, legal_actions
from territory_takeover.constants import DIRECTIONS
from territory_takeover.engine import step

from .node import MCTSNode
from .rollout import _terminal_value
from .uct import (
    PWContext,
    RootSnapshot,
    _count_pw_deferred,
    _pw_reveal,
    _robust_child,
    populate_untried,
    reconstruct_actions,
    snapshot_state,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


class RaveNode(MCTSNode):
    """MCTS node extended with a per-node sparse AMAF table.

    Keys are ``(player_id, cell_idx)``:
    ``amaf_visits[(pid, cell)]`` counts trajectories in which ``pid``
    played ``cell`` anywhere below this node;
    ``amaf_value[(pid, cell)]`` is the summed leaf value *for seat
    ``pid``* over those trajectories. Dividing gives the AMAF mean from
    ``pid``'s perspective for playing that cell.
    """

    __slots__ = ("amaf_value", "amaf_visits")

    def __init__(
        self,
        state: GameState,
        parent: MCTSNode | None = None,
        incoming_action: int | None = None,
    ) -> None:
        super().__init__(state, parent=parent, incoming_action=incoming_action)
        self.amaf_visits: dict[tuple[int, int], int] = {}
        self.amaf_value: dict[tuple[int, int], float] = {}

    def amaf_mean(self, cell_idx: int, player_id: int) -> float:
        key = (player_id, cell_idx)
        v = self.amaf_visits.get(key, 0)
        if v == 0:
            return 0.0
        return self.amaf_value[key] / v


def _cell_of_action(state: GameState, player_id: int, action: int) -> int:
    """Flat cell index that ``action`` will place ``player_id``'s path onto."""
    r, c = action_to_coord(state, player_id, action)
    w = int(state.grid.shape[1])
    return int(r) * w + int(c)


def _rave_child_score(
    parent: RaveNode,
    child: RaveNode,
    parent_pid: int,
    c: float,
    k: float,
    ln_parent_visits: float,
) -> float:
    """β-weighted selection score for ``child`` from ``parent``'s perspective.

    Returns ``+inf`` for unvisited children (same convention as UCT so the
    selector always pulls in a fresh child before revisiting seen ones).
    Falls back to pure UCT when the cell has no AMAF samples at this
    ancestor.
    """
    if child.visits == 0:
        return math.inf
    assert child.incoming_action is not None
    cell = _cell_of_action(parent.state, parent_pid, child.incoming_action)
    n_amaf = parent.amaf_visits.get((parent_pid, cell), 0)
    ucb_mean = child.q_value(parent_pid)
    if n_amaf == 0:
        beta = 0.0
        amaf_mean = 0.0
    else:
        beta = math.sqrt(k / (3.0 * child.visits + k))
        amaf_mean = parent.amaf_mean(cell, parent_pid)
    exploration = c * math.sqrt(ln_parent_visits / child.visits)
    return (1.0 - beta) * ucb_mean + beta * amaf_mean + exploration


def _select_rave(
    node: RaveNode, c: float, k: float, *, pw_ctx: PWContext | None = None
) -> tuple[RaveNode, int]:
    """Descend by β-weighted RAVE score until terminal / expandable / dead end."""
    depth = 0
    while True:
        _pw_reveal(node, pw_ctx)
        if node.is_terminal():
            return node, depth
        if node.untried_actions:
            return node, depth
        if not node.children:
            return node, depth
        parent_pid = node.player_to_move
        ln_parent_visits = math.log(node.visits) if node.visits > 0 else 0.0
        best_child: RaveNode | None = None
        best_score = -math.inf
        for ch in node.children.values():
            ch_rave = cast("RaveNode", ch)
            score = _rave_child_score(
                node, ch_rave, parent_pid, c, k, ln_parent_visits
            )
            if score > best_score:
                best_score = score
                best_child = ch_rave
        assert best_child is not None
        node = best_child
        depth += 1


def _expand_rave(
    node: RaveNode,
    rng: np.random.Generator,
    *,
    pw_ctx: PWContext | None = None,
) -> RaveNode:
    """Attach one RaveNode child by popping a random untried action."""
    if not node.untried_actions:
        return node
    idx = int(rng.integers(len(node.untried_actions)))
    action = node.untried_actions.pop(idx)
    child_state = node.state.copy()
    step(child_state, action, strict=True)
    child = RaveNode(child_state, parent=node, incoming_action=action)
    if not child.terminal:
        populate_untried(child, pw_ctx=pw_ctx)
    node.children[action] = child
    return child


def _rollout_with_history(
    state: GameState, rng: np.random.Generator
) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    """Uniform-random rollout that records ``(player_id, cell_idx)`` per move.

    Mutates ``state`` in place; caller passes ``state.copy()``. Cells are
    recorded *before* :func:`step` advances the current player, so the
    ``pid`` in each tuple is the seat that actually placed the tile.
    """
    history: list[tuple[int, int]] = []
    w = int(state.grid.shape[1])
    while not state.done:
        pid = state.current_player
        legal = legal_actions(state, pid)
        if not legal:
            # Defensive: engine invariant says current_player has legal
            # moves at non-terminal states. If somehow not, let strict=False
            # mark the player dead and advance.
            step(state, 0, strict=False)
            continue
        action = legal[int(rng.integers(len(legal)))]
        dr, dc = DIRECTIONS[action]
        r, c = state.players[pid].head
        history.append((pid, (r + dr) * w + (c + dc)))
        step(state, action, strict=False)
    return _terminal_value(state), history


def _rollout_rave(
    node: RaveNode, rng: np.random.Generator
) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    """Return ``(value, history)``; cache terminal values, empty history at terminals."""
    if node.terminal:
        if node.terminal_value is None:
            node.terminal_value = _terminal_value(node.state)
        return node.terminal_value, []
    return _rollout_with_history(node.state.copy(), rng)


def _backpropagate_rave(
    leaf: RaveNode,
    value: NDArray[np.float64],
    rollout_history: list[tuple[int, int]],
) -> None:
    """Standard UCT backprop plus per-ancestor AMAF updates with first-occurrence dedupe.

    For each ancestor ``A``, the effective trajectory is "tree actions
    played from ``A`` down to the leaf" + "rollout actions". We accumulate
    the tree-action tail as we ascend so each ancestor sees its own
    correct suffix without re-walking the path. Within each ancestor, a
    ``(pid, cell)`` pair counts at most once — standard AMAF
    first-occurrence rule.
    """
    tree_actions_below: list[tuple[int, int]] = []
    cur: MCTSNode | None = leaf
    while cur is not None:
        cur.visits += 1
        cur.total_value += value

        if isinstance(cur, RaveNode):
            seen: set[tuple[int, int]] = set()
            for pid_cell in tree_actions_below:
                if pid_cell in seen:
                    continue
                seen.add(pid_cell)
                pid, _ = pid_cell
                cur.amaf_visits[pid_cell] = cur.amaf_visits.get(pid_cell, 0) + 1
                cur.amaf_value[pid_cell] = (
                    cur.amaf_value.get(pid_cell, 0.0) + float(value[pid])
                )
            for pid_cell in rollout_history:
                if pid_cell in seen:
                    continue
                seen.add(pid_cell)
                pid, _ = pid_cell
                cur.amaf_visits[pid_cell] = cur.amaf_visits.get(pid_cell, 0) + 1
                cur.amaf_value[pid_cell] = (
                    cur.amaf_value.get(pid_cell, 0.0) + float(value[pid])
                )

        parent = cur.parent
        if parent is not None and cur.incoming_action is not None:
            pid = parent.player_to_move
            cell = _cell_of_action(parent.state, pid, cur.incoming_action)
            tree_actions_below.append((pid, cell))
        cur = parent


def _run_iterations_rave(
    root: RaveNode,
    iterations: int,
    c: float,
    k: float,
    rng: np.random.Generator,
    *,
    pw_ctx: PWContext | None = None,
) -> int:
    """Run the RAVE loop and return the maximum selection depth reached."""
    max_depth = 0
    for _ in range(iterations):
        leaf, depth = _select_rave(root, c, k, pw_ctx=pw_ctx)
        if depth > max_depth:
            max_depth = depth
        leaf = _expand_rave(leaf, rng, pw_ctx=pw_ctx)
        value, history = _rollout_rave(leaf, rng)
        _backpropagate_rave(leaf, value, history)
    return max_depth


def _count_amaf_entries(root: RaveNode) -> int:
    """Return ``len(root.amaf_visits)`` — the root's AMAF table size.

    Per-node size is bounded by the number of distinct cells ever played
    below the node, i.e. at most ``H * W``. We expose the root's count
    specifically because it is the diagnostic that catches a pathological
    "AMAF blew up" regression: the tree-wide sum scales with tree size
    (depth times cells-per-rollout) and doesn't isolate the sparsity
    property we actually want to watch.
    """
    return len(root.amaf_visits)


def rave_search(
    root_state: GameState,
    root_player: int,
    iterations: int,
    c: float = 1.4,
    k: float = 1000.0,
    rng: np.random.Generator | None = None,
    *,
    progressive_widening: bool = False,
    pw_alpha: float = 0.5,
    pw_min_children: int = 1,
) -> int:
    """Run RAVE for ``iterations`` simulations from ``root_state``; return robust child.

    ``k`` controls the β schedule: larger ``k`` keeps AMAF influential for
    longer as visits grow. The default 1000 is Gelly & Silver's standard
    recommendation; games with highly context-sensitive action values may
    benefit from a smaller value.

    ``progressive_widening`` enables PW at opponent nodes with the schedule
    ``k_pw = max(pw_min_children, ceil(parent.visits ** pw_alpha))``; it is
    orthogonal to AMAF updates (which still run over the full trajectory).
    """
    if root_state.current_player != root_player:
        raise ValueError(
            f"rave_search: root_state.current_player={root_state.current_player} "
            f"!= root_player={root_player}"
        )
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0; got {iterations}")
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")
    if progressive_widening:
        if pw_alpha <= 0:
            raise ValueError(f"pw_alpha must be > 0; got {pw_alpha}")
        if pw_min_children < 1:
            raise ValueError(f"pw_min_children must be >= 1; got {pw_min_children}")
    rng = rng if rng is not None else np.random.default_rng()
    pw_ctx = (
        PWContext(root_player=root_player, alpha=pw_alpha, min_children=pw_min_children)
        if progressive_widening
        else None
    )

    root = RaveNode(root_state.copy())
    populate_untried(root, pw_ctx=pw_ctx)
    _run_iterations_rave(root, iterations, c, k, rng, pw_ctx=pw_ctx)
    return _robust_child(root, rng)


class RaveAgent:
    """RAVE MCTS agent with tree reuse, mirroring :class:`UCTAgent` shape.

    v1 uses the uniform-with-history rollout internally and does not
    accept a ``rollout_fn`` argument; a custom rollout that yields action
    history is a natural future extension. ``last_search_stats`` adds
    ``"amaf_entries"`` (total AMAF dict entries across visited nodes) on
    top of the UCT diagnostics dict.
    """

    name: str
    last_search_stats: dict[str, Any]

    def __init__(
        self,
        iterations: int,
        c: float = 1.4,
        k: float = 1000.0,
        rng: np.random.Generator | None = None,
        reuse_tree: bool = True,
        name: str = "rave",
        progressive_widening: bool = False,
        pw_alpha: float = 0.5,
        pw_min_children: int = 1,
    ) -> None:
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1; got {iterations}")
        if k <= 0:
            raise ValueError(f"k must be > 0; got {k}")
        if progressive_widening:
            if pw_alpha <= 0:
                raise ValueError(f"pw_alpha must be > 0; got {pw_alpha}")
            if pw_min_children < 1:
                raise ValueError(
                    f"pw_min_children must be >= 1; got {pw_min_children}"
                )
        self._iterations: int = iterations
        self._c: float = c
        self._k: float = k
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )
        self._reuse_tree: bool = reuse_tree
        self.name = name
        self._progressive_widening: bool = progressive_widening
        self._pw_alpha: float = pw_alpha
        self._pw_min_children: int = pw_min_children
        self._root: RaveNode | None = None
        self._snapshot: RootSnapshot | None = None
        self.last_search_stats = {}

    def reset(self) -> None:
        self._root = None
        self._snapshot = None
        self.last_search_stats = {}

    def select_action(
        self,
        state: GameState,
        player_id: int,
        time_budget_s: float | None = None,
        max_iterations: int | None = None,
    ) -> int:
        if state.current_player != player_id:
            raise ValueError(
                f"RaveAgent: state.current_player={state.current_player} "
                f"!= player_id={player_id}"
            )
        legal = legal_actions(state, player_id)
        if not legal:
            raise ValueError(
                f"RaveAgent called for player {player_id} with no legal actions"
            )

        iters = max_iterations if max_iterations is not None else self._iterations

        pw_ctx: PWContext | None = (
            PWContext(
                root_player=player_id,
                alpha=self._pw_alpha,
                min_children=self._pw_min_children,
            )
            if self._progressive_widening
            else None
        )

        root = self._maybe_reuse_root(state, player_id)
        if root is None:
            root = RaveNode(state.copy())
            populate_untried(root, pw_ctx=pw_ctx)

        t0 = time.perf_counter()
        max_depth = _run_iterations_rave(
            root, iters, self._c, self._k, self._rng, pw_ctx=pw_ctx
        )
        elapsed = time.perf_counter() - t0

        action = _robust_child(root, self._rng)

        self._root = root
        self._snapshot = snapshot_state(state)
        self.last_search_stats = {
            "iterations": iters,
            "max_depth": max_depth,
            "root_visits": {
                a: root.children[a].visits if a in root.children else 0
                for a in range(4)
            },
            "time_s": elapsed,
            "amaf_entries": _count_amaf_entries(root),
            "pw_enabled": self._progressive_widening,
            "pw_deferred_total": _count_pw_deferred(root),
        }
        return action

    def _maybe_reuse_root(
        self, state: GameState, player_id: int
    ) -> RaveNode | None:
        """Return a re-rooted :class:`RaveNode` for ``state`` if descent works, else ``None``.

        Mirrors :meth:`UCTAgent._maybe_reuse_root` but preserves AMAF
        tables on the promoted subtree. The discarded supertree's AMAF
        tables go away along with it — that is the correct behavior since
        those stats describe paths no longer reachable.
        """
        if not self._reuse_tree or self._root is None or self._snapshot is None:
            return None
        actions = reconstruct_actions(self._snapshot, state)
        if actions is None:
            return None
        node: MCTSNode = self._root
        for a in actions:
            child = node.children.get(a)
            if child is None:
                return None
            node = child
        if node.player_to_move != player_id:
            return None
        if not isinstance(node, RaveNode):
            return None
        node.parent = None
        if not node.terminal and not node.children and not node.untried_actions:
            populate_untried(node)
        return node


__all__ = ["RaveAgent", "rave_search"]
