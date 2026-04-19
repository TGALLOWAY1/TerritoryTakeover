"""UCT (Upper Confidence bounds applied to Trees) search and agent.

The four classic MCTS phases live as module-private helpers: :func:`_select`
descends the tree by UCB1 from the *parent's* perspective, :func:`_expand`
attaches one previously-unvisited child, :func:`_rollout` plays out to a
terminal state via the configured rollout policy, and
:func:`_backpropagate` walks the parent chain incrementing visits and
adding the per-player value vector. Backed by :class:`MCTSNode` from
:mod:`territory_takeover.search.mcts.node`, which already stores
``total_value`` as an N-dim per-player vector so the same tree is usable
for any seat.

:class:`UCTAgent` wraps :func:`uct_search` into the
:class:`~territory_takeover.search.agent.Agent` protocol. It supports
tree reuse across successive calls in the same game — the standard MCTS
optimization where the subtree below the actually-played joint move is
promoted to the new root, preserving the visit counts already invested
there. Reconstruction of the played action sequence is done from
:attr:`PlayerState.path` deltas (see :func:`reconstruct_actions`); any
anomaly (missing child, player mismatch, illegal move flagged dead)
falls back to rebuilding from scratch — never silent corruption.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from territory_takeover.actions import legal_actions
from territory_takeover.constants import DIRECTIONS
from territory_takeover.engine import step

from .node import MCTSNode
from .rollout import RolloutFn, _score_action, _terminal_value, make_rollout, uniform_rollout

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


@dataclass(frozen=True, slots=True)
class PWContext:
    """Progressive-widening configuration threaded through the search.

    ``root_player`` identifies the seat whose turn it is at the root — PW
    only fires at opponent nodes (where ``node.player_to_move !=
    root_player``). The schedule ``k = max(min_children, ceil(parent.visits
    ** alpha))`` grows with parent.visits, so by the time k >= len(legal)
    every move is exposed and PW becomes a no-op. No separate threshold
    parameter is needed — the formula saturates naturally.

    Package-internal, like :class:`RootSnapshot`: used by both UCT and RAVE
    so it lives here but is intentionally absent from ``__all__``.
    """

    root_player: int
    alpha: float
    min_children: int


def _pw_reveal(node: MCTSNode, pw_ctx: PWContext | None) -> None:
    """Move deferred PW actions back into ``untried_actions`` as k grows.

    No-op when PW is off, the node is the root player's (not an opponent),
    the node has no parent (the root itself), or the reserve is empty.
    Otherwise moves entries from the head of ``pw_reserve`` (already sorted
    descending by heuristic score) into ``untried_actions`` until
    ``len(children) + len(untried_actions)`` reaches
    ``max(min_children, ceil(parent.visits ** alpha))``.
    """
    if pw_ctx is None:
        return
    reserve = node.pw_reserve
    if reserve is None or not reserve:
        return
    parent = node.parent
    if parent is None or node.player_to_move == pw_ctx.root_player:
        return
    k = max(pw_ctx.min_children, math.ceil(parent.visits ** pw_ctx.alpha))
    while reserve and len(node.children) + len(node.untried_actions) < k:
        node.untried_actions.append(reserve.pop(0))


def _select(
    node: MCTSNode, c: float, *, pw_ctx: PWContext | None = None
) -> tuple[MCTSNode, int]:
    """Descend by UCB until a node that is terminal, has untried actions, or is a dead end.

    UCB is evaluated for the *parent's* ``player_to_move`` — every child
    of ``parent`` is scored from the perspective of the seat that will
    pick among them. This is the standard multi-player UCT correction
    (avoids the classic bug of the mover optimizing for the next seat).

    When ``pw_ctx`` is provided, each visited node first has its PW
    reserve checked so newly-eligible actions become expandable within
    the same descent that drove the visit count.

    Returns ``(leaf, depth)`` so the caller can record the maximum
    selection depth reached for instrumentation.
    """
    depth = 0
    while True:
        _pw_reveal(node, pw_ctx)
        if node.is_terminal():
            return node, depth
        if node.untried_actions:
            return node, depth
        if not node.children:
            # Non-terminal node with no children and no untried actions:
            # a dead end (no legal moves). Stop here; rollout from current
            # state. In practice this should be rare since terminal-ness
            # is set by the engine when alive_count <= 1.
            return node, depth
        parent_pid = node.player_to_move
        node = max(
            node.children.values(),
            key=lambda ch: ch.ucb_score(parent_pid, c),
        )
        depth += 1


def _expand(
    node: MCTSNode,
    rng: np.random.Generator,
    *,
    pw_ctx: PWContext | None = None,
) -> MCTSNode:
    """Attach one child by popping a random untried action; no-op if none remain."""
    if not node.untried_actions:
        return node
    idx = int(rng.integers(len(node.untried_actions)))
    action = node.untried_actions.pop(idx)
    child_state = node.state.copy()
    step(child_state, action, strict=True)
    child = MCTSNode(child_state, parent=node, incoming_action=action)
    if not child.terminal:
        populate_untried(child, pw_ctx=pw_ctx)
    node.children[action] = child
    return child


def _rollout(
    node: MCTSNode,
    rollout_fn: RolloutFn,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return a per-player value vector for ``node``; cache terminal evaluations."""
    if node.terminal:
        if node.terminal_value is None:
            node.terminal_value = _terminal_value(node.state)
        return node.terminal_value
    return rollout_fn(node.state.copy(), rng)


def _backpropagate(node: MCTSNode, value: NDArray[np.float64]) -> None:
    """Walk to the root incrementing visits and adding ``value`` at each node."""
    cur: MCTSNode | None = node
    while cur is not None:
        cur.visits += 1
        cur.total_value += value
        cur = cur.parent


def populate_untried(
    node: MCTSNode, *, pw_ctx: PWContext | None = None
) -> None:
    """Initialize ``untried_actions`` from the engine's legal-action enumerator.

    When ``pw_ctx`` is provided and ``node`` is an opponent node with a
    parent (i.e. not the root), apply progressive widening: rank the
    legal actions via :func:`_score_action` and split them into
    ``untried_actions`` (top-k visible) and ``node.pw_reserve`` (deferred).
    k is ``max(min_children, ceil(parent.visits ** alpha))`` clipped to
    ``len(legal)``. Ties in score break by lower action id for
    determinism.

    Package-internal helper; shared with :mod:`territory_takeover.search.mcts.rave`
    so RAVE can reuse the same lazy-expansion semantics. Not part of the public
    API (absent from ``__all__``).
    """
    if node.terminal:
        return
    legal = list(legal_actions(node.state, node.state.current_player))
    parent = node.parent
    if (
        pw_ctx is None
        or parent is None
        or node.player_to_move == pw_ctx.root_player
        or not legal
    ):
        node.untried_actions = legal
        return
    pid = node.player_to_move
    head_r, head_c = node.state.players[pid].head
    scored: list[tuple[float, int, int]] = []
    for a in legal:
        score = _score_action(node.state, pid, a, head_r, head_c, None)
        # Negative action id as the secondary sort key so that when we sort
        # descending overall, ties in ``score`` resolve to ascending action
        # id (deterministic).
        scored.append((score, -a, a))
    scored.sort(reverse=True)
    sorted_actions = [t[2] for t in scored]
    k = max(pw_ctx.min_children, math.ceil(parent.visits ** pw_ctx.alpha))
    k = min(k, len(sorted_actions))
    node.untried_actions = sorted_actions[:k]
    node.pw_reserve = sorted_actions[k:]


def _run_iterations(
    root: MCTSNode,
    iterations: int,
    c: float,
    rollout_fn: RolloutFn,
    rng: np.random.Generator,
    *,
    pw_ctx: PWContext | None = None,
) -> int:
    """Run the MCTS loop and return the maximum selection depth reached."""
    max_depth = 0
    for _ in range(iterations):
        leaf, depth = _select(root, c, pw_ctx=pw_ctx)
        if depth > max_depth:
            max_depth = depth
        leaf = _expand(leaf, rng, pw_ctx=pw_ctx)
        value = _rollout(leaf, rollout_fn, rng)
        _backpropagate(leaf, value)
    return max_depth


def _count_pw_deferred(root: MCTSNode) -> int:
    """Sum of deferred PW actions across root + root's immediate children.

    Diagnostic for how aggressively PW is currently hiding moves at the
    top of the tree. The root itself always reports 0 (it's the player's
    own node), but we include the walk for uniformity with potential
    future rerooting edge cases.
    """
    total = len(root.pw_reserve) if root.pw_reserve is not None else 0
    for child in root.children.values():
        if child.pw_reserve is not None:
            total += len(child.pw_reserve)
    return total


def _robust_child(root: MCTSNode, rng: np.random.Generator) -> int:
    """Return the action of the most-visited child, breaking ties via ``rng``."""
    if not root.children:
        # Degenerate: no expansions happened. Caller's responsibility to avoid.
        raise ValueError("_robust_child called on a root with no children")
    best_visits = max(ch.visits for ch in root.children.values())
    tied = [a for a, ch in root.children.items() if ch.visits == best_visits]
    return int(tied[int(rng.integers(len(tied)))])


def uct_search(
    root_state: GameState,
    root_player: int,
    iterations: int,
    c: float = 1.4,
    rollout_fn: RolloutFn | None = None,
    rng: np.random.Generator | None = None,
    *,
    progressive_widening: bool = False,
    pw_alpha: float = 0.5,
    pw_min_children: int = 1,
) -> int:
    """Run UCT for ``iterations`` simulations from ``root_state``; return robust child.

    ``root_player`` is accepted for API symmetry with the agent and other
    search functions in this package; the loop itself is player-agnostic
    because UCB at each node uses the node's own ``player_to_move``. We
    do require ``root_state.current_player == root_player`` as a sanity
    check — passing a state where it's not your turn is almost always a
    caller bug.

    When ``progressive_widening`` is True, PW is applied at opponent nodes
    (``node.player_to_move != root_player``) using the schedule
    ``k = max(pw_min_children, ceil(parent.visits ** pw_alpha))``.
    """
    if root_state.current_player != root_player:
        raise ValueError(
            f"uct_search: root_state.current_player={root_state.current_player} "
            f"!= root_player={root_player}"
        )
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0; got {iterations}")
    if progressive_widening:
        if pw_alpha <= 0:
            raise ValueError(f"pw_alpha must be > 0; got {pw_alpha}")
        if pw_min_children < 1:
            raise ValueError(f"pw_min_children must be >= 1; got {pw_min_children}")
    rng = rng if rng is not None else np.random.default_rng()
    rollout_fn = rollout_fn if rollout_fn is not None else uniform_rollout
    pw_ctx = (
        PWContext(root_player=root_player, alpha=pw_alpha, min_children=pw_min_children)
        if progressive_widening
        else None
    )

    root = MCTSNode(root_state.copy())
    populate_untried(root, pw_ctx=pw_ctx)
    _run_iterations(root, iterations, c, rollout_fn, rng, pw_ctx=pw_ctx)
    return _robust_child(root, rng)


@dataclass(frozen=True, slots=True)
class RootSnapshot:
    """Minimal record of the root state used to detect descent on the next call.

    All fields are immutable / value-typed so the snapshot is safe to
    keep alongside a mutable :class:`MCTSNode`. ``path_lens`` and
    ``alive`` are tuples (one entry per player) so equality checks are
    cheap.

    Package-internal: shared with :mod:`territory_takeover.search.mcts.rave`
    but intentionally omitted from ``__all__``.
    """

    current_player: int
    turn_number: int
    path_lens: tuple[int, ...]
    heads: tuple[tuple[int, int], ...]
    alive: tuple[bool, ...]


def snapshot_state(state: GameState) -> RootSnapshot:
    return RootSnapshot(
        current_player=state.current_player,
        turn_number=state.turn_number,
        path_lens=tuple(len(p.path) for p in state.players),
        heads=tuple(p.head for p in state.players),
        alive=tuple(p.alive for p in state.players),
    )


def reconstruct_actions(
    snapshot: RootSnapshot, state: GameState
) -> list[int] | None:
    """Reconstruct the action sequence played between ``snapshot`` and ``state``.

    Replays the engine's round-robin: starting from ``snapshot.current_player``
    we walk forward, attributing each path-length increase to the seat
    whose turn it would be at that point. Each new path tail cell minus
    the prior head determines the action via ``DIRECTIONS``. Returns
    ``None`` (caller rebuilds the tree) on any anomaly:

    - a player's path got shorter (impossible without a state mismatch)
    - a delta that doesn't match any direction (corruption)
    - the round-robin can't pick a next mover (stuck before catching up)
    """
    n = len(state.players)
    target_lens = [len(p.path) for p in state.players]
    cur_lens = list(snapshot.path_lens)
    cur_heads = list(snapshot.heads)

    for i in range(n):
        if target_lens[i] < cur_lens[i]:
            return None
        # The cell at index cur_lens[i]-1 in the live path must match the
        # snapshot's recorded head for seat i. If it doesn't, the live
        # state isn't a continuation of the snapshot at all (e.g., a fresh
        # game with permuted spawns) — bail and rebuild.
        if cur_lens[i] >= 1 and state.players[i].path[cur_lens[i] - 1] != cur_heads[i]:
            return None

    actions: list[int] = []
    seat = snapshot.current_player
    # Bound the loop generously so a corrupted snapshot can't spin.
    max_iters = sum(target_lens[i] - cur_lens[i] for i in range(n)) + n + 1

    for _ in range(max_iters):
        if cur_lens == target_lens:
            return actions
        # Try to attribute a move to `seat`. If they have moves left,
        # record one; otherwise advance to the next seat that still
        # owes a move (mirroring the engine's round-robin skip).
        if cur_lens[seat] < target_lens[seat]:
            new_cell = state.players[seat].path[cur_lens[seat]]
            old_r, old_c = cur_heads[seat]
            new_r, new_c = new_cell
            delta = (new_r - old_r, new_c - old_c)
            if delta not in DIRECTIONS:
                return None
            actions.append(DIRECTIONS.index(delta))
            cur_lens[seat] += 1
            cur_heads[seat] = new_cell
        # Advance round-robin to the next seat with remaining moves.
        next_seat = -1
        for off in range(1, n + 1):
            cand = (seat + off) % n
            if cur_lens[cand] < target_lens[cand]:
                next_seat = cand
                break
        if next_seat == -1:
            # Nobody else owes a move; we should be done.
            return actions if cur_lens == target_lens else None
        seat = next_seat

    return None


class UCTAgent:
    """UCT MCTS agent with tree reuse across successive calls in the same game.

    The ``rng`` is threaded through every random choice (action expansion,
    rollout, robust-child tie-break) so seeded runs are fully
    reproducible. ``last_search_stats`` is refreshed each
    :meth:`select_action` call with diagnostics: actual iteration count,
    max selection depth reached, per-action visit counts at the root,
    and wall-clock time spent.
    """

    name: str
    last_search_stats: dict[str, Any]

    def __init__(
        self,
        iterations: int,
        c: float = 1.4,
        rollout_fn: RolloutFn | None = None,
        rollout_kind: str | None = None,
        rng: np.random.Generator | None = None,
        reuse_tree: bool = True,
        name: str = "uct",
        progressive_widening: bool = False,
        pw_alpha: float = 0.5,
        pw_min_children: int = 1,
    ) -> None:
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1; got {iterations}")
        if rollout_fn is not None and rollout_kind is not None:
            raise ValueError(
                "pass either rollout_fn or rollout_kind, not both"
            )
        if progressive_widening:
            if pw_alpha <= 0:
                raise ValueError(f"pw_alpha must be > 0; got {pw_alpha}")
            if pw_min_children < 1:
                raise ValueError(
                    f"pw_min_children must be >= 1; got {pw_min_children}"
                )
        self._iterations: int = iterations
        self._c: float = c
        resolved: RolloutFn
        if rollout_fn is not None:
            resolved = rollout_fn
        elif rollout_kind is not None:
            resolved = make_rollout(rollout_kind)
        else:
            resolved = uniform_rollout
        self._rollout_fn: RolloutFn = resolved
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )
        self._reuse_tree: bool = reuse_tree
        self.name = name
        self._progressive_widening: bool = progressive_widening
        self._pw_alpha: float = pw_alpha
        self._pw_min_children: int = pw_min_children
        self._root: MCTSNode | None = None
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
                f"UCTAgent: state.current_player={state.current_player} "
                f"!= player_id={player_id}"
            )
        legal = legal_actions(state, player_id)
        if not legal:
            raise ValueError(
                f"UCTAgent called for player {player_id} with no legal actions"
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
            root = MCTSNode(state.copy())
            populate_untried(root, pw_ctx=pw_ctx)

        t0 = time.perf_counter()
        max_depth = _run_iterations(
            root, iters, self._c, self._rollout_fn, self._rng, pw_ctx=pw_ctx
        )
        elapsed = time.perf_counter() - t0

        action = _robust_child(root, self._rng)

        self._root = root
        self._snapshot = snapshot_state(state)
        self.last_search_stats = {
            "iterations": iters,
            "max_depth": max_depth,
            "root_visits": {a: root.children[a].visits if a in root.children else 0
                            for a in range(4)},
            "time_s": elapsed,
            "pw_enabled": self._progressive_widening,
            "pw_deferred_total": _count_pw_deferred(root),
        }
        return action

    def _maybe_reuse_root(
        self, state: GameState, player_id: int
    ) -> MCTSNode | None:
        """Return a re-rooted MCTSNode for ``state`` if descent is possible, else None."""
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
        # Detach from the discarded supertree so backprop stops at the new root
        # and the older subtrees can be garbage collected.
        node.parent = None
        if not node.terminal and not node.children and not node.untried_actions:
            populate_untried(node)
        return node


__all__ = ["UCTAgent", "uct_search"]
