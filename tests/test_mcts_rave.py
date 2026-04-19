"""Tests for RAVE search and :class:`RaveAgent`.

Mirrors the structure of ``test_mcts_uct.py``: fast determinism / legality
/ tree-reuse / sparsity checks plus a slower head-to-head tournament that
establishes RAVE's win rate against vanilla UCT at moderate iteration
counts. The 5000-iter convergence-gap test is skipped by default and
documented qualitatively — it takes roughly an hour and serves as
reference behavior rather than a CI gate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from territory_takeover.engine import new_game, step
from territory_takeover.search import (
    RaveAgent,
    UCTAgent,
    UniformRandomAgent,
    play_game,
    tournament,
)
from territory_takeover.search.agent import Agent
from territory_takeover.search.mcts.node import MCTSNode
from territory_takeover.search.mcts.rave import (
    RaveNode,
    _backpropagate_rave,
    _rollout_with_history,
    rave_search,
)


def _record_action_sequence(
    agent_factory: Callable[[], tuple[Agent, Agent]],
    board_size: int,
    seed: int,
    max_turns: int = 5_000,
) -> list[int]:
    """Run a single 2-player game and return the chronological action list."""
    seat0, seat1 = agent_factory()
    seat0.reset()
    seat1.reset()
    state = new_game(board_size=board_size, num_players=2, seed=seed)
    actions: list[int] = []
    while not state.done and state.turn_number < max_turns:
        pid = state.current_player
        agent = seat0 if pid == 0 else seat1
        action = agent.select_action(state, pid)
        actions.append(action)
        step(state, action, strict=True)
    return actions


def test_rollout_with_history_terminates_and_matches_terminal_value() -> None:
    for i in range(5):
        rng = np.random.default_rng(100 + i)
        state = new_game(board_size=10, num_players=2, seed=100 + i)
        h, w = state.grid.shape
        value, history = _rollout_with_history(state, rng)
        assert state.done, f"trial={i} rollout did not terminate"
        assert value.shape == (2,), f"trial={i} wrong shape {value.shape}"
        assert value.dtype == np.float64, f"trial={i} wrong dtype"
        for p in range(2):
            assert 0.0 <= value[p] <= 1.0, (
                f"trial={i} player={p} value out of range: {value[p]}"
            )
        assert len(history) >= 1, f"trial={i} empty rollout history"
        for pid, cell in history:
            assert 0 <= pid < 2, f"trial={i} bad pid {pid}"
            assert 0 <= cell < h * w, f"trial={i} cell {cell} out of range"


def test_rave_search_returns_legal_action() -> None:
    from territory_takeover.actions import legal_actions
    for i in range(5):
        rng = np.random.default_rng(200 + i)
        state = new_game(board_size=10, num_players=2, seed=200 + i)
        action = rave_search(state, 0, iterations=64, rng=rng)
        assert action in legal_actions(state, 0), (
            f"trial={i} action {action} not legal"
        )


def test_rave_deterministic_given_seed() -> None:
    """Same RNG seed → identical action sequences across two independent agents."""

    def factory() -> tuple[Agent, Agent]:
        rave = RaveAgent(
            iterations=64,
            rng=np.random.default_rng(7777),
            reuse_tree=False,
        )
        opp = UniformRandomAgent(rng=np.random.default_rng(8888))
        return rave, opp

    for i in range(3):
        seed = 9000 + i
        a = _record_action_sequence(factory, board_size=10, seed=seed)
        b = _record_action_sequence(factory, board_size=10, seed=seed)
        assert a == b, f"trial={i} RAVE agent was non-deterministic"


def test_rave_agent_plays_full_game_and_stats_populated() -> None:
    for i in range(3):
        agents: list[Agent] = [
            RaveAgent(iterations=32, rng=np.random.default_rng(300 + i)),
            RaveAgent(iterations=32, rng=np.random.default_rng(400 + i)),
        ]
        terminal = play_game(
            agents, board_size=8, num_players=2, seed=500 + i, max_turns=5_000
        )
        assert terminal.done, f"trial={i} game did not terminate"
        for a in agents:
            stats = a.last_search_stats  # type: ignore[attr-defined]
            assert set(stats.keys()) == {
                "iterations",
                "max_depth",
                "root_visits",
                "time_s",
                "amaf_entries",
                "pw_enabled",
                "pw_deferred_total",
            }, f"trial={i} stats keys: {set(stats.keys())}"
            assert stats["amaf_entries"] > 0
            assert set(stats["root_visits"].keys()) == {0, 1, 2, 3}


def test_rave_amaf_entries_sparse_bound_on_40x40() -> None:
    """1000 RAVE iterations on 40x40 should not explode the AMAF dicts."""
    agent = RaveAgent(
        iterations=1000,
        rng=np.random.default_rng(0),
        reuse_tree=False,
    )
    state = new_game(board_size=40, num_players=2, seed=0)
    agent.select_action(state, 0)
    entries = agent.last_search_stats["amaf_entries"]
    assert entries < 10_000, (
        f"amaf_entries={entries} exceeded sparsity bound 10_000"
    )


def test_rave_amaf_first_occurrence_dedupe() -> None:
    """Unit test: duplicate ``(pid, cell)`` pairs count at most once per ancestor."""
    # Build a minimal 3-node chain: root → mid → leaf.
    # We don't need real GameState mechanics because _backpropagate_rave only
    # touches visits, total_value, amaf_*, parent, and incoming_action. But
    # the constructor needs a GameState-like object with .players. Rather
    # than mocking, just use a fresh small game and hand-build the chain.
    state_root = new_game(board_size=6, num_players=2, seed=0)
    root = RaveNode(state_root)

    # mid and leaf share the root's state object for AMAF-update purposes;
    # _backpropagate_rave does read state via _cell_of_action on the parent
    # to build tree_actions_below, so mid.incoming_action must be legal
    # from root.state. Use a known legal direction from p0's spawn.
    from territory_takeover.actions import legal_actions as _legal
    legal = _legal(state_root, 0)
    assert legal, "spawn should have legal moves"
    action_for_mid = legal[0]
    # We don't actually step the state — we're synthesizing a tree purely
    # to exercise backprop bookkeeping.
    mid = RaveNode(state_root, parent=root, incoming_action=action_for_mid)
    root.children[action_for_mid] = mid

    value = np.array([0.6, 0.4], dtype=np.float64)
    rollout_history: list[tuple[int, int]] = [
        (0, 5),
        (1, 5),
        (0, 5),  # duplicate of first (pid=0, cell=5)
        (1, 7),
    ]
    _backpropagate_rave(mid, value.copy(), rollout_history)

    # Root AMAF after dedupe (keys are (pid, cell)):
    #   (0, 5): +1, value[0] = 0.6
    #   (1, 5): +1, value[1] = 0.4
    #   Second (0, 5) deduped out.
    #   (1, 7): +1, value[1] = 0.4
    # Plus mid's incoming action (pid=0, cell=cell_of_mid_action): +1, value[0].
    assert root.amaf_visits[(0, 5)] == 1
    assert root.amaf_visits[(1, 5)] == 1
    assert root.amaf_visits[(1, 7)] == 1
    assert root.amaf_value[(0, 5)] == pytest.approx(0.6)
    assert root.amaf_value[(1, 5)] == pytest.approx(0.4)
    assert root.amaf_value[(1, 7)] == pytest.approx(0.4)

    # Mid AMAF: same rollout-history entries, no tree_actions_below from mid.
    assert mid.amaf_visits[(0, 5)] == 1
    assert mid.amaf_visits[(1, 5)] == 1
    assert mid.amaf_visits[(1, 7)] == 1

    # Visits / total_value updated on both.
    assert root.visits == 1
    assert mid.visits == 1
    np.testing.assert_allclose(root.total_value, value)
    np.testing.assert_allclose(mid.total_value, value)


def test_rave_tree_reuse_descends_and_preserves_amaf() -> None:
    """Promoting the (a, b) subtree keeps its AMAF tables intact."""
    state = new_game(board_size=8, num_players=2, seed=12345)
    iters = 128
    agent = RaveAgent(
        iterations=iters,
        rng=np.random.default_rng(54321),
        reuse_tree=True,
    )

    a = agent.select_action(state, 0)
    root_after_first = agent._root
    assert root_after_first is not None
    assert root_after_first.visits == iters
    a_child = root_after_first.children.get(a)
    assert a_child is not None

    explored_b = sorted(
        a_child.children.items(), key=lambda kv: kv[1].visits, reverse=True
    )
    assert explored_b, "expected expanded grandchild after 128 iters"
    b, grandchild = explored_b[0]
    v_grand_pre = grandchild.visits
    assert v_grand_pre > 0
    assert isinstance(grandchild, RaveNode)
    amaf_entries_pre = len(grandchild.amaf_visits)

    step(state, a, strict=True)
    step(state, b, strict=True)

    agent.select_action(state, 0)
    root_after_second = agent._root
    assert root_after_second is not None
    assert root_after_second.parent is None
    assert root_after_second.player_to_move == 0
    assert root_after_second.visits == v_grand_pre + iters
    # AMAF dict on the promoted subtree must survive the re-root — it can
    # only grow (new iterations add entries), never reset.
    assert len(root_after_second.amaf_visits) >= amaf_entries_pre


def test_rave_search_validates_inputs() -> None:
    state = new_game(board_size=6, num_players=2, seed=0)
    with pytest.raises(ValueError):
        rave_search(state, 1, iterations=10)
    with pytest.raises(ValueError):
        rave_search(state, 0, iterations=-1)
    with pytest.raises(ValueError):
        rave_search(state, 0, iterations=10, k=0.0)
    with pytest.raises(ValueError):
        RaveAgent(iterations=0)
    with pytest.raises(ValueError):
        RaveAgent(iterations=10, k=-1.0)


def test_rave_is_not_materially_worse_than_uct_at_moderate_iters() -> None:
    """RAVE vs UCT at 200 iters on 10x10 should be roughly even (>= 40%).

    TerritoryTakeover's action values are highly context-dependent —
    enclosure outcomes depend on path topology, not just cell placement —
    so AMAF's "permutation-invariant move value" assumption is weak. In
    practice RAVE matches UCT at 200 iters on 10x10 rather than
    dominating it: measured win rates are in the 0.45-0.50 range with
    either ``k=500`` or ``k=1000``. This test gates against regressions
    that would push RAVE materially below UCT (e.g., a broken AMAF
    indexing bug that tanked an earlier implementation to 0.20). Fixed
    seed for reproducibility; runs 20 alternating-seat games (≈2.5
    minutes).
    """
    rave = RaveAgent(iterations=200, rng=np.random.default_rng(2024))
    uct = UCTAgent(iterations=200, rng=np.random.default_rng(2025))
    results = tournament(
        agent_a=rave,
        agent_b=uct,
        num_games=20,
        board_size=10,
        seed=42,
    )
    total = results["wins_a"] + results["wins_b"] + results["ties"]
    assert total == 20
    win_rate = results["wins_a"] / 20
    assert win_rate >= 0.40, (
        f"RAVE win rate {win_rate:.2f} below 0.40 vs UCT "
        f"(wins_a={results['wins_a']}, wins_b={results['wins_b']}, "
        f"ties={results['ties']})"
    )


def test_rave_gap_shrinks_at_high_iters() -> None:
    """At 5000 iters the RAVE-vs-UCT gap should shrink to [0.4, 0.7].

    Skipped by default (≈1h). Qualitative documentation test — when run
    manually, record the observed win rate here so future regressions are
    visible. In-body ``pytest.skip`` rather than a decorator because the
    strict mypy config flags ``@pytest.mark.skip`` as an untyped decorator
    (missing pytest stubs).
    """
    pytest.skip("Slow (≈1h): documents convergence-gap shrinking at 5000 iters.")
    rave = RaveAgent(iterations=5000, rng=np.random.default_rng(3024))
    uct = UCTAgent(iterations=5000, rng=np.random.default_rng(3025))
    results = tournament(
        agent_a=rave, agent_b=uct, num_games=10, board_size=8, seed=42
    )
    total = results["wins_a"] + results["wins_b"] + results["ties"]
    assert total == 10
    win_rate = results["wins_a"] / 10
    assert 0.4 <= win_rate <= 0.7, (
        f"expected gap-shrunk RAVE win rate in [0.4, 0.7], got {win_rate:.2f}"
    )


def test_rave_backprop_updates_mctsnode_like_uct() -> None:
    """A non-RAVE ancestor in the chain should still get visit/value updates.

    Exercises the ``isinstance(cur, RaveNode)`` branch in backprop: an
    :class:`MCTSNode` inserted into the chain must receive the standard
    visit and total-value updates but no AMAF updates (it has no dicts).
    """
    state = new_game(board_size=6, num_players=2, seed=0)
    mixed_root = MCTSNode(state)
    # mid is a RaveNode whose parent is a plain MCTSNode — synthetic, but
    # models the defensive isinstance check in _backpropagate_rave.
    from territory_takeover.actions import legal_actions as _legal
    action = _legal(state, 0)[0]
    mid = RaveNode(state, parent=mixed_root, incoming_action=action)
    mixed_root.children[action] = mid

    value = np.array([0.3, 0.7], dtype=np.float64)
    _backpropagate_rave(mid, value.copy(), [(1, 4)])

    assert mixed_root.visits == 1
    np.testing.assert_allclose(mixed_root.total_value, value)
    assert mid.visits == 1
    # mid should still have AMAF updates since it's a RaveNode.
    assert mid.amaf_visits.get((1, 4)) == 1
