"""Tests for the MCTS tree node (construction, expansion invariants, UCB scoring)."""

from __future__ import annotations

import math

import numpy as np

from territory_takeover.engine import new_game
from territory_takeover.search.mcts.node import MCTSNode


def test_node_construction_defaults() -> None:
    state = new_game(board_size=10, num_players=4)
    node = MCTSNode(state)

    assert node.state is state
    assert node.parent is None
    assert node.incoming_action is None
    assert node.player_to_move == state.current_player
    assert node.children == {}
    assert node.untried_actions == []
    assert node.visits == 0
    assert node.total_value.shape == (4,)
    assert node.total_value.dtype == np.float64
    assert bool(np.all(node.total_value == 0.0))
    assert node.terminal is False
    assert node.is_terminal() is False
    assert node.terminal_value is None


def test_node_construction_2_player_value_vector_shape() -> None:
    state = new_game(board_size=10, num_players=2)
    node = MCTSNode(state)
    assert node.total_value.shape == (2,)


def test_node_construction_terminal_flag_mirrors_state_done() -> None:
    state = new_game(board_size=10, num_players=4)
    state.done = True
    node = MCTSNode(state)
    assert node.terminal is True
    assert node.is_terminal() is True


def test_child_add_and_parent_link() -> None:
    root_state = new_game(board_size=10, num_players=4)
    root = MCTSNode(root_state)

    child_state = root_state.copy()
    action = 2
    child = MCTSNode(child_state, parent=root, incoming_action=action)
    root.children[action] = child

    assert child.parent is root
    assert child.incoming_action == action
    assert root.children[action] is child
    assert len(root.children) == 1


def test_is_fully_expanded_lifecycle() -> None:
    state = new_game(board_size=10, num_players=4)
    node = MCTSNode(state)

    # Freshly constructed: untried not populated, no children -> not fully expanded.
    assert node.is_fully_expanded() is False

    # Populate untried and attach one child: still has untried -> not fully expanded.
    node.untried_actions = [0, 1]
    child_a = MCTSNode(state.copy(), parent=node, incoming_action=3)
    node.children[3] = child_a
    assert node.is_fully_expanded() is False

    # Drain untried: now fully expanded.
    node.untried_actions = []
    assert node.is_fully_expanded() is True

    # Edge case: dead end with no legal moves at all. By definition *not*
    # fully expanded; the UCT driver should detect this via `is_terminal`
    # (set by the engine when a player has no legal moves) and stop
    # descending there rather than relying on `is_fully_expanded`.
    dead_end = MCTSNode(state.copy())
    assert dead_end.children == {}
    assert dead_end.untried_actions == []
    assert dead_end.is_fully_expanded() is False


def test_q_value_unvisited_returns_zero() -> None:
    state = new_game(board_size=10, num_players=4)
    node = MCTSNode(state)
    for p in range(4):
        assert node.q_value(p) == 0.0


def test_q_value_visited_matches_manual_mean() -> None:
    state = new_game(board_size=10, num_players=4)
    node = MCTSNode(state)
    node.visits = 4
    node.total_value = np.array([2.0, 0.5, 0.0, 1.5], dtype=np.float64)

    assert node.q_value(0) == 0.5
    assert node.q_value(1) == 0.125
    assert node.q_value(2) == 0.0
    assert node.q_value(3) == 0.375


def test_ucb_score_unvisited_child_returns_inf() -> None:
    state = new_game(board_size=10, num_players=4)
    root = MCTSNode(state)
    root.visits = 10

    child = MCTSNode(state.copy(), parent=root, incoming_action=0)
    root.children[0] = child

    assert child.visits == 0
    assert child.ucb_score(player_id=0, c=1.4) == math.inf


def test_ucb_score_hand_computed() -> None:
    state = new_game(board_size=10, num_players=4)
    root = MCTSNode(state)
    root.visits = 100

    child = MCTSNode(state.copy(), parent=root, incoming_action=0)
    root.children[0] = child
    child.visits = 10
    child.total_value = np.array([4.0, 0.0, 0.0, 0.0], dtype=np.float64)

    c = math.sqrt(2)
    expected = 0.4 + c * math.sqrt(math.log(100) / 10)
    actual = child.ucb_score(player_id=0, c=c)

    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_ucb_score_uses_parent_player_perspective() -> None:
    # Same child/parent but scored from two different perspectives; mean
    # differs per player, exploration term is identical.
    state = new_game(board_size=10, num_players=4)
    root = MCTSNode(state)
    root.visits = 25

    child = MCTSNode(state.copy(), parent=root, incoming_action=1)
    root.children[1] = child
    child.visits = 5
    child.total_value = np.array([1.0, 3.0, 0.0, 0.0], dtype=np.float64)

    c = 1.0
    exploration = c * math.sqrt(math.log(25) / 5)
    assert math.isclose(child.ucb_score(0, c), 0.2 + exploration, rel_tol=1e-12)
    assert math.isclose(child.ucb_score(1, c), 0.6 + exploration, rel_tol=1e-12)


def test_ucb_score_root_fallback_returns_mean() -> None:
    # Scoring a parentless node: no exploration bonus, just the mean.
    state = new_game(board_size=10, num_players=4)
    root = MCTSNode(state)
    root.visits = 5
    root.total_value = np.array([2.5, 0.0, 0.0, 0.0], dtype=np.float64)

    assert root.parent is None
    assert root.ucb_score(player_id=0, c=1.4) == 0.5
