"""Tests for engine.step, StepResult, IllegalMoveError, compute_terminal_reward."""

from __future__ import annotations

import numpy as np
import pytest

from territory_takeover.actions import legal_actions
from territory_takeover.constants import EMPTY, OWNED_CODES
from territory_takeover.engine import (
    IllegalMoveError,
    compute_terminal_reward,
    new_game,
    step,
)


def test_claiming_move_rewards_one_and_updates_state() -> None:
    state = new_game(board_size=5, num_players=2)
    result = step(state, 1)  # P0 S -> (1,0), empty
    assert result.reward == 1.0
    assert result.info["claimed_this_turn"] == 1
    assert result.info["player_who_moved"] == 0
    assert not result.info["illegal_move"]
    assert state.players[0].head == (1, 0)
    assert state.players[0].territory_count == 2
    assert state.grid[1, 0] == OWNED_CODES[0]
    assert state.empty_count == 25 - 2 - 1
    assert state.current_player == 1


def test_traversal_move_rewards_zero_and_keeps_territory() -> None:
    state = new_game(board_size=5, num_players=2)
    step(state, 1)  # P0 -> (1,0)
    step(state, 0)  # P1 -> (3,4)
    result = step(state, 0)  # P0 back onto own spawn (0,0): traversal
    assert result.reward == 0.0
    assert result.info["claimed_this_turn"] == 0
    assert not result.info["illegal_move"]
    assert state.players[0].head == (0, 0)
    assert state.players[0].territory_count == 2


def test_illegal_move_is_a_noop_wasted_turn() -> None:
    state = new_game(board_size=5, num_players=2)
    result = step(state, 0)  # P0 N from (0,0): out of bounds
    assert result.info["illegal_move"]
    assert result.reward == 0.0
    assert state.players[0].head == (0, 0)
    assert state.players[0].alive, "illegal moves must not kill under corrected rules"
    assert state.current_player == 1, "turn still passes"
    # Out-of-range action index behaves the same.
    result = step(state, 7)
    assert result.info["illegal_move"]
    assert state.players[1].alive


def test_strict_mode_raises_on_illegal() -> None:
    state = new_game(board_size=5, num_players=2)
    with pytest.raises(IllegalMoveError):
        step(state, 0, strict=True)
    # State untouched by the raised move.
    assert state.players[0].head == (0, 0)
    assert state.current_player == 0


def test_step_on_finished_game_raises() -> None:
    state = new_game(board_size=5, num_players=2)
    state.done = True
    with pytest.raises(ValueError):
        step(state, 1)


def test_moving_into_opponent_cell_is_illegal() -> None:
    state = new_game(board_size=3, num_players=2, spawn_positions=[(0, 0), (0, 1)])
    result = step(state, 3)  # E into P1's cell
    assert result.info["illegal_move"]
    assert state.players[0].head == (0, 0)


def test_walled_out_player_dies_on_turn_advance() -> None:
    # 3x3, P0 at corner (0,0), P1 owns (0,1),(1,0),(1,1): P0 is sealed.
    state = new_game(board_size=3, num_players=2, spawn_positions=[(0, 0), (1, 1)])
    grid = state.grid
    grid[0, 1] = OWNED_CODES[1]
    grid[1, 0] = OWNED_CODES[1]
    state.players[1].territory_count = 3
    state.empty_count = int((grid == EMPTY).sum())
    # P0 to move but has no legal move; step(0) is illegal -> no-op wasted
    # turn; the turn passes to P1 (who can still claim).
    step(state, 0)
    assert state.current_player == 1
    # After P1 moves, the round-robin scan reaches P0, finds no reachable
    # EMPTY cell, and marks them dead.
    acts = legal_actions(state, 1)
    step(state, acts[0])
    assert not state.players[0].alive
    assert state.players[1].alive
    assert not state.done, "P1 still has empties to claim"


def test_game_runs_until_all_players_dead_and_winner_by_territory() -> None:
    state = new_game(board_size=4, num_players=2)
    rng = np.random.default_rng(0)
    guard = 0
    while not state.done:
        acts = legal_actions(state, state.current_player)
        assert acts, "engine must only hand the turn to players with legal moves"
        step(state, int(rng.choice(acts)))
        guard += 1
        assert guard < 5000
    assert state.alive_count == 0
    assert all(not p.alive for p in state.players)
    scores = [p.territory_count for p in state.players]
    assert sum(scores) + state.empty_count == 16
    if state.winner is None:
        assert scores[0] == scores[1]
    else:
        assert scores[state.winner] == max(scores)


def test_dead_player_keeps_territory_and_can_win() -> None:
    # P0 claims a lot then is sealed; P1 survives longer but owns less.
    state = new_game(board_size=3, num_players=2, spawn_positions=[(0, 0), (2, 2)])
    grid = state.grid
    # Give P0 five cells (top row + (1,0),(1,1)), P1 stays at 1 cell.
    for cell in [(0, 1), (0, 2), (1, 0), (1, 1)]:
        grid[cell] = OWNED_CODES[0]
    state.players[0].territory_count = 5
    state.players[0].head = (1, 1)
    # Remaining empties: (1,2),(2,0),(2,1) — all adjacent to both players.
    state.empty_count = int((grid == EMPTY).sum())
    rng = np.random.default_rng(1)
    guard = 0
    while not state.done:
        acts = legal_actions(state, state.current_player)
        step(state, int(rng.choice(acts)) if acts else 0)
        guard += 1
        assert guard < 200
    scores = [p.territory_count for p in state.players]
    assert state.winner == 0, f"scores={scores}"


def test_compute_terminal_reward_sparse_and_relative() -> None:
    state = new_game(board_size=5, num_players=2)
    state.players[0].territory_count = 10
    state.players[1].territory_count = 4
    state.winner = 0
    assert compute_terminal_reward(state, "sparse") == [1.0, -1.0]
    rel = compute_terminal_reward(state, "relative")
    assert rel[0] == pytest.approx(1.0)
    assert rel[1] == pytest.approx(-1.0)
    state.winner = None
    assert compute_terminal_reward(state, "sparse") == [0.0, 0.0]
    state.players[1].territory_count = 10
    assert compute_terminal_reward(state, "relative") == [0.0, 0.0]
    with pytest.raises(ValueError):
        compute_terminal_reward(state, "nope")


def test_turn_advance_skips_dead_players() -> None:
    state = new_game(board_size=6, num_players=4)
    state.players[1].alive = False
    state.alive_count = 3
    step(state, 1)  # P0 moves
    assert state.current_player == 2, "dead seat 1 must be skipped"
