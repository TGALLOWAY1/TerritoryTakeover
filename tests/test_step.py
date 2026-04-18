"""Tests for engine.step, StepResult, IllegalMoveError, compute_terminal_reward."""

from __future__ import annotations

import numpy as np
import pytest

from territory_takeover.actions import legal_actions
from territory_takeover.constants import CLAIMED_CODES, PATH_CODES
from territory_takeover.engine import (
    IllegalMoveError,
    StepResult,
    compute_terminal_reward,
    new_game,
    step,
)
from territory_takeover.state import GameState, PlayerState


def _make_state(
    grid: np.ndarray,
    heads: list[tuple[int, int]],
    *,
    current_player: int = 0,
    paths: list[list[tuple[int, int]]] | None = None,
    claimed: list[int] | None = None,
    alive: list[bool] | None = None,
) -> GameState:
    n = len(heads)
    if paths is None:
        paths = [[h] for h in heads]
    if claimed is None:
        claimed = [0] * n
    if alive is None:
        alive = [True] * n
    players = [
        PlayerState(
            player_id=i,
            path=list(paths[i]),
            path_set=set(paths[i]),
            head=heads[i],
            claimed_count=claimed[i],
            alive=alive[i],
        )
        for i in range(n)
    ]
    return GameState(grid=grid, players=players, current_player=current_player)


def test_single_legal_step_advances_turn() -> None:
    state = new_game(board_size=5, num_players=2)
    # Player 0 spawns at (4, 4) on a 5x5 (inset=4). Move N (action=0) to (3, 4).
    assert state.players[0].head == (4, 4)

    result = step(state, 0)

    assert isinstance(result, StepResult)
    assert result.state is state
    assert state.current_player == 1
    assert state.turn_number == 1
    assert state.players[0].head == (3, 4)
    assert state.grid[3, 4] == PATH_CODES[0]
    assert state.players[0].path == [(4, 4), (3, 4)]
    assert (3, 4) in state.players[0].path_set
    assert result.reward == 1.0
    assert result.done is False
    assert state.done is False
    assert result.info["player_who_moved"] == 0
    assert result.info["illegal_move"] is False
    assert result.info["claimed_this_turn"] == 0


def test_illegal_action_non_strict_kills_player() -> None:
    # 3x3 board, player 0 at corner (0,0); action 0 (N) is off-grid → illegal.
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[2, 2] = PATH_CODES[1]
    state = _make_state(grid, [(0, 0), (2, 2)])

    result = step(state, 0)

    assert result.info["illegal_move"] is True
    assert result.info["player_who_moved"] == 0
    assert state.players[0].alive is False
    assert result.reward == 0.0
    # Turn advanced to the surviving player.
    assert state.current_player == 1
    assert state.turn_number == 1
    # Only one player left alive → game ends.
    assert state.done is True
    assert result.done is True


def test_illegal_action_strict_raises() -> None:
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[2, 2] = PATH_CODES[1]
    state = _make_state(grid, [(0, 0), (2, 2)])

    with pytest.raises(IllegalMoveError):
        step(state, 0, strict=True)


def test_out_of_range_action_treated_illegal() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = PATH_CODES[0]
    grid[4, 4] = PATH_CODES[1]
    state = _make_state(grid, [(2, 2), (4, 4)])

    result = step(state, 99)

    assert result.info["illegal_move"] is True
    assert state.players[0].alive is False


def test_out_of_range_action_strict_raises() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = PATH_CODES[0]
    grid[4, 4] = PATH_CODES[1]
    state = _make_state(grid, [(2, 2), (4, 4)])

    with pytest.raises(IllegalMoveError):
        step(state, -1, strict=True)


def test_reward_includes_claimed_cells() -> None:
    # Near-loop from test_enclosure. Path so far (7 cells), head at (3, 1);
    # action N → (2, 1) closes the loop and encloses (2, 2).
    grid = np.zeros((5, 5), dtype=np.int8)
    path0 = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1)]
    for r, c in path0:
        grid[r, c] = PATH_CODES[0]
    grid[0, 0] = PATH_CODES[1]
    state = _make_state(
        grid,
        heads=[(3, 1), (0, 0)],
        paths=[path0, [(0, 0)]],
    )

    result = step(state, 0)

    assert result.info["illegal_move"] is False
    assert result.info["claimed_this_turn"] == 1
    assert state.grid[2, 2] == CLAIMED_CODES[0]
    assert state.players[0].claimed_count == 1
    assert result.reward == 2.0  # 1 placement + 1 claimed


def test_game_ends_when_board_fills() -> None:
    # Drive a tiny 5x5 game with first-legal-action until done.
    state = new_game(board_size=5, num_players=2)
    max_steps = 5 * 5 + 5
    for i in range(max_steps):
        if state.done:
            break
        acts = legal_actions(state, state.current_player)
        assert acts, f"step {i}: current player should have legal actions"
        step(state, acts[0])

    assert state.done is True, "game must terminate within safety cap"
    assert state.winner in (None, 0, 1)


def test_winner_is_argmax_of_path_plus_claimed() -> None:
    # Hand-built endgame: player 1 has far more territory.
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[2, 2] = PATH_CODES[1]
    state = _make_state(
        grid,
        heads=[(0, 0), (2, 2)],
        paths=[[(0, 0)], [(2, 2)]],
        claimed=[0, 5],
    )
    state.done = True
    state.winner = 1

    rewards = compute_terminal_reward(state)
    assert rewards == [-1.0, 1.0]


def test_compute_terminal_reward_sparse_tie() -> None:
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[2, 2] = PATH_CODES[1]
    state = _make_state(grid, [(0, 0), (2, 2)])
    state.done = True
    state.winner = None

    assert compute_terminal_reward(state, scheme="sparse") == [0.0, 0.0]


def test_compute_terminal_reward_sparse_win_loss() -> None:
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[2, 2] = PATH_CODES[1]
    state = _make_state(grid, [(0, 0), (2, 2)])
    state.winner = 0
    assert compute_terminal_reward(state) == [1.0, -1.0]


def test_compute_terminal_reward_relative() -> None:
    # p0 score=1 (path len 1 + 0 claimed), p1 score=3 (path 1 + claimed 2).
    # mean=2, centered=[-1,1], max|c|=1 → [-1, 1].
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[2, 2] = PATH_CODES[1]
    state = _make_state(
        grid,
        heads=[(0, 0), (2, 2)],
        paths=[[(0, 0)], [(2, 2)]],
        claimed=[0, 2],
    )

    rewards = compute_terminal_reward(state, scheme="relative")
    assert rewards == [-1.0, 1.0]


def test_compute_terminal_reward_relative_all_equal() -> None:
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[2, 2] = PATH_CODES[1]
    state = _make_state(grid, [(0, 0), (2, 2)])
    assert compute_terminal_reward(state, scheme="relative") == [0.0, 0.0]


def test_compute_terminal_reward_unknown_scheme_raises() -> None:
    state = new_game(board_size=5, num_players=2)
    with pytest.raises(ValueError):
        compute_terminal_reward(state, scheme="bogus")


def test_step_on_finished_game_raises() -> None:
    state = new_game(board_size=5, num_players=2)
    state.done = True
    with pytest.raises(ValueError):
        step(state, 0)


def test_step_skips_dead_player_in_turn_advancement() -> None:
    # 3 players on a 5x5; middle player already dead. After player 0 moves,
    # the turn should skip player 1 and land on player 2.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = PATH_CODES[0]
    grid[0, 4] = PATH_CODES[1]
    grid[4, 4] = PATH_CODES[2]
    state = _make_state(
        grid,
        heads=[(2, 2), (0, 4), (4, 4)],
        alive=[True, False, True],
    )

    step(state, 0)  # N: (2,2) → (1,2)

    assert state.current_player == 2
    assert state.turn_number == 1
    assert state.players[1].alive is False  # unchanged
    assert state.done is False  # 2 players still alive


def test_step_auto_kills_surrounded_player_on_turn_advance() -> None:
    # 5x5: player 0 at corner free to move; player 1 at (2,2) boxed in on all
    # four sides. When player 0 moves, _advance_turn tries player 1, finds 0
    # legal actions, kills them, ending the game with player 0 as winner.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[0, 0] = PATH_CODES[0]
    grid[2, 2] = PATH_CODES[1]
    for r, c in [(1, 2), (3, 2), (2, 1), (2, 3)]:
        grid[r, c] = CLAIMED_CODES[0]

    state = _make_state(
        grid,
        heads=[(0, 0), (2, 2)],
        paths=[[(0, 0)], [(2, 2)]],
        claimed=[4, 0],
    )

    result = step(state, 1)  # player 0 moves S: (0,0) → (1,0)

    assert result.info["illegal_move"] is False
    assert state.players[1].alive is False
    assert state.done is True
    assert state.winner == 0


def test_step_current_player_always_has_legal_actions() -> None:
    # After step(), if not done, the new current_player must have legal actions
    # (invariant guaranteed by _advance_turn).
    state = new_game(board_size=6, num_players=2)
    for _ in range(20):
        if state.done:
            break
        acts = legal_actions(state, state.current_player)
        assert acts, "invariant violated: current_player has no legal actions"
        step(state, acts[0])
