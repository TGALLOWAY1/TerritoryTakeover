"""Tests for engine.new_game and engine.reset."""

from __future__ import annotations

import numpy as np
import pytest

from territory_takeover.constants import EMPTY, PATH_CODES
from territory_takeover.engine import new_game, reset


def test_grid_has_exactly_four_path_tiles_after_init() -> None:
    state = new_game()
    assert int((state.grid != EMPTY).sum()) == 4


def test_each_player_head_equals_spawn() -> None:
    expected = [(4, 4), (4, 35), (35, 4), (35, 35)]
    state = new_game()
    for i, spawn in enumerate(expected):
        p = state.players[i]
        assert p.head == spawn
        assert p.path == [spawn]
        assert p.path_set == {spawn}
        assert state.grid[spawn] == PATH_CODES[i]


def test_spawn_collision_raises() -> None:
    with pytest.raises(ValueError):
        new_game(spawn_positions=[(4, 4), (4, 4), (35, 4), (35, 35)])


def test_default_40x40_spawn_coordinates() -> None:
    state = new_game()
    heads = [p.head for p in state.players]
    assert heads == [(4, 4), (4, 35), (35, 4), (35, 35)]


def test_out_of_bounds_spawn_raises() -> None:
    with pytest.raises(ValueError):
        new_game(spawn_positions=[(-1, 0), (4, 35), (35, 4), (35, 35)])
    with pytest.raises(ValueError):
        new_game(spawn_positions=[(0, 0), (4, 35), (35, 4), (40, 0)])


def test_spawn_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        new_game(num_players=4, spawn_positions=[(4, 4), (35, 35)])


def test_invalid_num_players_raises() -> None:
    with pytest.raises(ValueError):
        new_game(num_players=0)
    with pytest.raises(ValueError):
        new_game(num_players=5)


def test_default_spawns_require_two_or_four_players() -> None:
    with pytest.raises(ValueError):
        new_game(num_players=3)
    # Explicit spawns bypass the 2/4 restriction.
    state = new_game(num_players=3, spawn_positions=[(0, 0), (0, 5), (5, 0)])
    assert [p.head for p in state.players] == [(0, 0), (0, 5), (5, 0)]


def test_new_game_initial_counters() -> None:
    state = new_game()
    assert state.current_player == 0
    assert state.turn_number == 0
    assert state.winner is None
    assert state.done is False
    for p in state.players:
        assert p.claimed_count == 0
        assert p.alive is True


def test_two_player_default_spawns() -> None:
    state = new_game(num_players=2)
    heads = [p.head for p in state.players]
    assert heads == [(4, 4), (35, 35)]
    assert int((state.grid != EMPTY).sum()) == 2


def test_seed_shuffles_spawn_assignment_reproducibly() -> None:
    a = new_game(seed=0)
    b = new_game(seed=0)
    heads_a = [p.head for p in a.players]
    heads_b = [p.head for p in b.players]
    assert heads_a == heads_b

    # Seeded assignment is still a permutation of the canonical corners.
    canonical = {(4, 4), (4, 35), (35, 4), (35, 35)}
    assert set(heads_a) == canonical

    # Default (seed=None) preserves canonical order.
    default = new_game()
    assert [p.head for p in default.players] == [(4, 4), (4, 35), (35, 4), (35, 35)]


def test_custom_board_size_default_spawns() -> None:
    state = new_game(board_size=20)
    heads = [p.head for p in state.players]
    assert heads == [(4, 4), (4, 15), (15, 4), (15, 15)]


def test_default_spawns_never_adjacent() -> None:
    """Regression for the `_default_spawns` clamp.

    A fixed 4-cell inset placed 2-player 8x8 spawns at (4,4) and (3,3)
    (diagonally adjacent, Manhattan distance 2) and 10x10 spawns at
    (4,4) and (5,5) (same problem). The clamp in `_default_spawns`
    guarantees every pair of default spawns is at least Manhattan
    distance 3 apart for every supported (board_size, num_players)
    pair from 6 upwards — the smallest distance a player can move
    toward another spawn without immediately colliding on the next
    ply.
    """
    for board_size in (6, 7, 8, 9, 10, 11, 12, 16, 20, 40):
        for num_players in (2, 4):
            state = new_game(board_size=board_size, num_players=num_players)
            heads = [p.head for p in state.players]
            for i, a in enumerate(heads):
                for b in heads[i + 1:]:
                    dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
                    assert dist >= 3, (
                        f"board={board_size} np={num_players}: spawns {a} and "
                        f"{b} only {dist} apart (want >= 3)"
                    )


def test_reset_restores_initial_state() -> None:
    state = new_game()
    # Mutate state to simulate mid-game progress.
    state.grid[10, 10] = PATH_CODES[0]
    state.players[0].path.append((10, 10))
    state.players[0].path_set.add((10, 10))
    state.players[0].head = (10, 10)
    state.players[0].claimed_count = 7
    state.players[1].alive = False
    state.current_player = 2
    state.turn_number = 42
    state.winner = 1
    state.done = True

    reset(state)

    fresh = new_game()
    assert np.array_equal(state.grid, fresh.grid)
    for got, want in zip(state.players, fresh.players, strict=True):
        assert got.head == want.head
        assert got.path == want.path
        assert got.path_set == want.path_set
        assert got.claimed_count == want.claimed_count
        assert got.alive == want.alive
    assert state.current_player == 0
    assert state.turn_number == 0
    assert state.winner is None
    assert state.done is False


def test_reset_reuses_grid_buffer() -> None:
    state = new_game()
    grid_id_before = id(state.grid)
    state.grid[10, 10] = PATH_CODES[0]
    reset(state)
    assert id(state.grid) == grid_id_before


def test_reset_two_players() -> None:
    state = new_game(num_players=2)
    state.grid[2, 2] = PATH_CODES[0]
    state.players[0].head = (2, 2)
    state.turn_number = 5
    reset(state)
    assert [p.head for p in state.players] == [(4, 4), (35, 35)]
    assert state.turn_number == 0
    assert int((state.grid != EMPTY).sum()) == 2
