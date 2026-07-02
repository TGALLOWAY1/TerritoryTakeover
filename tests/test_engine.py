"""Tests for engine.new_game, engine.reset, and engine.has_reachable_empty."""

from __future__ import annotations

import numpy as np
import pytest

from territory_takeover.constants import EMPTY, OWNED_CODES
from territory_takeover.engine import has_reachable_empty, new_game, reset, step


def test_grid_has_exactly_four_owned_tiles_after_init() -> None:
    state = new_game()
    assert int((state.grid != EMPTY).sum()) == 4


def test_each_player_head_equals_spawn_and_counts_as_territory() -> None:
    expected = [(0, 0), (0, 39), (39, 0), (39, 39)]
    state = new_game()
    for i, spawn in enumerate(expected):
        p = state.players[i]
        assert p.head == spawn
        assert p.territory_count == 1
        assert p.alive
        assert state.grid[spawn] == OWNED_CODES[i]
    assert state.empty_count == 40 * 40 - 4
    assert state.alive_count == 4


def test_default_2p_spawns_are_diagonal_corners() -> None:
    state = new_game(board_size=9, num_players=2)
    assert [p.head for p in state.players] == [(0, 0), (8, 8)]


def test_spawn_collision_raises() -> None:
    with pytest.raises(ValueError):
        new_game(spawn_positions=[(4, 4), (4, 4), (35, 4), (35, 35)])


def test_out_of_bounds_spawn_raises() -> None:
    with pytest.raises(ValueError):
        new_game(spawn_positions=[(-1, 0), (4, 35), (35, 4), (35, 35)])
    with pytest.raises(ValueError):
        new_game(spawn_positions=[(0, 0), (4, 35), (35, 4), (40, 0)])


def test_spawn_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        new_game(num_players=4, spawn_positions=[(4, 4), (35, 35)])


def test_bad_board_size_and_player_count_raise() -> None:
    with pytest.raises(ValueError):
        new_game(board_size=0)
    with pytest.raises(ValueError):
        new_game(num_players=0)
    with pytest.raises(ValueError):
        new_game(num_players=5)
    with pytest.raises(ValueError):
        new_game(board_size=1, num_players=2)


def test_default_spawns_undefined_for_3_players() -> None:
    with pytest.raises(ValueError):
        new_game(board_size=8, num_players=3)
    # Explicit spawns make 3 players fine.
    state = new_game(board_size=8, num_players=3, spawn_positions=[(0, 0), (0, 7), (7, 0)])
    assert state.alive_count == 3


def test_seeded_spawn_permutation_is_reproducible() -> None:
    a = new_game(board_size=10, num_players=4, seed=123)
    b = new_game(board_size=10, num_players=4, seed=123)
    assert [p.head for p in a.players] == [p.head for p in b.players]
    corners = {(0, 0), (0, 9), (9, 0), (9, 9)}
    assert {p.head for p in a.players} == corners


def test_reset_restores_initial_configuration_in_place() -> None:
    state = new_game(board_size=8, num_players=2)
    grid_buf = state.grid
    for a in (1, 0, 3, 2, 1, 0):
        if state.done:
            break
        step(state, a)
    reset(state)
    assert state.grid is grid_buf, "reset must reuse the grid buffer"
    assert int((state.grid != EMPTY).sum()) == 2
    assert [p.head for p in state.players] == [(0, 0), (7, 7)]
    assert [p.territory_count for p in state.players] == [1, 1]
    assert state.turn_number == 0
    assert state.current_player == 0
    assert state.winner is None
    assert not state.done
    assert state.empty_count == 62
    assert state.alive_count == 2


def test_has_reachable_empty_open_board() -> None:
    state = new_game(board_size=5, num_players=2)
    assert has_reachable_empty(state, 0)
    assert has_reachable_empty(state, 1)


def test_has_reachable_empty_through_own_territory() -> None:
    # P0 owns a corridor (0,0)..(0,3); head deep at (0,0) walled by P1 below,
    # but an empty cell at (0,4) is reachable by walking own cells East.
    state = new_game(board_size=5, num_players=2, spawn_positions=[(0, 0), (4, 4)])
    grid = state.grid
    for c in range(4):
        grid[0, c] = OWNED_CODES[0]
    for c in range(4):
        grid[1, c] = OWNED_CODES[1]
    state.players[0].territory_count = 4
    state.players[1].territory_count = 5
    state.empty_count = int((grid == EMPTY).sum())
    # Head at (0,0): no empty neighbor ((0,1) own, (1,0) opponent), yet alive.
    assert has_reachable_empty(state, 0)


def test_has_reachable_empty_false_when_walled_out() -> None:
    # P0 boxed into the corner by P1's wall; empties exist beyond the wall.
    state = new_game(board_size=5, num_players=2, spawn_positions=[(0, 0), (4, 4)])
    grid = state.grid
    grid[0, 1] = OWNED_CODES[1]
    grid[1, 0] = OWNED_CODES[1]
    grid[1, 1] = OWNED_CODES[1]
    state.players[1].territory_count = 4
    state.empty_count = int((grid == EMPTY).sum())
    assert not has_reachable_empty(state, 0)
    assert has_reachable_empty(state, 1)


def test_witness_cache_invalidation() -> None:
    # First check caches a witness; opponent then claims that exact cell and
    # every other empty around P0 — the stale witness must not keep P0 alive.
    state = new_game(board_size=4, num_players=2, spawn_positions=[(0, 0), (3, 3)])
    assert has_reachable_empty(state, 0)
    witness = state.players[0].alive_witness
    assert witness is not None
    grid = state.grid
    for r in range(4):
        for c in range(4):
            if grid.item(r, c) == EMPTY:
                grid[r, c] = OWNED_CODES[1]
    state.empty_count = 0
    assert not has_reachable_empty(state, 0)


def test_full_random_game_terminates_with_all_dead(num_trials: int = 3) -> None:
    rng = np.random.default_rng(7)
    for trial in range(num_trials):
        state = new_game(board_size=7, num_players=4, seed=trial)
        guard = 0
        while not state.done:
            from territory_takeover.actions import legal_actions

            acts = legal_actions(state, state.current_player)
            step(state, int(rng.choice(acts)) if acts else 0)
            guard += 1
            assert guard < 20000, f"trial={trial}: game did not terminate"
        assert state.alive_count == 0, f"trial={trial}"
        total = sum(p.territory_count for p in state.players) + state.empty_count
        assert total == 49, f"trial={trial}: cell conservation violated"
