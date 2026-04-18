"""Tests for legal_actions / legal_action_mask / action_to_coord."""

from __future__ import annotations

import time

import numpy as np

from territory_takeover.actions import (
    action_to_coord,
    legal_action_mask,
    legal_actions,
)
from territory_takeover.constants import (
    PLAYER_1_PATH,
    PLAYER_2_CLAIMED,
    PLAYER_2_PATH,
)
from territory_takeover.state import GameState, PlayerState


def _make_state(
    grid: np.ndarray, heads: list[tuple[int, int]]
) -> GameState:
    players = [
        PlayerState(
            player_id=i,
            path=[h],
            path_set={h},
            head=h,
            claimed_count=0,
            alive=True,
        )
        for i, h in enumerate(heads)
    ]
    return GameState(grid=grid, players=players)


def test_corner_head_has_at_most_two_actions() -> None:
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[0, 0] = PLAYER_1_PATH
    state = _make_state(grid, [(0, 0)])

    actions = legal_actions(state, 0)
    assert sorted(actions) == [1, 3]  # S, E
    assert len(actions) <= 2


def test_head_in_every_corner() -> None:
    # Top-right, bottom-left, bottom-right corners of a 5x5 board.
    cases = [
        ((0, 4), [1, 2]),   # S, W
        ((4, 0), [0, 3]),   # N, E
        ((4, 4), [0, 2]),   # N, W
    ]
    for head, expected in cases:
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[head] = PLAYER_1_PATH
        state = _make_state(grid, [head])
        assert sorted(legal_actions(state, 0)) == expected


def test_surrounded_by_own_path_returns_zero_actions() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    head = (2, 2)
    grid[head] = PLAYER_1_PATH
    grid[1, 2] = PLAYER_1_PATH
    grid[3, 2] = PLAYER_1_PATH
    grid[2, 1] = PLAYER_1_PATH
    grid[2, 3] = PLAYER_1_PATH
    state = _make_state(grid, [head])

    assert legal_actions(state, 0) == []
    mask = legal_action_mask(state, 0)
    assert mask.shape == (4,)
    assert mask.dtype == np.bool_
    assert not mask.any()


def test_blocked_by_opponent_path() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    head = (2, 2)
    grid[head] = PLAYER_1_PATH
    grid[2, 3] = PLAYER_2_PATH  # block E
    state = _make_state(grid, [head, (4, 4)])

    actions = legal_actions(state, 0)
    assert 3 not in actions
    assert sorted(actions) == [0, 1, 2]


def test_blocked_by_opponent_claimed_territory() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    head = (2, 2)
    grid[head] = PLAYER_1_PATH
    grid[2, 3] = PLAYER_2_CLAIMED  # block E
    state = _make_state(grid, [head, (4, 4)])

    actions = legal_actions(state, 0)
    assert 3 not in actions
    assert sorted(actions) == [0, 1, 2]


def test_mask_and_list_agree() -> None:
    configs: list[tuple[tuple[int, int], list[tuple[int, int, int]]]] = [
        # (head, [(r, c, code), ...])
        ((0, 0), []),                                   # corner, empty
        ((2, 2), [(1, 2, PLAYER_2_PATH)]),              # N blocked
        ((2, 2), [(2, 1, PLAYER_2_CLAIMED),
                  (2, 3, PLAYER_1_PATH)]),              # W, E blocked
        ((3, 3), [(2, 3, PLAYER_2_PATH),
                  (4, 3, PLAYER_2_PATH),
                  (3, 2, PLAYER_2_PATH),
                  (3, 4, PLAYER_2_PATH)]),              # all 4 blocked
    ]
    for head, blocks in configs:
        grid = np.zeros((6, 6), dtype=np.int8)
        grid[head] = PLAYER_1_PATH
        for r, c, code in blocks:
            grid[r, c] = code
        state = _make_state(grid, [head])

        lst = legal_actions(state, 0)
        mask = legal_action_mask(state, 0)
        assert sorted(lst) == np.nonzero(mask)[0].tolist()


def test_action_to_coord_all_directions() -> None:
    grid = np.zeros((6, 6), dtype=np.int8)
    head = (3, 4)
    grid[head] = PLAYER_1_PATH
    state = _make_state(grid, [head])

    assert action_to_coord(state, 0, 0) == (2, 4)  # N
    assert action_to_coord(state, 0, 1) == (4, 4)  # S
    assert action_to_coord(state, 0, 2) == (3, 3)  # W
    assert action_to_coord(state, 0, 3) == (3, 5)  # E


def test_one_by_one_grid_has_zero_actions() -> None:
    grid = np.zeros((1, 1), dtype=np.int8)
    grid[0, 0] = PLAYER_1_PATH
    state = _make_state(grid, [(0, 0)])

    assert legal_actions(state, 0) == []
    assert not legal_action_mask(state, 0).any()


def test_mask_shape_and_dtype() -> None:
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[1, 1] = PLAYER_1_PATH
    state = _make_state(grid, [(1, 1)])
    mask = legal_action_mask(state, 0)

    assert mask.shape == (4,)
    assert mask.dtype == np.bool_


def test_per_player_head_resolution() -> None:
    # Two players; legality must depend on the queried player's head.
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[0, 0] = PLAYER_1_PATH
    grid[4, 4] = PLAYER_2_PATH
    state = _make_state(grid, [(0, 0), (4, 4)])

    assert sorted(legal_actions(state, 0)) == [1, 3]  # corner (0,0)
    assert sorted(legal_actions(state, 1)) == [0, 2]  # corner (4,4)


def test_perf_legal_actions_smoke(capsys: object) -> None:
    grid = np.zeros((40, 40), dtype=np.int8)
    grid[20, 20] = PLAYER_1_PATH
    state = _make_state(grid, [(20, 20)])

    iterations = 10_000
    for _ in range(500):
        legal_actions(state, 0)

    start = time.perf_counter()
    for _ in range(iterations):
        legal_actions(state, 0)
    mean_us = (time.perf_counter() - start) / iterations * 1e6

    print(f"legal_actions mean: {mean_us:.3f} µs over {iterations} iters")
    # Target is < 1 µs; assert a loose CI-friendly upper bound.
    assert mean_us < 5.0, f"legal_actions too slow: {mean_us:.3f} µs mean"
