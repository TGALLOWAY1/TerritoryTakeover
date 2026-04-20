"""Randomized equivalence tests: optimized engine vs legacy reference.

The optimized `detect_and_apply_enclosure` uses a localized BFS that seeds
from interior cells adjacent to the newly-closed loop. The reference
`_legacy_detect_and_apply_enclosure_full_bfs` uses the original boundary
BFS over the entire grid. These tests drive identical random trajectories
through both engines and assert that the resulting `GameState` is
byte-identical at every step: same grid, same per-player path + claimed
count, same turn counter, same winner / done flag.

This is the safety net for the localized-BFS rewrite. Any divergence here
is a correctness regression.
"""

from __future__ import annotations

import random

import numpy as np

from territory_takeover import GameState, new_game, step
from territory_takeover.actions import legal_actions
from territory_takeover.constants import PATH_CODES
from territory_takeover.engine import (
    _advance_turn,
    _compute_winner,
    _legacy_detect_and_apply_enclosure_full_bfs,
)


def _legacy_step(state: GameState, action: int) -> int:
    """Reference step(): inline-replica of engine.step() but using the
    legacy full-board BFS for enclosure detection. Returns claimed-this-turn.
    """
    if state.alive_count < 0:
        state.alive_count = sum(1 for p in state.players if p.alive)

    pid = state.current_player
    p = state.players[pid]
    grid = state.grid
    h, w = grid.shape

    legal = False
    target: tuple[int, int] | None = None
    if 0 <= action < 4:
        from territory_takeover.constants import DIRECTIONS, EMPTY

        dr, dc = DIRECTIONS[action]
        tr, tc = p.head[0] + dr, p.head[1] + dc
        if 0 <= tr < h and 0 <= tc < w and grid[tr, tc] == EMPTY:
            legal = True
            target = (tr, tc)

    claimed = 0
    if not legal:
        if p.alive:
            p.alive = False
            state.alive_count -= 1
    else:
        assert target is not None
        grid[target] = PATH_CODES[pid]
        p.path.append(target)
        p.path_set.add(target)
        p.head = target
        claimed = _legacy_detect_and_apply_enclosure_full_bfs(state, pid, target)

    _advance_turn(state)

    if state.alive_count <= 1:
        state.done = True
        state.winner = _compute_winner(state)
    return claimed


def _assert_states_equal(a: GameState, b: GameState, msg: str) -> None:
    assert np.array_equal(a.grid, b.grid), f"{msg}: grid diverged"
    assert len(a.players) == len(b.players)
    for i, (pa, pb) in enumerate(zip(a.players, b.players, strict=True)):
        assert pa.path == pb.path, f"{msg}: player {i} path diverged"
        assert pa.path_set == pb.path_set, f"{msg}: player {i} path_set diverged"
        assert pa.head == pb.head, f"{msg}: player {i} head diverged"
        assert pa.claimed_count == pb.claimed_count, (
            f"{msg}: player {i} claimed_count {pa.claimed_count} vs "
            f"{pb.claimed_count}"
        )
        assert pa.alive == pb.alive, f"{msg}: player {i} alive diverged"
    assert a.current_player == b.current_player, f"{msg}: current_player diverged"
    assert a.turn_number == b.turn_number, f"{msg}: turn_number diverged"
    assert a.winner == b.winner, f"{msg}: winner diverged"
    assert a.done == b.done, f"{msg}: done diverged"


def _equivalence_trajectory(
    board_size: int, num_players: int, seed: int, max_plies: int
) -> None:
    opt_state = new_game(board_size=board_size, num_players=num_players, seed=seed)
    ref_state = new_game(board_size=board_size, num_players=num_players, seed=seed)
    rng = random.Random(seed)

    for ply in range(max_plies):
        if opt_state.done and ref_state.done:
            break
        assert opt_state.done == ref_state.done, (
            f"seed={seed} ply={ply}: done diverged"
        )
        if opt_state.done:
            break
        # Same current_player at this point (verified by earlier equality).
        acts = legal_actions(opt_state, opt_state.current_player)
        action = rng.choice(acts) if acts else 0

        step(opt_state, action)
        _legacy_step(ref_state, action)

        _assert_states_equal(
            opt_state, ref_state, f"seed={seed} ply={ply} board={board_size}"
        )


def test_equivalence_20x20_4p_50_seeds() -> None:
    for seed in range(50):
        _equivalence_trajectory(
            board_size=20, num_players=4, seed=seed, max_plies=400
        )


def test_equivalence_40x40_4p_20_seeds() -> None:
    # Fewer seeds at 40x40 because each game is ~300 plies. 20 seeds ~= 6k plies.
    for seed in range(20):
        _equivalence_trajectory(
            board_size=40, num_players=4, seed=seed, max_plies=800
        )


def test_equivalence_10x10_2p_20_seeds() -> None:
    # Small 2-player games trigger more enclosures per ply on average.
    for seed in range(20):
        _equivalence_trajectory(
            board_size=10, num_players=2, seed=seed, max_plies=200
        )


def test_opponent_tile_inside_perimeter_equivalence() -> None:
    # The exact scenario exercised by test_enclosure.test_loop_encloses_opponent_path_tile
    # — a pocket split by an opponent path tile. This ensures the localized
    # BFS correctly seeds from every loop-adjacent EMPTY cell so both sides of
    # the split are claimed.
    from territory_takeover.constants import CLAIMED_CODES
    from territory_takeover.engine import detect_and_apply_enclosure

    def build_state() -> GameState:
        state = GameState.empty(5, 5, num_players=2)
        state.players[1].path = [(2, 2)]
        state.players[1].path_set = {(2, 2)}
        state.players[1].head = (2, 2)
        state.grid[2, 2] = PATH_CODES[1]

        p0_path = [
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 4),
            (3, 4), (3, 3), (3, 2), (3, 1), (3, 0),
            (2, 0),
        ]
        state.players[0].path = list(p0_path)
        state.players[0].path_set = set(p0_path)
        state.players[0].head = (2, 0)
        for r, c in p0_path:
            state.grid[r, c] = PATH_CODES[0]
        return state

    opt = build_state()
    ref = build_state()
    opt_claimed = detect_and_apply_enclosure(opt, 0, placed_cell=(2, 0))
    ref_claimed = _legacy_detect_and_apply_enclosure_full_bfs(
        ref, 0, placed_cell=(2, 0)
    )

    assert opt_claimed == ref_claimed == 2
    assert np.array_equal(opt.grid, ref.grid)
    assert opt.grid[2, 1] == CLAIMED_CODES[0]
    assert opt.grid[2, 3] == CLAIMED_CODES[0]
    assert opt.grid[2, 2] == PATH_CODES[1]
