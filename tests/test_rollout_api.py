"""Tests for the `simulate_random_rollout` fast-path API.

The contract: `simulate_random_rollout(state, rng)` must drive `state` to
terminal with EXACTLY the same trajectory as a reference loop of
`step(state, action)` calls that picks each action uniformly at random among
the current `legal_actions(...)` (falling back to `step(state, 0)` when there
are none — the "no legal moves" path).

These tests fix an RNG seed, clone the state, drive both implementations, and
assert the final `GameState` is byte-identical: same grid, same per-player
`(path, path_set, head, claimed_count, alive)`, same turn counter, same winner
and `done` flag.
"""

from __future__ import annotations

import random

import numpy as np

from territory_takeover import GameState, new_game, simulate_random_rollout, step
from territory_takeover.actions import legal_actions


def _reference_rollout(state: GameState, rng: random.Random) -> int:
    """Reference: drive `state` to terminal via step() + random legal action."""
    turns = 0
    while not state.done:
        acts = legal_actions(state, state.current_player)
        action = acts[rng.randrange(len(acts))] if acts else 0
        step(state, action)
        turns += 1
    return turns


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


def _equivalence(board_size: int, num_players: int, seed: int) -> None:
    fast_state = new_game(
        board_size=board_size, num_players=num_players, seed=seed
    )
    ref_state = new_game(
        board_size=board_size, num_players=num_players, seed=seed
    )
    fast_rng = random.Random(seed)
    ref_rng = random.Random(seed)

    fast_turns = simulate_random_rollout(fast_state, fast_rng)
    ref_turns = _reference_rollout(ref_state, ref_rng)

    assert fast_turns == ref_turns, (
        f"seed={seed} board={board_size}: turn count {fast_turns} vs {ref_turns}"
    )
    _assert_states_equal(
        fast_state, ref_state, f"seed={seed} board={board_size}"
    )


def test_rollout_equivalence_20x20_4p() -> None:
    for seed in range(30):
        _equivalence(board_size=20, num_players=4, seed=seed)


def test_rollout_equivalence_40x40_4p() -> None:
    for seed in range(10):
        _equivalence(board_size=40, num_players=4, seed=seed)


def test_rollout_equivalence_10x10_2p() -> None:
    for seed in range(20):
        _equivalence(board_size=10, num_players=2, seed=seed)


def test_rollout_terminates_and_game_is_done() -> None:
    state = new_game(board_size=10, num_players=2, seed=0)
    rng = random.Random(0)
    turns = simulate_random_rollout(state, rng)

    assert state.done is True
    assert state.alive_count <= 1
    assert turns > 0
    # Winner either is an int in [0, num_players) or None on a tie.
    assert state.winner is None or 0 <= state.winner < len(state.players)


def test_rollout_raises_on_finished_state() -> None:
    state = new_game(board_size=6, num_players=2, seed=0)
    rng = random.Random(0)
    simulate_random_rollout(state, rng)
    assert state.done

    try:
        simulate_random_rollout(state, rng)
    except ValueError:
        return
    msg = "simulate_random_rollout on a done state must raise ValueError"
    raise AssertionError(msg)
