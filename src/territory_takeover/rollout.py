"""Fast-path rollout API for MCTS / RL self-play.

`simulate_random_rollout` drives an existing `GameState` to terminal under a
uniform-random legal policy, in place. It is semantically equivalent to
calling `step()` in a loop with a uniformly-chosen legal action — and to the
`illegal-fallback` step(0) branch when the current player has no legal moves
— but skips the `StepResult`/`info` dict allocations, the numpy 4-element
action-mask allocation, and inlines the legality / coord / path-append /
enclosure-detection path into a single tight Python loop.

Use `step()` when you need per-step `reward`/`info`/`StepResult` semantics
(RL rollouts, debugging). Use `simulate_random_rollout` when you just need
the terminal state (MCTS playouts, bulk data generation).

Correctness: the equivalence suite in `tests/test_rollout_api.py` asserts
that for any seed, `simulate_random_rollout(state, rng)` produces a final
`GameState` with identical `grid`, `players[i].path`, `players[i].claimed_count`,
`winner`, and `done` as running `step()` with the same action stream.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .constants import DIRECTIONS, EMPTY, PATH_CODES
from .engine import _advance_turn, _compute_winner, detect_and_apply_enclosure

if TYPE_CHECKING:
    from .state import GameState


def simulate_random_rollout(state: GameState, rng: random.Random) -> int:
    """Play `state` to terminal under a uniform-random legal policy, in place.

    Returns the total number of half-moves (turns) executed, including the
    "no legal actions" bump-and-kill steps.

    The caller is expected to pass a live state (`state.done is False`);
    passing an already-done state raises `ValueError` to mirror `step()`.
    """
    if state.done:
        raise ValueError(
            "simulate_random_rollout() called on a finished game "
            "(state.done is True)"
        )

    if state.alive_count < 0:
        state.alive_count = sum(1 for p in state.players if p.alive)

    grid = state.grid
    h, w = grid.shape
    players = state.players
    turns = 0

    # Local bindings avoid attribute lookups in the hot loop.
    path_codes = PATH_CODES
    dirs = DIRECTIONS
    randrange = rng.randrange

    while not state.done:
        pid = state.current_player
        p = players[pid]
        hr, hc = p.head

        # Collect legal direction indices inline. One four-way check, no list
        # allocation for the 0-legal fast path — we instead assemble into a
        # small fixed 4-slot buffer.
        acts: list[int] = []
        if hr > 0 and grid.item(hr - 1, hc) == EMPTY:
            acts.append(0)
        if hr < h - 1 and grid.item(hr + 1, hc) == EMPTY:
            acts.append(1)
        if hc > 0 and grid.item(hr, hc - 1) == EMPTY:
            acts.append(2)
        if hc < w - 1 and grid.item(hr, hc + 1) == EMPTY:
            acts.append(3)

        if not acts:
            # No legal moves: player dies via the normal turn-advance path.
            # step() uses action=0 here; we replicate by marking the seat
            # dead and advancing.
            if p.alive:
                p.alive = False
                state.alive_count -= 1
            _advance_turn(state)
            if state.alive_count <= 1:
                state.done = True
                state.winner = _compute_winner(state)
            turns += 1
            continue

        # Uniform random legal action. randrange(1) returns 0, so this
        # collapses to acts[0] when only one action is legal — avoiding the
        # random.choice overhead of a list-bounds check.
        action = acts[randrange(len(acts))]

        dr, dc = dirs[action]
        tr, tc = hr + dr, hc + dc
        grid[tr, tc] = path_codes[pid]
        p.path.append((tr, tc))
        p.path_set.add((tr, tc))
        p.head = (tr, tc)
        if state.empty_count > 0:
            state.empty_count -= 1
        detect_and_apply_enclosure(state, pid, (tr, tc))

        _advance_turn(state)
        if state.alive_count <= 1:
            state.done = True
            state.winner = _compute_winner(state)
        turns += 1

    return turns
