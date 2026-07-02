"""Fast-path rollout API for MCTS / RL self-play.

`simulate_random_rollout` drives an existing `GameState` to terminal under a
claim-biased random policy, in place: each turn the mover picks uniformly at
random among *claiming* moves (EMPTY targets) when any exist, and uniformly
among traversal moves (own-territory targets) otherwise. The claim bias is
what keeps playouts short — a uniform policy over all legal moves would
random-walk over already-owned cells and stretch playouts by an order of
magnitude without changing which regions each player can ever reach.

Use `step()` when you need per-step `reward`/`info`/`StepResult` semantics
(RL rollouts, debugging). Use `simulate_random_rollout` when you just need
the terminal state (MCTS playouts, bulk data generation).

Correctness: the equivalence suite in `tests/test_rollout_api.py` asserts
that for any seed, `simulate_random_rollout(state, rng)` produces a final
`GameState` with identical `grid`, `players[i].territory_count`, `winner`,
and `done` as running `step()` with the same claim-biased action stream.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .constants import DIRECTIONS, EMPTY, OWNED_CODES
from .engine import _advance_turn, _compute_winner

if TYPE_CHECKING:
    from .state import GameState


def simulate_random_rollout(state: GameState, rng: random.Random) -> int:
    """Play `state` to terminal under a claim-biased random policy, in place.

    Returns the total number of half-moves (turns) executed.

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
    owned_codes = OWNED_CODES
    dirs = DIRECTIONS
    randrange = rng.randrange

    while not state.done:
        pid = state.current_player
        p = players[pid]
        hr, hc = p.head
        own = owned_codes[pid]

        # Collect claiming moves (EMPTY targets) and traversal moves (own
        # cells) inline. Claiming moves take priority.
        claims: list[int] = []
        traverses: list[int] = []
        if hr > 0:
            v = grid.item(hr - 1, hc)
            if v == EMPTY:
                claims.append(0)
            elif v == own:
                traverses.append(0)
        if hr < h - 1:
            v = grid.item(hr + 1, hc)
            if v == EMPTY:
                claims.append(1)
            elif v == own:
                traverses.append(1)
        if hc > 0:
            v = grid.item(hr, hc - 1)
            if v == EMPTY:
                claims.append(2)
            elif v == own:
                traverses.append(2)
        if hc < w - 1:
            v = grid.item(hr, hc + 1)
            if v == EMPTY:
                claims.append(3)
            elif v == own:
                traverses.append(3)

        if claims:
            action = claims[randrange(len(claims))]
            dr, dc = dirs[action]
            tr, tc = hr + dr, hc + dc
            grid[tr, tc] = own
            p.head = (tr, tc)
            p.territory_count += 1
            if state.empty_count > 0:
                state.empty_count -= 1
        elif traverses:
            action = traverses[randrange(len(traverses))]
            dr, dc = dirs[action]
            p.head = (hr + dr, hc + dc)
        # No legal moves at all (fully walled in): fall through — the
        # turn-advance liveness check below marks the seat dead.

        _advance_turn(state)
        if state.alive_count <= 0:
            state.done = True
            state.winner = _compute_winner(state)
        turns += 1

    return turns
