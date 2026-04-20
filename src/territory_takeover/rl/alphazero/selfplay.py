"""Self-play rollouts for AlphaZero training.

``play_game`` drives every seat with a single PUCT search and emits one
:class:`Sample` per visited state. The move distribution used as the
policy target is ``visits / visits.sum()`` at the root; the value target
is the final game's per-seat normalized score vector, the same for every
sample in that game (standard AlphaZero).

Temperature handling mirrors AlphaGo Zero's canonical schedule: the
first ``temperature_moves`` half-moves sample actions proportional to
visits (exploration), then the agent plays greedily (temperature 0).
This is the single piece of diversity that prevents self-play from
collapsing into deterministic loops; cutting it too short strangles the
buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.actions import legal_action_mask
from territory_takeover.engine import new_game, step
from territory_takeover.rl.alphazero.mcts import (
    _sample_action_from_visits,
    _terminal_value_normalized,
    puct_search,
)
from territory_takeover.rl.alphazero.replay import Sample
from territory_takeover.rl.alphazero.spaces import encode_az_observation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.rl.alphazero.evaluator import NNEvaluator


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    board_size: int
    num_players: int
    puct_iterations: int = 64
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature_moves: int = 16
    """Half-moves at the start of a game during which actions are sampled
    proportional to visit counts. After this, moves are argmax.
    """

    max_half_moves: int | None = None
    """Hard cap on half-moves per self-play game. ``None`` disables the
    cap and the engine's terminal condition is the only stopper.
    """


def _normalized_final_scores(state: object) -> NDArray[np.float32]:
    """Compute the per-seat final value vector in ``[-1, 1]`` from a terminal state.

    Reuses :func:`_terminal_value_normalized`; kept as a thin helper for
    readability and to make the value-target path easy to override in
    ablations (e.g. scalar head => active-player-only scalar).
    """
    return _terminal_value_normalized(state).astype(np.float32)  # type: ignore[arg-type]


def play_game(
    evaluator: NNEvaluator,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    seed: int | None = None,
    spawn_positions: list[tuple[int, int]] | None = None,
) -> list[Sample]:
    """Run one self-play game and return every visited state as a Sample.

    Each step runs an independent PUCT search from the current state; the
    visit vector at the root becomes the policy target. The final value
    target is the same per-seat vector for every sample in the game,
    computed once at termination via :func:`_terminal_value_normalized`.
    """
    samples, _ = play_game_instrumented(
        evaluator, cfg, rng=rng, seed=seed, spawn_positions=spawn_positions
    )
    return samples


def play_game_instrumented(
    evaluator: NNEvaluator,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    seed: int | None = None,
    spawn_positions: list[tuple[int, int]] | None = None,
) -> tuple[list[Sample], int | None]:
    """Self-play driver with Phase 3d first-enclosure instrumentation.

    Returns ``(samples, first_enclosure_half_move)`` where the second
    element is the half-move index at which any player's
    ``claimed_count`` first became non-zero, or ``None`` if no enclosure
    occurred in this game. The check is a single integer read per
    half-move; the hot path is unaffected.
    """
    state = new_game(
        board_size=cfg.board_size,
        num_players=cfg.num_players,
        spawn_positions=spawn_positions,
        seed=seed,
    )
    samples_pending: list[
        tuple[
            NDArray[np.float32],
            NDArray[np.float32],
            NDArray[np.bool_],
            NDArray[np.float32],
        ]
    ] = []

    first_enclosure_half_move: int | None = None
    half_move = 0
    while not state.done:
        if cfg.max_half_moves is not None and half_move >= cfg.max_half_moves:
            break
        active = state.current_player
        grid_obs, scalar_obs = encode_az_observation(state, active)
        mask = legal_action_mask(state, active)

        _, visits = puct_search(
            state,
            active,
            evaluator,
            iterations=cfg.puct_iterations,
            c_puct=cfg.c_puct,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_eps=cfg.dirichlet_eps,
            rng=rng,
            temperature=1.0,  # we sample ourselves below
        )
        samples_pending.append((grid_obs, scalar_obs, mask, visits.copy()))

        temperature = 1.0 if half_move < cfg.temperature_moves else 0.0
        action = _sample_action_from_visits(visits, temperature, rng)
        step(state, action, strict=False)
        half_move += 1

        if first_enclosure_half_move is None and any(
            p.claimed_count > 0 for p in state.players
        ):
            first_enclosure_half_move = half_move

    final = _normalized_final_scores(state)
    samples = [
        Sample(grid=g, scalars=s, mask=m, visits=v, final_scores=final)
        for (g, s, m, v) in samples_pending
    ]
    return samples, first_enclosure_half_move


__all__ = ["SelfPlayConfig", "play_game", "play_game_instrumented"]
