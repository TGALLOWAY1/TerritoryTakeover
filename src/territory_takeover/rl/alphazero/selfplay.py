"""Self-play rollouts for AlphaZero training.

``play_game`` drives every seat with a single PUCT search and emits one
:class:`Sample` per visited state. The move distribution used as the
policy target is ``visits / visits.sum()`` at the root; the value target
is controlled by ``SelfPlayConfig.value_target_mode``:

- ``"terminal"`` (default, legacy behaviour): the final game's per-seat
  normalized score vector, the same for every sample in that game
  (standard AlphaZero).
- ``"nstep"``: per-sample n-step bootstrapped return, using the engine's
  per-step ``claimed_this_turn`` delta (normalized by board area) and
  bootstrapping V̂ from the self-play net at the horizon. Flag-gated;
  the terminal-mode Phase 3c tests continue to pass unchanged.

Temperature handling mirrors AlphaGo Zero's canonical schedule: the
first ``temperature_moves`` half-moves sample actions proportional to
visits (exploration), then the agent plays greedily (temperature 0).
This is the single piece of diversity that prevents self-play from
collapsing into deterministic loops; cutting it too short strangles the
buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

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


ValueTargetMode = Literal["terminal", "nstep"]


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

    value_target_mode: ValueTargetMode = "terminal"
    """Which value-head target to store in each :class:`Sample`.

    - ``"terminal"``: normalized per-seat final score (legacy; default).
    - ``"nstep"``: n-step bootstrapped per-seat return computed from
      the engine's per-step ``claimed_this_turn`` reward, with horizon
      :attr:`n_step` and discount :attr:`gamma`. Bootstrap comes from
      the self-play net's value head on the horizon observation.
    """

    n_step: int = 16
    """Horizon for the n-step bootstrap target (``value_target_mode='nstep'``)."""

    gamma: float = 0.99
    """Discount factor for the n-step bootstrap target."""


def _normalized_final_scores(state: object) -> NDArray[np.float32]:
    """Compute the per-seat final value vector in ``[-1, 1]`` from a terminal state.

    Reuses :func:`_terminal_value_normalized`; kept as a thin helper for
    readability and to make the value-target path easy to override in
    ablations (e.g. scalar head => active-player-only scalar).
    """
    return _terminal_value_normalized(state).astype(np.float32)  # type: ignore[arg-type]


def compute_nstep_value_targets(
    active_seats: NDArray[np.int32],
    per_step_rewards: NDArray[np.float32],
    bootstrap_values: NDArray[np.float32],
    terminal_scores: NDArray[np.float32],
    *,
    n_step: int,
    gamma: float,
    terminal: bool,
    num_players: int,
) -> NDArray[np.float32]:
    """Pure helper — per-sample per-seat n-step returns.

    Inputs (all length ``T``, where ``T`` is trajectory length):

    - ``active_seats[t]``: seat index that moved at step t.
    - ``per_step_rewards[t]``: scalar reward the active seat collected
      at step t (already normalized by board area).
    - ``bootstrap_values[t]``: per-seat V̂ at the *pre-move* observation
      of step t. Shape ``(T, num_players)``.
    - ``terminal_scores``: per-seat normalized terminal vector. Shape
      ``(num_players,)``. Used only when ``terminal=True`` and the
      horizon runs off the end of the trajectory.
    - ``n_step``: horizon length.
    - ``gamma``: discount.
    - ``terminal``: ``True`` iff the trajectory's last step ended the
      game (``state.done``). ``False`` iff the trajectory was cut by a
      ``max_half_moves`` cap.

    Returns ``(T, num_players)`` float32 target vectors, clamped to
    ``[-1, 1]``.

    For sample at step ``t``:

    ``G_t[p] = sum_{k=0..K-1} gamma^k * r^p_{t+k}  +  gamma^K * B_t[p]``

    where:

    - ``K = min(n_step, T - t)``.
    - ``r^p_{t+k} = per_step_rewards[t+k]`` iff ``active_seats[t+k] == p``
      else 0.
    - If ``t + n_step < T``: ``B_t = bootstrap_values[t + n_step]``.
    - Else (horizon runs off): if ``terminal`` then ``B_t = terminal_scores``
      (the engine already emitted the terminal; use the true label);
      else ``B_t = bootstrap_values[T - 1]`` (capped trajectory; 1-step
      lag is acceptable for the prototype).
    """
    t_len = active_seats.shape[0]
    if t_len == 0:
        return np.zeros((0, num_players), dtype=np.float32)
    # Per-seat per-step reward matrix r[p, t].
    rewards = np.zeros((num_players, t_len), dtype=np.float32)
    t_idx = np.arange(t_len)
    rewards[active_seats, t_idx] = per_step_rewards

    targets = np.zeros((t_len, num_players), dtype=np.float32)
    for t in range(t_len):
        k_max = min(n_step, t_len - t)
        gamma_pow = 1.0
        ret = np.zeros(num_players, dtype=np.float32)
        for k in range(k_max):
            ret += gamma_pow * rewards[:, t + k]
            gamma_pow *= gamma
        # gamma_pow is now gamma^k_max.
        if t + n_step < t_len:
            bootstrap = bootstrap_values[t + n_step]
        elif terminal:
            bootstrap = terminal_scores
        else:
            bootstrap = bootstrap_values[t_len - 1]
        ret += gamma_pow * bootstrap
        targets[t] = ret
    return np.clip(targets, -1.0, 1.0).astype(np.float32)


def _batch_value_head(
    net: torch.nn.Module,
    device: torch.device,
    grids: NDArray[np.float32],
    scalars: NDArray[np.float32],
    masks: NDArray[np.bool_],
    num_players: int,
) -> NDArray[np.float32]:
    """Forward all pending observations once, return ``(T, num_players)``.

    Scalar-value-head networks return ``(T, 1)``; we broadcast to
    ``(T, num_players)`` so the n-step math is uniform.
    """
    net.eval()
    g_t = torch.from_numpy(grids).to(device)
    s_t = torch.from_numpy(scalars).to(device)
    m_t = torch.from_numpy(masks).to(device)
    with torch.no_grad():
        _, values = net(g_t, s_t, m_t)
    arr = values.detach().cpu().numpy().astype(np.float32)
    if arr.shape[1] == 1:
        arr = np.repeat(arr, num_players, axis=1)
    return arr  # type: ignore[no-any-return]


def play_game(
    evaluator: NNEvaluator,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    seed: int | None = None,
    spawn_positions: list[tuple[int, int]] | None = None,
) -> list[Sample]:
    """Run one self-play game and return every visited state as a Sample.

    Each step runs an independent PUCT search from the current state; the
    visit vector at the root becomes the policy target. The value
    target depends on ``cfg.value_target_mode`` — see the module
    docstring.
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
            int,    # active_seat
            float,  # per_step_reward (normalized by board area)
        ]
    ] = []

    board_area = float(cfg.board_size * cfg.board_size)

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

        temperature = 1.0 if half_move < cfg.temperature_moves else 0.0
        action = _sample_action_from_visits(visits, temperature, rng)
        result = step(state, action, strict=False)
        claimed = int(result.info["claimed_this_turn"])
        per_step_r = float(claimed) / board_area

        samples_pending.append(
            (grid_obs, scalar_obs, mask, visits.copy(), active, per_step_r)
        )

        half_move += 1
        if first_enclosure_half_move is None and any(
            p.claimed_count > 0 for p in state.players
        ):
            first_enclosure_half_move = half_move

    terminal_scores = _normalized_final_scores(state)

    if cfg.value_target_mode == "terminal":
        # Legacy path: same per-seat vector for every sample.
        value_targets = np.broadcast_to(
            terminal_scores, (len(samples_pending), cfg.num_players)
        ).astype(np.float32)
    else:
        # n-step bootstrap target computed at self-play time using the
        # current self-play net. We already encoded each obs in the
        # loop, so one batched forward gives us all bootstrap values.
        t_len = len(samples_pending)
        if t_len == 0:
            value_targets = np.zeros((0, cfg.num_players), dtype=np.float32)
        else:
            grids = np.stack([p[0] for p in samples_pending], axis=0)
            scalars = np.stack([p[1] for p in samples_pending], axis=0)
            masks = np.stack([p[2] for p in samples_pending], axis=0)
            bootstrap = _batch_value_head(
                evaluator.net,
                evaluator.device,
                grids,
                scalars,
                masks,
                num_players=cfg.num_players,
            )
            active_seats = np.array([p[4] for p in samples_pending], dtype=np.int32)
            per_step_rewards = np.array(
                [p[5] for p in samples_pending], dtype=np.float32
            )
            value_targets = compute_nstep_value_targets(
                active_seats=active_seats,
                per_step_rewards=per_step_rewards,
                bootstrap_values=bootstrap,
                terminal_scores=terminal_scores,
                n_step=cfg.n_step,
                gamma=cfg.gamma,
                terminal=bool(state.done),
                num_players=cfg.num_players,
            )

    samples: list[Sample] = []
    for t, (g, s, m, v, _active, per_step_r) in enumerate(samples_pending):
        samples.append(
            Sample(
                grid=g,
                scalars=s,
                mask=m,
                visits=v,
                final_scores=value_targets[t],
                per_step_reward=per_step_r,
                step_index=t,
            )
        )
    return samples, first_enclosure_half_move


__all__ = [
    "SelfPlayConfig",
    "ValueTargetMode",
    "compute_nstep_value_targets",
    "play_game",
    "play_game_instrumented",
]
