"""Self-play training loop for :class:`TabularQAgent`.

Design choices:

- **Single shared Q-agent across all seats.** Symmetric game; shared table
  quadruples data efficiency. Per-call ``encode_state(state, player_id)``
  produces the current seat's view.
- **My-turn-only MDP.** When seat ``p`` moves at turn ``t`` and again at
  ``t'``, we treat everything in between as part of the environment and
  store one transition ``(s_t, a_t, accumulated_reward_from_p's_steps,
  s_{t'})``. No credit assignment across opponent turns.
- **Death handling.** When a seat is marked ``alive=False`` (illegal move
  or no legal actions), we accrue a one-time ``death_penalty`` and flush
  that seat's buffer on the very next opportunity — which is game-end,
  since dead seats never take another turn.
- **Terminal flush.** At game-end, every seat with an un-flushed
  ``(s, a)`` pair gets a terminal TD update with reward =
  ``accumulated + terminal_rank_bonus``.
- **Logging.** Episode metrics (return per seat, epsilon, Q-table size)
  go to both TensorBoard and ``episode_log.csv``. Periodic evaluation
  (vs random / greedy / UCT at ``eval_every``) goes to ``eval_curves.csv``
  and TensorBoard. Checkpoints every ``checkpoint_every`` episodes.
"""

from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from territory_takeover.actions import legal_action_mask
from territory_takeover.engine import new_game, step
from territory_takeover.search.random_agent import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
)

from .config import TrainConfig, save_config
from .eval import evaluate_vs, evaluate_vs_4p
from .q_agent import TabularQAgent
from .reward import death_penalty, step_reward, terminal_rank_bonus
from .state_encoder import encode_state

if TYPE_CHECKING:
    from territory_takeover.state import GameState


# --- Optional TensorBoard writer -----------------------------------------


class _Writer:
    """Lightweight TB shim: uses tensorboardX if available; else a no-op."""

    def __init__(self, logdir: Path) -> None:
        self._logdir = logdir
        self._tb: Any = None
        try:
            from tensorboardX import SummaryWriter

            logdir.mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(str(logdir))
        except Exception:  # pragma: no cover - TB is optional
            self._tb = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._tb is not None:
            self._tb.add_scalar(tag, float(value), int(step))

    def close(self) -> None:
        if self._tb is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._tb.close()


# --- CSV helpers ---------------------------------------------------------


def _append_csv(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            writer.writeheader()
        writer.writerow(row)


# --- Core training loop --------------------------------------------------


def _run_one_episode(
    agent: TabularQAgent,
    cfg: TrainConfig,
    rng_seed: int,
) -> tuple[int, list[float]]:
    """Play one self-play episode; return (turns, per-seat cumulative reward)."""
    state: GameState = new_game(
        board_size=cfg.board_size,
        num_players=cfg.num_players,
        spawn_positions=cfg.spawn_tuples(),
        seed=rng_seed,
    )

    # Per-seat rolling buffers.
    last_sa: list[tuple[tuple[int, int, int, int, int, int, int], int] | None] = [
        None
    ] * cfg.num_players
    pending_reward: list[float] = [0.0] * cfg.num_players
    ep_return: list[float] = [0.0] * cfg.num_players
    prev_alive = [True] * cfg.num_players

    while not state.done:
        seat = state.current_player
        current_key = encode_state(state, seat)

        # If we have a pending (s, a) from this seat's last move, flush it now
        # with the accumulated reward between then and now.
        pending_sa = last_sa[seat]
        if pending_sa is not None:
            prev_s, prev_a = pending_sa
            mask_now = legal_action_mask(state, seat)
            agent.td_update(
                prev_s, prev_a, pending_reward[seat], current_key, mask_now
            )
            pending_reward[seat] = 0.0

        action = agent.select_action(state, seat)

        step_result = step(state, action, strict=False)
        scaled = step_reward(step_result.reward, cfg.reward)
        pending_reward[seat] += scaled
        ep_return[seat] += scaled

        # Death penalty: if this move flipped the mover's alive False.
        if prev_alive[seat] and not state.players[seat].alive:
            penalty = death_penalty(len(state.players[seat].path), cfg.reward)
            pending_reward[seat] += penalty
            ep_return[seat] += penalty
            prev_alive[seat] = False

        # _advance_turn may also mark other seats dead on no-legal-moves.
        for other in range(cfg.num_players):
            if prev_alive[other] and not state.players[other].alive:
                penalty = death_penalty(
                    len(state.players[other].path), cfg.reward
                )
                pending_reward[other] += penalty
                ep_return[other] += penalty
                prev_alive[other] = False

        last_sa[seat] = (current_key, action)

    # Terminal: flush each seat with the rank bonus + any accumulated reward.
    for seat in range(cfg.num_players):
        pending_sa = last_sa[seat]
        if pending_sa is None:
            continue
        prev_s, prev_a = pending_sa
        bonus = terminal_rank_bonus(state, seat, cfg.reward)
        final_r = pending_reward[seat] + bonus
        ep_return[seat] += bonus
        agent.td_update(prev_s, prev_a, final_r, None, None)

    return state.turn_number, ep_return


# --- Evaluation ----------------------------------------------------------


def _evaluate_suite(
    agent: TabularQAgent,
    cfg: TrainConfig,
    episode: int,
    eval_seed: int,
) -> dict[str, float]:
    """Run the full eval suite (random, greedy, uct); return dict of metrics."""
    agent.set_greedy(True)
    results: dict[str, float] = {}

    rng = np.random.default_rng(eval_seed)
    rand = UniformRandomAgent(rng=np.random.default_rng(rng.integers(1 << 32)))
    greedy = HeuristicGreedyAgent(rng=np.random.default_rng(rng.integers(1 << 32)))

    if cfg.num_players == 2:
        if cfg.eval_games_vs_random > 0:
            r = evaluate_vs(
                agent,
                rand,
                cfg.eval_games_vs_random,
                cfg.board_size,
                cfg.spawn_tuples(),
                int(rng.integers(1 << 32)),
            )
            results["win_rate_vs_random"] = r["win_rate"]
            results["ci_low_vs_random"] = r["ci_low"]
            results["ci_high_vs_random"] = r["ci_high"]
        if cfg.eval_games_vs_greedy > 0:
            r = evaluate_vs(
                agent,
                greedy,
                cfg.eval_games_vs_greedy,
                cfg.board_size,
                cfg.spawn_tuples(),
                int(rng.integers(1 << 32)),
            )
            results["win_rate_vs_greedy"] = r["win_rate"]
            results["ci_low_vs_greedy"] = r["ci_low"]
            results["ci_high_vs_greedy"] = r["ci_high"]
        if cfg.eval_games_vs_uct > 0:
            # Import here so users without optional deps can still train.
            from territory_takeover.search.mcts.uct import UCTAgent

            uct = UCTAgent(
                iterations=cfg.uct_iters,
                rng=np.random.default_rng(rng.integers(1 << 32)),
            )
            r = evaluate_vs(
                agent,
                uct,
                cfg.eval_games_vs_uct,
                cfg.board_size,
                cfg.spawn_tuples(),
                int(rng.integers(1 << 32)),
            )
            results["win_rate_vs_uct"] = r["win_rate"]
            results["ci_low_vs_uct"] = r["ci_low"]
            results["ci_high_vs_uct"] = r["ci_high"]
    else:  # 4p diagnostic
        if cfg.eval_games_vs_random > 0:
            r = evaluate_vs_4p(
                agent,
                rand,
                cfg.eval_games_vs_random,
                cfg.board_size,
                cfg.spawn_tuples(),
                int(rng.integers(1 << 32)),
            )
            results["win_rate_vs_random_4p"] = r["win_rate"]
            results["ci_low_vs_random_4p"] = r["ci_low"]
            results["ci_high_vs_random_4p"] = r["ci_high"]
        if cfg.eval_games_vs_greedy > 0:
            r = evaluate_vs_4p(
                agent,
                greedy,
                cfg.eval_games_vs_greedy,
                cfg.board_size,
                cfg.spawn_tuples(),
                int(rng.integers(1 << 32)),
            )
            results["win_rate_vs_greedy_4p"] = r["win_rate"]
            results["ci_low_vs_greedy_4p"] = r["ci_low"]
            results["ci_high_vs_greedy_4p"] = r["ci_high"]

    agent.set_greedy(False)
    results["episode"] = float(episode)
    return results


# --- Run driver ----------------------------------------------------------


def _timestamp() -> str:
    return _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def train(cfg: TrainConfig, run_tag: str | None = None) -> Path:
    """Run one full training job per ``cfg`` and return the output directory.

    Writes ``config.yaml``, ``episode_log.csv``, ``eval_curves.csv``,
    ``q_table.pkl`` (final + checkpoints), and ``tb/`` TensorBoard events.
    """
    tag = run_tag or _timestamp()
    out_dir = Path(cfg.out_dir) / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    episode_csv = out_dir / "episode_log.csv"
    eval_csv = out_dir / "eval_curves.csv"
    writer = _Writer(out_dir / "tb")

    rng = np.random.default_rng(cfg.seed)
    agent = TabularQAgent(
        cfg=cfg.q, rng=np.random.default_rng(rng.integers(1 << 32)), name="tabQ"
    )

    start_wall = time.perf_counter()
    ep_start = time.perf_counter()
    for episode in range(cfg.num_episodes):
        agent.set_episode(episode)
        ep_seed = int(rng.integers(1 << 32))
        turns, returns = _run_one_episode(agent, cfg, ep_seed)

        # Lightweight per-episode log every 1k episodes to keep CSV bounded.
        if episode % 1_000 == 0 or episode == cfg.num_episodes - 1:
            elapsed = time.perf_counter() - ep_start
            ep_start = time.perf_counter()
            mean_return = float(sum(returns) / len(returns))
            row: dict[str, float | int | str] = {
                "episode": episode,
                "turns": turns,
                "mean_return": mean_return,
                "epsilon": agent.epsilon,
                "q_table_size": len(agent.q_table),
                "seconds_per_1k_ep": elapsed,
            }
            _append_csv(episode_csv, row)
            writer.scalar("episode/mean_return", mean_return, episode)
            writer.scalar("episode/epsilon", agent.epsilon, episode)
            writer.scalar("episode/q_table_size", float(len(agent.q_table)), episode)
            writer.scalar("episode/turns", float(turns), episode)

        # Evaluation.
        if cfg.eval_every > 0 and (
            (episode > 0 and episode % cfg.eval_every == 0)
            or episode == cfg.num_episodes - 1
        ):
            eval_seed = int(rng.integers(1 << 32))
            metrics = _evaluate_suite(agent, cfg, episode, eval_seed)
            for k, v in metrics.items():
                writer.scalar(f"eval/{k}", float(v), episode)
            _append_csv(eval_csv, {k: v for k, v in metrics.items()})

        # Checkpoint.
        if cfg.checkpoint_every > 0 and episode > 0 and (
            (episode % cfg.checkpoint_every == 0)
            or (episode == cfg.num_episodes - 1)
        ):
            ckpt = out_dir / f"q_table_ep{episode}.pkl"
            agent.save(ckpt)

    # Final save.
    agent.save(out_dir / "q_table.pkl")
    writer.close()

    total_time = time.perf_counter() - start_wall
    summary = {
        "total_seconds": total_time,
        "final_q_table_size": len(agent.q_table),
        "num_episodes": cfg.num_episodes,
        "config": asdict(cfg),
    }
    (out_dir / "summary.yaml").write_text(_yaml_dump(summary))
    return out_dir


def _yaml_dump(obj: object) -> str:
    import yaml

    out = yaml.safe_dump(obj, sort_keys=False)
    return str(out)


__all__ = ["train"]
