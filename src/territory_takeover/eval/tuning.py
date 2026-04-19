"""Black-box tuning of :class:`LinearEvaluator` weights via (1+lambda)-ES.

:func:`tune_weights` searches a six-dimensional box over the feature keys
exposed by :class:`territory_takeover.eval.heuristic.LinearEvaluator`,
using win rate against a fixed pool of ``opponent_agents`` as fitness.
The search is a plain (1+lambda) evolution strategy: each generation
the parent spawns ``population_size - 1`` Gaussian-perturbed children
(bound-normalized per-axis sigma) and the best child replaces the parent
iff it beats the parent's freshly re-measured win rate. This keeps the
search dependency-free (numpy only) while handling the stochastic
fitness robustly.

Every evaluated candidate is recorded to a JSONL log; the ``seed`` field
on each record is the exact seed that drove the underlying
:func:`territory_takeover.search.harness.run_match` call, so any single
log line can be replayed bit-for-bit.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from territory_takeover.eval.heuristic import default_evaluator
from territory_takeover.search.harness import run_match

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.search.agent import Agent


FEATURE_KEYS: tuple[str, ...] = (
    "territory_total",
    "reachable_area",
    "mobility",
    "enclosure_potential_area",
    "choke_pressure",
    "opponent_distance",
)

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {k: (-2.0, 2.0) for k in FEATURE_KEYS}

DEFAULT_LOG_PATH: Path = Path("results/weight_tuning.jsonl")

_SIGMA_FLOOR: float = 0.05


@dataclass(frozen=True)
class TuningConfig:
    """Resolved hyperparameters for a single :func:`tune_weights` run."""

    num_generations: int
    population_size: int
    games_per_eval: int
    board_size: int
    num_players: int
    bounds: dict[str, tuple[float, float]]
    sigma0: float
    sigma_decay: float
    seed: int
    parallel: bool
    log_path: Path


@dataclass(frozen=True)
class CandidateRecord:
    """One evaluation's worth of data, appended to the JSONL log."""

    generation: int
    candidate_index: int
    weights: dict[str, float]
    wins: int
    games: int
    win_rate: float
    seed: int
    elapsed_s: float


# --- Helpers ---------------------------------------------------------------


def _clip_weights(
    weights: dict[str, float], bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
    clipped: dict[str, float] = {}
    for key in FEATURE_KEYS:
        lo, hi = bounds[key]
        clipped[key] = float(min(hi, max(lo, weights[key])))
    return clipped


def _weights_to_vec(weights: dict[str, float]) -> NDArray[np.float64]:
    return np.array([weights[k] for k in FEATURE_KEYS], dtype=np.float64)


def _vec_to_weights(vec: NDArray[np.float64]) -> dict[str, float]:
    return {k: float(vec[i]) for i, k in enumerate(FEATURE_KEYS)}


def _bounds_span(bounds: dict[str, tuple[float, float]]) -> NDArray[np.float64]:
    return np.array([hi - lo for k in FEATURE_KEYS for lo, hi in (bounds[k],)], dtype=np.float64)


def _mutate(
    parent_vec: NDArray[np.float64],
    sigma_scalar: float,
    span: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    noise = rng.standard_normal(parent_vec.shape[0])
    return parent_vec + sigma_scalar * span * noise


def _resolve_bounds(
    bounds: dict[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    if bounds is None:
        return dict(DEFAULT_BOUNDS)
    missing = [k for k in FEATURE_KEYS if k not in bounds]
    if missing:
        raise ValueError(
            f"bounds missing keys: {missing}; bounds must cover every feature in FEATURE_KEYS"
        )
    return {k: (float(bounds[k][0]), float(bounds[k][1])) for k in FEATURE_KEYS}


# --- Fitness oracle --------------------------------------------------------


def _evaluate_candidate(
    weights: dict[str, float],
    base_agent_factory: Callable[[dict[str, float]], Agent],
    opponent_agents: list[Agent],
    games_per_eval: int,
    board_size: int,
    num_players: int,
    eval_seed: int,
    parallel: bool,
) -> tuple[int, int]:
    """Return ``(wins, games)`` summed across every opponent in the pool."""
    if not opponent_agents:
        raise ValueError("opponent_agents must be non-empty")
    candidate = base_agent_factory(weights)

    sub_seqs = np.random.SeedSequence(eval_seed).spawn(len(opponent_agents))

    total_wins = 0
    total_games = 0
    for opp_idx, opponent in enumerate(opponent_agents):
        sub_seed = int(sub_seqs[opp_idx].generate_state(1, dtype=np.uint32)[0])
        result = run_match(
            agents=[candidate, opponent],
            num_games=games_per_eval,
            board_size=board_size,
            swap_seats=True,
            seed=sub_seed,
            parallel=parallel,
            num_players=num_players,
        )
        total_wins += result.per_agent[0].wins
        total_games += result.num_games
    return total_wins, total_games


# --- JSONL log -------------------------------------------------------------


def _append_record(log_path: Path, record: CandidateRecord) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        json.dump(
            {
                "generation": record.generation,
                "candidate_index": record.candidate_index,
                "weights": record.weights,
                "wins": record.wins,
                "games": record.games,
                "win_rate": record.win_rate,
                "seed": record.seed,
                "elapsed_s": record.elapsed_s,
            },
            f,
            sort_keys=True,
        )
        f.write("\n")


# --- Public entry point ----------------------------------------------------


def tune_weights(
    base_agent_factory: Callable[[dict[str, float]], Agent],
    opponent_agents: list[Agent],
    num_generations: int = 20,
    population_size: int = 16,
    games_per_eval: int = 12,
    board_size: int = 20,
    num_players: int = 2,
    seed: int = 0,
    bounds: dict[str, tuple[float, float]] | None = None,
    sigma0: float = 0.3,
    sigma_decay: float = 0.9,
    parallel: bool = False,
    log_path: Path | str = DEFAULT_LOG_PATH,
    resume: bool = False,
) -> dict[str, float]:
    """Optimize ``LinearEvaluator`` weights with a (1+lambda)-ES.

    ``base_agent_factory`` is called in the main process with a weights
    dict and must return a fully-constructed :class:`Agent`. The
    resulting agent is handed to :func:`run_match`, which pickles it for
    each game — so the factory closure itself does not need to cross
    process boundaries, but the returned agent does.

    The search space is the Cartesian product of ``bounds[key]`` over
    ``FEATURE_KEYS``. Generation 0 pins the parent to
    :func:`default_evaluator` so the search never regresses below the
    hand-tuned baseline. The parent is re-evaluated every generation
    because fitness is stochastic.

    Returns the best-seen weights dict. The full search log (one record
    per evaluated candidate) is written to ``log_path`` in JSONL format
    and contains the exact seed used, so any candidate can be replayed.
    """
    if num_generations < 1:
        raise ValueError(f"num_generations must be >= 1; got {num_generations}")
    if population_size < 2:
        raise ValueError(f"population_size must be >= 2; got {population_size}")
    if games_per_eval < 2 or games_per_eval % 2 != 0:
        raise ValueError(
            f"games_per_eval must be even and >= 2; got {games_per_eval} "
            "(swap_seats=True requires num_games to be a multiple of len(agents)=2)"
        )
    if sigma0 <= 0.0:
        raise ValueError(f"sigma0 must be > 0; got {sigma0}")
    if not 0.0 < sigma_decay <= 1.0:
        raise ValueError(f"sigma_decay must be in (0, 1]; got {sigma_decay}")

    resolved_bounds = _resolve_bounds(bounds)
    resolved_log_path = Path(log_path)
    cfg = TuningConfig(
        num_generations=num_generations,
        population_size=population_size,
        games_per_eval=games_per_eval,
        board_size=board_size,
        num_players=num_players,
        bounds=resolved_bounds,
        sigma0=sigma0,
        sigma_decay=sigma_decay,
        seed=seed,
        parallel=parallel,
        log_path=resolved_log_path,
    )

    if not resume and resolved_log_path.exists():
        resolved_log_path.unlink()

    span = _bounds_span(resolved_bounds)

    root_ss = np.random.SeedSequence(cfg.seed)
    gen_seqs = root_ss.spawn(cfg.num_generations + 1)
    search_rng = np.random.default_rng(gen_seqs[-1])

    parent_weights = _clip_weights(default_evaluator().weights, resolved_bounds)
    parent_vec = _weights_to_vec(parent_weights)
    # Sentinel: parent gets a real win rate during gen 0's candidate-0 eval.
    parent_win_rate = -1.0

    best_weights = dict(parent_weights)
    best_win_rate = -1.0

    sigma_scalar = cfg.sigma0

    for g in range(cfg.num_generations):
        cand_seqs = gen_seqs[g].spawn(cfg.population_size)

        # Candidate 0: re-evaluate the parent with this generation's seed.
        parent_seed = int(cand_seqs[0].generate_state(1, dtype=np.uint32)[0])
        t0 = time.perf_counter()
        wins, games = _evaluate_candidate(
            parent_weights,
            base_agent_factory,
            opponent_agents,
            cfg.games_per_eval,
            cfg.board_size,
            cfg.num_players,
            parent_seed,
            cfg.parallel,
        )
        elapsed = time.perf_counter() - t0
        parent_win_rate = wins / games if games > 0 else 0.0
        _append_record(
            resolved_log_path,
            CandidateRecord(
                generation=g,
                candidate_index=0,
                weights=dict(parent_weights),
                wins=wins,
                games=games,
                win_rate=parent_win_rate,
                seed=parent_seed,
                elapsed_s=elapsed,
            ),
        )
        if parent_win_rate > best_win_rate:
            best_win_rate = parent_win_rate
            best_weights = dict(parent_weights)

        best_child_vec = parent_vec
        best_child_weights = parent_weights
        best_child_win_rate = -1.0
        best_child_wins = -1

        for c in range(1, cfg.population_size):
            child_vec = _mutate(parent_vec, sigma_scalar, span, search_rng)
            child_weights = _clip_weights(_vec_to_weights(child_vec), resolved_bounds)
            child_vec = _weights_to_vec(child_weights)

            eval_seed = int(cand_seqs[c].generate_state(1, dtype=np.uint32)[0])
            t0 = time.perf_counter()
            wins, games = _evaluate_candidate(
                child_weights,
                base_agent_factory,
                opponent_agents,
                cfg.games_per_eval,
                cfg.board_size,
                cfg.num_players,
                eval_seed,
                cfg.parallel,
            )
            elapsed = time.perf_counter() - t0
            win_rate = wins / games if games > 0 else 0.0

            _append_record(
                resolved_log_path,
                CandidateRecord(
                    generation=g,
                    candidate_index=c,
                    weights=dict(child_weights),
                    wins=wins,
                    games=games,
                    win_rate=win_rate,
                    seed=eval_seed,
                    elapsed_s=elapsed,
                ),
            )

            better = win_rate > best_child_win_rate or (
                win_rate == best_child_win_rate and wins > best_child_wins
            )
            if better:
                best_child_vec = child_vec
                best_child_weights = child_weights
                best_child_win_rate = win_rate
                best_child_wins = wins

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_weights = dict(child_weights)

        # Selection: promote best child iff it beat the parent this generation.
        if best_child_win_rate > parent_win_rate:
            parent_vec = best_child_vec
            parent_weights = best_child_weights

        sigma_scalar = max(_SIGMA_FLOOR, sigma_scalar * cfg.sigma_decay)

    return best_weights


__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_LOG_PATH",
    "FEATURE_KEYS",
    "CandidateRecord",
    "TuningConfig",
    "tune_weights",
]
