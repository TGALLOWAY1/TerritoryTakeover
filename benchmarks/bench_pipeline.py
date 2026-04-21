"""Method-level profiling harness for search + AlphaZero pipelines.

Where ``bench_engine.py`` measures engine primitives (``step``,
``state.copy``, ``legal_actions``, random rollouts), this harness profiles
whole-pipeline workloads — a full UCT game, one AlphaZero self-play +
train iteration — and buckets cumulative time by logical stage so the
optimization report can answer "what percentage of total time is spent
in each major part of the optimization pipeline?".

Usage::

    python benchmarks/bench_pipeline.py rollout --board 40 --games 50
    python benchmarks/bench_pipeline.py uct --board 10 --iters 200
    python benchmarks/bench_pipeline.py az --config \\
        configs/phase3c_alphazero_smoke.yaml
    python benchmarks/bench_pipeline.py all          # runs a standard suite

Profile outputs go to ``benchmarks/profiles/<workload>.json`` and the
raw pstats are emitted next to them as ``<workload>.pstats``. The JSON
is intentionally small and self-describing so the cost-breakdown report
can cite numbers by filename without re-parsing the pstats.

The buckets are derived from substring-matching the function-qualified
name against a deny-first, allow-last ladder so a function appears in
exactly one bucket. The last bucket (``other``) sweeps up anything that
did not match — we want it small but non-zero, and we list the top
unmatched functions in the JSON so the ladder can be extended if it
grows.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BENCH_DIR / "profiles"


# ---------- bucket rules -----------------------------------------------------
#
# A bucket rule is ``(bucket_name, predicate)`` where predicate is
# ``(filename, funcname) -> bool``. The first rule that matches wins, so
# order the ladder from most-specific to most-general. ``other`` is the
# implicit final bucket for anything that didn't match.

BucketRule = tuple[str, Callable[[str, str], bool]]


def _fn_contains(*needles: str) -> Callable[[str, str], bool]:
    def _pred(filename: str, funcname: str) -> bool:
        return any(n in funcname for n in needles)

    return _pred


def _file_and_fn(
    file_needle: str, *fn_needles: str
) -> Callable[[str, str], bool]:
    def _pred(filename: str, funcname: str) -> bool:
        return file_needle in filename and any(n in funcname for n in fn_needles)

    return _pred


def _file_any_of(*file_needles: str) -> Callable[[str, str], bool]:
    def _pred(filename: str, funcname: str) -> bool:
        return any(n in filename for n in file_needles)

    return _pred


def _builtin_contains(*fn_needles: str) -> Callable[[str, str], bool]:
    """Match ``~`` (C-implemented) entries by funcname substring."""

    def _pred(filename: str, funcname: str) -> bool:
        return filename == "~" and any(n in funcname for n in fn_needles)

    return _pred


def _or(*preds: Callable[[str, str], bool]) -> Callable[[str, str], bool]:
    def _pred(filename: str, funcname: str) -> bool:
        return any(p(filename, funcname) for p in preds)

    return _pred


# Ladder for search-style (UCT) workloads.
#
# Order matters: put the most specific engine-internal buckets first.
# ``state_copy`` is split into two predicates because cProfile reports
# methods by short funcname (``copy``) but the filename pins it to
# ``state.py``.
SEARCH_BUCKETS: list[BucketRule] = [
    ("enclosure", _file_and_fn("engine.py", "detect_and_apply_enclosure", "_bfs")),
    ("state_copy", _file_any_of("territory_takeover/state.py")),
    ("legal_actions", _file_and_fn(
        "actions.py",
        "legal_actions",
        "legal_action_mask",
        "has_any_legal_action",
    )),
    ("step_glue", _file_any_of("territory_takeover/engine.py")),
    ("rollout_mcts", _file_any_of("mcts/rollout.py")),
    ("mcts_selection", _file_and_fn("mcts/uct.py", "_select", "_ucb")),
    ("mcts_expansion", _file_and_fn("mcts/uct.py", "_expand", "populate_untried")),
    ("mcts_backprop", _file_and_fn("mcts/uct.py", "_backpropagate", "_run_iterations")),
    ("mcts_rave", _file_any_of("mcts/rave.py")),
    ("mcts_node", _file_any_of("mcts/node.py")),
    ("evaluator_heuristic", _file_any_of(
        "eval/heuristic.py", "eval/features.py", "eval/voronoi.py"
    )),
    ("random_rollout_fast", _file_any_of("territory_takeover/rollout.py")),
    # Numpy / deque / random primitives called from inside the engine —
    # their cost is effectively engine-attributable but cProfile reports
    # them as C builtins with filename ``~``. Split out so ``other`` stays
    # small and the report can cite them separately if needed.
    ("numpy_ops", _builtin_contains("numpy.", "ufunc")),
    ("python_stdlib", _or(
        _builtin_contains("deque", "random.", "Random"),
        _file_any_of("/python3.11/random.py", "/python3.11/collections"),
    )),
]


# Ladder for AlphaZero training workloads.
AZ_BUCKETS: list[BucketRule] = [
    ("enclosure", _file_and_fn("engine.py", "detect_and_apply_enclosure", "_bfs")),
    ("state_copy", _file_any_of("territory_takeover/state.py")),
    ("legal_actions", _file_and_fn(
        "actions.py",
        "legal_actions",
        "legal_action_mask",
        "has_any_legal_action",
    )),
    ("step_glue", _file_any_of("territory_takeover/engine.py")),
    ("mcts_puct", _file_any_of("alphazero/mcts.py")),
    ("evaluator_cache", _file_any_of("alphazero/evaluator.py")),
    ("observation_encode", _file_any_of("alphazero/spaces.py")),
    ("replay_buffer", _file_any_of("alphazero/replay.py")),
    ("selfplay_driver", _file_any_of("alphazero/selfplay.py")),
    ("nn_module", _file_any_of("alphazero/network.py")),
    ("nn_train", _file_any_of("alphazero/train.py")),
    # One-time module-loading cost (posix.stat, marshal.loads, tokenize,
    # importlib, sympy). Torch has a large lazy-import graph that fires
    # on first use; isolated so it can be excluded from per-iteration
    # cost totals.
    ("import_overhead", _or(
        _builtin_contains(
            "posix.", "marshal.", "io.open_code", "exec",
            "__build_class__",
        ),
        _file_any_of(
            "/importlib", "/tokenize.py", "<frozen importlib",
            "<frozen posixpath", "/sympy/",
        ),
    )),
    # Torch / numpy primitives: cluster by the broad category cProfile
    # shows for C-level ops.
    ("torch_ops", _builtin_contains("Tensor", "torch", "autograd", "_C.", "nn.functional")),
    ("numpy_ops", _builtin_contains("numpy.", "ufunc")),
    ("python_stdlib", _or(
        _builtin_contains("deque", "random.", "Random"),
        _file_any_of("/python3.11/random.py", "/python3.11/collections"),
    )),
    # Any other torch-nn code.
    ("torch_nn", _file_any_of("/torch/nn/", "/torch/optim/")),
    ("torch_other", _file_any_of("/torch/")),
]


# ---------- profile bucketing -----------------------------------------------


@dataclass
class BucketStats:
    time_s: float
    ncalls: int

    def to_dict(self, total_s: float) -> dict[str, Any]:
        pct = 100.0 * self.time_s / total_s if total_s > 0 else 0.0
        return {
            "time_s": round(self.time_s, 6),
            "ncalls": self.ncalls,
            "pct": round(pct, 2),
        }


def _bucket_profile(
    profiler: cProfile.Profile,
    rules: list[BucketRule],
    top_n: int = 15,
) -> dict[str, Any]:
    """Reduce a cProfile.Profile into per-bucket totals using ``rules``.

    We use ``tottime`` (self time excluding sub-calls) to avoid the
    double-counting a caller-callee pair incurs under ``cumtime``. With
    disjoint buckets this gives an exact partition of wall-clock into
    one bucket per function.
    """
    stats = pstats.Stats(profiler)
    total_tottime = 0.0
    buckets: dict[str, BucketStats] = {name: BucketStats(0.0, 0) for name, _ in rules}
    buckets["other"] = BucketStats(0.0, 0)

    top_fns: list[tuple[str, float, float, int]] = []
    unmatched_fns: list[tuple[str, float, int]] = []

    for key, values in stats.stats.items():  # type: ignore[attr-defined]
        filename, _lineno, funcname = key
        _cc, ncalls, tottime, _cumtime, _callers = values
        total_tottime += tottime

        assigned: str | None = None
        for name, pred in rules:
            try:
                if pred(filename, funcname):
                    assigned = name
                    break
            except Exception:
                continue
        if assigned is None:
            assigned = "other"
            unmatched_fns.append((f"{filename}:{funcname}", tottime, int(ncalls)))

        buckets[assigned].time_s += float(tottime)
        buckets[assigned].ncalls += int(ncalls)

        top_fns.append((f"{filename}:{funcname}", float(tottime), float(_cumtime), int(ncalls)))

    top_fns.sort(key=lambda r: r[1], reverse=True)
    unmatched_fns.sort(key=lambda r: r[1], reverse=True)

    return {
        "total_tottime_s": round(total_tottime, 6),
        "buckets": {k: v.to_dict(total_tottime) for k, v in buckets.items()},
        "top_tottime": [
            {"name": name, "tottime_s": round(tt, 6), "cumtime_s": round(ct, 6), "ncalls": nc}
            for (name, tt, ct, nc) in top_fns[:top_n]
        ],
        "top_unmatched": [
            {"name": name, "tottime_s": round(tt, 6), "ncalls": nc}
            for (name, tt, nc) in unmatched_fns[:top_n]
        ],
    }


def _write_profile(
    profiler: cProfile.Profile,
    summary: dict[str, Any],
    workload: str,
) -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    pstats_path = PROFILES_DIR / f"{workload}.pstats"
    profiler.dump_stats(str(pstats_path))
    json_path = PROFILES_DIR / f"{workload}.json"
    summary["workload"] = workload
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"  wrote {json_path}")
    return json_path


# ---------- workload: rollout throughput -------------------------------------


def profile_rollout(board: int, games: int, seed: int = 0) -> dict[str, Any]:
    """Profile ``simulate_random_rollout`` from a mid-game state.

    Baseline anchor — this is the "engine only" workload with no search
    or learning overhead. Compare to the UCT and AlphaZero numbers to
    see how much method-level overhead each pipeline adds.
    """
    from territory_takeover import new_game, simulate_random_rollout, step
    from territory_takeover.actions import legal_actions

    # Build one mid-game state, then re-copy it ``games`` times.
    base = new_game(board_size=board, num_players=4, seed=seed)
    warm_rng = random.Random(seed)
    for _ in range(min(200, board * board // 4)):
        if base.done:
            break
        acts = legal_actions(base, base.current_player)
        if not acts:
            break
        step(base, warm_rng.choice(acts))

    rng = random.Random(seed + 1)
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    for _ in range(games):
        s = base.copy()
        simulate_random_rollout(s, rng)
    profiler.disable()
    wall_s = time.perf_counter() - t0

    summary = _bucket_profile(profiler, SEARCH_BUCKETS)
    summary["wall_time_s"] = round(wall_s, 6)
    summary["workload_params"] = {"board": board, "games": games, "seed": seed}
    summary["games_per_sec"] = round(games / wall_s, 3) if wall_s > 0 else None

    _write_profile(profiler, summary, f"rollout_{board}")
    return summary


# ---------- workload: UCT full game ------------------------------------------


def profile_uct_game(
    board: int,
    iters: int,
    num_players: int = 2,
    rollout_kind: str = "uniform",
    seed: int = 0,
) -> dict[str, Any]:
    """Play one full UCT-vs-Random game and profile it.

    UCT sits at seat 0; the remaining seats play uniform-random. The
    profile reflects the per-move MCTS cost plus the environment step
    cost, which is the dominant workload during tournament evaluation.
    """
    from territory_takeover import new_game, step
    from territory_takeover.actions import legal_actions
    from territory_takeover.search import UCTAgent, UniformRandomAgent

    state = new_game(board_size=board, num_players=num_players, seed=seed)

    np_rng = np.random.default_rng(seed)
    uct = UCTAgent(
        iterations=iters,
        c=1.4,
        rollout_kind=rollout_kind,
        rng=np_rng,
        reuse_tree=True,
    )
    agents = [uct] + [
        UniformRandomAgent(rng=np.random.default_rng(seed + 1 + i))
        for i in range(num_players - 1)
    ]

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    half_moves = 0
    while not state.done:
        pid = state.current_player
        agent = agents[pid]
        legal = legal_actions(state, pid)
        if not legal:
            step(state, 0, strict=False)
            half_moves += 1
            continue
        action = agent.select_action(state, pid)
        step(state, action, strict=False)
        half_moves += 1
        if half_moves > board * board * 3:
            break
    profiler.disable()
    wall_s = time.perf_counter() - t0

    summary = _bucket_profile(profiler, SEARCH_BUCKETS)
    summary["wall_time_s"] = round(wall_s, 6)
    summary["workload_params"] = {
        "board": board,
        "iters": iters,
        "num_players": num_players,
        "rollout_kind": rollout_kind,
        "seed": seed,
    }
    summary["half_moves"] = half_moves
    summary["moves_per_sec"] = round(half_moves / wall_s, 3) if wall_s > 0 else None

    _write_profile(profiler, summary, f"uct_{board}_{iters}")
    return summary


# ---------- workload: AlphaZero one iteration --------------------------------


def profile_alphazero_iteration(config_path: Path, seed: int = 0) -> dict[str, Any]:
    """Profile one AlphaZero train_loop iteration end-to-end.

    Drives self-play, replay-buffer fill, and one batch of SGD updates.
    Uses the provided config so a smoke config (6x6, tiny net) runs in
    seconds while a larger config can still be profiled if the operator
    has the wall-clock budget.
    """
    import yaml

    from territory_takeover.rl.alphazero.network import AZNetConfig
    from territory_takeover.rl.alphazero.selfplay import SelfPlayConfig
    from territory_takeover.rl.alphazero.train import TrainConfig, train_loop

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    # Force exactly one iteration so the profile reflects the inner loop,
    # not a multi-iteration run that amortizes startup costs.
    tr = dict(raw.get("train", {}))
    tr["num_iterations"] = 1
    train_cfg = TrainConfig(**tr)

    net_cfg = AZNetConfig(
        board_size=int(raw["board_size"]),
        num_players=int(raw["num_players"]),
        **raw.get("net", {}),
    )
    self_play_cfg = SelfPlayConfig(
        board_size=int(raw["board_size"]),
        num_players=int(raw["num_players"]),
        **raw.get("self_play", {}),
    )

    out_dir = PROFILES_DIR / f"az_iteration_{net_cfg.board_size}_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm up torch so the profile doesn't get dominated by one-time
    # module-import / codegen overhead (sympy, tokenize, marshal.loads)
    # that only fires on the first torch forward / backward pass of the
    # process. A full throw-away train_loop call exercises every import
    # path the profiled run will hit.
    warmup_train_cfg = TrainConfig(
        num_iterations=1,
        games_per_iteration=1,
        train_steps_per_iteration=1,
        batch_size=max(1, min(2, train_cfg.batch_size)),
        learning_rate=train_cfg.learning_rate,
        l2_weight=train_cfg.l2_weight,
        value_loss_coef=train_cfg.value_loss_coef,
        buffer_capacity=max(8, train_cfg.buffer_capacity // 32),
        snapshot_every=10,
    )
    warmup_dir = out_dir / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    train_loop(
        net_cfg,
        warmup_train_cfg,
        self_play_cfg,
        out_dir=warmup_dir,
        seed=seed,
        device="cpu",
    )

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    train_loop(
        net_cfg,
        train_cfg,
        self_play_cfg,
        out_dir=out_dir,
        seed=seed,
        device="cpu",
    )
    profiler.disable()
    wall_s = time.perf_counter() - t0

    summary = _bucket_profile(profiler, AZ_BUCKETS)
    summary["wall_time_s"] = round(wall_s, 6)
    summary["workload_params"] = {
        "config": str(config_path),
        "board": net_cfg.board_size,
        "num_players": net_cfg.num_players,
        "puct_iterations": self_play_cfg.puct_iterations,
        "games_per_iteration": train_cfg.games_per_iteration,
        "train_steps_per_iteration": train_cfg.train_steps_per_iteration,
        "batch_size": train_cfg.batch_size,
        "channels": net_cfg.channels,
        "num_res_blocks": net_cfg.num_res_blocks,
        "seed": seed,
    }

    _write_profile(profiler, summary, f"az_iteration_{net_cfg.board_size}")
    return summary


# ---------- CLI --------------------------------------------------------------


def _print_bucket_table(label: str, summary: dict[str, Any]) -> None:
    print(f"\n{label}  (wall {summary.get('wall_time_s', 0):.2f}s, "
          f"tot_tottime {summary['total_tottime_s']:.2f}s)")
    rows: list[tuple[str, float, int]] = []
    for name, b in summary["buckets"].items():
        rows.append((name, float(b["pct"]), int(b["ncalls"])))
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"  {'bucket':<22} {'%':>6} {'ncalls':>10}")
    print(f"  {'-' * 22} {'-' * 6} {'-' * 10}")
    for name, pct, nc in rows:
        print(f"  {name:<22} {pct:>6.2f} {nc:>10}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("rollout", help="profile simulate_random_rollout")
    pr.add_argument("--board", type=int, default=40)
    pr.add_argument("--games", type=int, default=50)
    pr.add_argument("--seed", type=int, default=0)

    pu = sub.add_parser("uct", help="profile a full UCT-vs-Random game")
    pu.add_argument("--board", type=int, default=10)
    pu.add_argument("--iters", type=int, default=200)
    pu.add_argument("--num-players", type=int, default=2)
    pu.add_argument("--rollout", type=str, default="uniform",
                    choices=["uniform", "informed", "voronoi"])
    pu.add_argument("--seed", type=int, default=0)

    pa = sub.add_parser("az", help="profile one AlphaZero iteration")
    pa.add_argument("--config", type=Path, required=True)
    pa.add_argument("--seed", type=int, default=0)

    sub.add_parser("all", help="run the standard suite used by the report")

    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.cmd == "rollout":
        s = profile_rollout(args.board, args.games, args.seed)
        _print_bucket_table(f"rollout board={args.board} games={args.games}", s)
    elif args.cmd == "uct":
        s = profile_uct_game(
            args.board, args.iters, args.num_players, args.rollout, args.seed
        )
        _print_bucket_table(
            f"uct board={args.board} iters={args.iters} "
            f"players={args.num_players} rollout={args.rollout}",
            s,
        )
    elif args.cmd == "az":
        s = profile_alphazero_iteration(args.config, args.seed)
        _print_bucket_table(f"alphazero config={args.config.name}", s)
    elif args.cmd == "all":
        results: list[tuple[str, dict[str, Any]]] = []
        results.append((
            "rollout board=20 games=50",
            profile_rollout(20, 50),
        ))
        results.append((
            "rollout board=40 games=30",
            profile_rollout(40, 30),
        ))
        results.append((
            "uct board=10 iters=200 p=2",
            profile_uct_game(10, 200, num_players=2),
        ))
        results.append((
            "uct board=20 iters=100 p=2",
            profile_uct_game(20, 100, num_players=2),
        ))
        cfg = BENCH_DIR.parent / "configs" / "phase3c_alphazero_smoke.yaml"
        if cfg.exists():
            results.append((
                f"alphazero {cfg.name}",
                profile_alphazero_iteration(cfg),
            ))
        for label, summary in results:
            _print_bucket_table(label, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
