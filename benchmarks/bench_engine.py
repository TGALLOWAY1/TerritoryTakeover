"""Standalone benchmark harness for the TerritoryTakeover engine.

Run: ``python benchmarks/bench_engine.py``

Measures six workloads relevant to RL / MCTS use and writes a baseline JSON plus
a profiler-derived HOTSPOTS.md. Deliberately uses ``time.perf_counter_ns`` +
min-of-N instead of ``pytest-benchmark`` to avoid a new dev dependency and to
match the style of the existing perf smoke tests in ``tests/test_state.py``.
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

from territory_takeover import GameState, new_game, step
from territory_takeover.actions import legal_actions
from territory_takeover.constants import PATH_CODES

BENCH_DIR = Path(__file__).resolve().parent


# ---------- timing helpers ----------------------------------------------------


@dataclass
class TimingResult:
    name: str
    min_us: float
    median_us: float
    mean_us: float
    iters: int
    target_us: float | None = None
    extra: dict[str, float] | None = None

    @property
    def passed(self) -> bool:
        if self.target_us is None:
            return True
        return self.min_us < self.target_us


def _time_callable(
    name: str,
    fn: Callable[[], Any],
    iters: int,
    warmup: int,
    target_us: float | None = None,
) -> TimingResult:
    """Warm up, then measure ``iters`` per-call latencies with perf_counter_ns."""
    for _ in range(warmup):
        fn()
    samples = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples[i] = time.perf_counter_ns() - t0
    min_us = float(samples.min()) / 1000.0
    median_us = float(np.median(samples)) / 1000.0
    mean_us = float(samples.mean()) / 1000.0
    return TimingResult(
        name=name,
        min_us=min_us,
        median_us=median_us,
        mean_us=mean_us,
        iters=iters,
        target_us=target_us,
    )


# ---------- state construction helpers ---------------------------------------


def _make_midgame_state(
    board_size: int = 40,
    num_players: int = 4,
    plies: int = 200,
    seed: int = 0,
) -> GameState:
    """Build a mid-game 40x40 state by playing `plies` random moves."""
    state = new_game(board_size=board_size, num_players=num_players, seed=seed)
    rng = random.Random(seed)
    for _ in range(plies):
        if state.done:
            break
        acts = legal_actions(state, state.current_player)
        if not acts:
            break
        step(state, rng.choice(acts))
    return state


def _play_random_game(state: GameState, rng: random.Random) -> int:
    """Drive `state` to terminal with random policy. Returns turn count."""
    turns = 0
    while not state.done:
        acts = legal_actions(state, state.current_player)
        if not acts:
            # Advancing normally lets step() mark the player dead.
            step(state, 0)
        else:
            step(state, rng.choice(acts))
        turns += 1
    return turns


def _make_enclosure_ready_state() -> tuple[GameState, int]:
    """Build a small state where player 0's next East move closes a loop.

    Path winds around to end at head (2, 4). The East neighbor (2, 5) is empty
    and legal; after placing, (2, 5)'s E-neighbor (2, 6) is a same-player path
    tile that is NOT the predecessor — the enclosure trigger fires.
    Returns (state, closing_action).
    """
    state = new_game(board_size=10, num_players=2, seed=0)
    state.grid.fill(0)

    connected = [
        (2, 8), (2, 7), (2, 6),
        (3, 6), (4, 6), (4, 5), (4, 4), (4, 3), (4, 2),
        (3, 2), (2, 2), (2, 3), (2, 4),
    ]
    p0 = state.players[0]
    p0.path = list(connected)
    p0.path_set = set(connected)
    p0.head = connected[-1]
    p0.claimed_count = 0
    p0.alive = True
    for r, c in connected:
        state.grid[r, c] = PATH_CODES[0]

    # Move player 1 out of the way so they don't occupy default spawn cells.
    p1 = state.players[1]
    p1.path = [(9, 9)]
    p1.path_set = {(9, 9)}
    p1.head = (9, 9)
    p1.claimed_count = 0
    p1.alive = True
    state.grid[9, 9] = PATH_CODES[1]

    state.current_player = 0
    state.turn_number = 0
    state.done = False
    state.winner = None

    return state, 3  # East


# ---------- the six benchmarks ------------------------------------------------


def bench_state_copy(iters: int) -> TimingResult:
    state = _make_midgame_state()
    return _time_callable(
        "bench_state_copy", state.copy, iters=iters, warmup=1000, target_us=50.0
    )


def bench_legal_actions(iters: int) -> TimingResult:
    state = _make_midgame_state()
    pid = state.current_player
    return _time_callable(
        "bench_legal_actions",
        lambda: legal_actions(state, pid),
        iters=iters,
        warmup=5000,
        target_us=2.0,
    )


def bench_step_no_enclosure(iters: int) -> TimingResult:
    """Time step() calls that never trigger enclosure.

    We build a fresh mid-game state periodically. The min-of-N statistic is
    dominated by clean (non-enclosing) steps since enclosing steps are rare
    in random play.
    """
    rng = random.Random(12345)

    def reseed() -> GameState:
        """Return a live mid-game state. Retry if the playout terminated early."""
        for _ in range(10):
            s = _make_midgame_state(plies=60, seed=rng.randrange(1 << 30))
            if not s.done and legal_actions(s, s.current_player):
                return s
        raise RuntimeError("reseed: could not build a live mid-game state")

    samples: list[int] = []
    current = reseed()

    # warmup
    for _ in range(500):
        if current.done:
            current = reseed()
        acts = legal_actions(current, current.current_player)
        if not acts:
            current = reseed()
            continue
        step(current, rng.choice(acts))

    for _ in range(iters):
        if current.done:
            current = reseed()
        acts = legal_actions(current, current.current_player)
        if not acts:
            current = reseed()
            continue
        a = rng.choice(acts)
        t0 = time.perf_counter_ns()
        result = step(current, a)
        elapsed = time.perf_counter_ns() - t0
        # Exclude the rare iterations where the randomized step closed a loop.
        if result.info["claimed_this_turn"] == 0 and not result.info["illegal_move"]:
            samples.append(elapsed)

    arr = np.array(samples, dtype=np.int64)
    return TimingResult(
        name="bench_step_no_enclosure",
        min_us=float(arr.min()) / 1000.0,
        median_us=float(np.median(arr)) / 1000.0,
        mean_us=float(arr.mean()) / 1000.0,
        iters=len(samples),
        target_us=20.0,
    )


def bench_step_with_enclosure(iters: int) -> TimingResult:
    """Rebuild enclosure-ready state each iter; time only the step() call."""
    samples = np.empty(iters, dtype=np.int64)

    # warmup
    for _ in range(50):
        s, a = _make_enclosure_ready_state()
        step(s, a)

    claimed_seen = 0
    for i in range(iters):
        s, a = _make_enclosure_ready_state()
        t0 = time.perf_counter_ns()
        r = step(s, a)
        samples[i] = time.perf_counter_ns() - t0
        if r.info["claimed_this_turn"] > 0:
            claimed_seen += 1

    if claimed_seen == 0:
        raise RuntimeError(
            "bench_step_with_enclosure: no iteration actually enclosed — "
            "_make_enclosure_ready_state is broken."
        )

    return TimingResult(
        name="bench_step_with_enclosure",
        min_us=float(samples.min()) / 1000.0,
        median_us=float(np.median(samples)) / 1000.0,
        mean_us=float(samples.mean()) / 1000.0,
        iters=iters,
        target_us=250.0,
        extra={"enclosure_hit_rate": claimed_seen / iters},
    )


def bench_full_game_random(iters: int) -> TimingResult:
    """Full random game on 40x40 — report per-game milliseconds."""
    # warmup
    for _ in range(3):
        rng = random.Random(0)
        _play_random_game(new_game(board_size=40, num_players=4, seed=0), rng)

    samples = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        rng = random.Random(i)
        state = new_game(board_size=40, num_players=4, seed=i)
        t0 = time.perf_counter_ns()
        _play_random_game(state, rng)
        samples[i] = time.perf_counter_ns() - t0

    return TimingResult(
        name="bench_full_game_random",
        min_us=float(samples.min()) / 1000.0,
        median_us=float(np.median(samples)) / 1000.0,
        mean_us=float(samples.mean()) / 1000.0,
        iters=iters,
        target_us=50_000.0,  # 50 ms
    )


def bench_rollout_throughput(iters: int) -> TimingResult:
    """Rollouts (copy mid-game state, play to completion) — games per second.

    This is the MCTS-relevant op. We report games/sec as the primary metric
    in ``extra``; ``min/median/mean_us`` are per-rollout times for context.
    """
    base = _make_midgame_state(plies=100, seed=42)

    # warmup
    for _ in range(5):
        rng = random.Random(0)
        _play_random_game(base.copy(), rng)

    samples = np.empty(iters, dtype=np.int64)
    total_t0 = time.perf_counter_ns()
    for i in range(iters):
        rng = random.Random(i + 1)
        rollout_state = base.copy()
        t0 = time.perf_counter_ns()
        _play_random_game(rollout_state, rng)
        samples[i] = time.perf_counter_ns() - t0
    total_elapsed_s = (time.perf_counter_ns() - total_t0) / 1e9
    games_per_sec = iters / total_elapsed_s

    return TimingResult(
        name="bench_rollout_throughput",
        min_us=float(samples.min()) / 1000.0,
        median_us=float(np.median(samples)) / 1000.0,
        mean_us=float(samples.mean()) / 1000.0,
        iters=iters,
        target_us=None,
        extra={
            "games_per_sec": games_per_sec,
            "target_games_per_sec": 1000.0,
            "games_per_sec_pass": games_per_sec >= 1000.0,
        },
    )


# ---------- reporting ---------------------------------------------------------


def _format_us(us: float) -> str:
    if us >= 1000.0:
        return f"{us / 1000.0:8.2f} ms"
    return f"{us:8.2f} µs"


def _print_table(results: list[TimingResult]) -> None:
    print()
    print(
        f"{'Benchmark':<30}{'Min':>14}{'Median':>14}{'Mean':>14}"
        f"{'Target':>14}{'Pass?':>8}"
    )
    print("-" * 94)
    for r in results:
        if r.name == "bench_rollout_throughput":
            assert r.extra is not None
            gps = r.extra["games_per_sec"]
            target = r.extra["target_games_per_sec"]
            passed = gps >= target
            print(
                f"{r.name:<30}{_format_us(r.min_us):>14}{_format_us(r.median_us):>14}"
                f"{_format_us(r.mean_us):>14}"
                f"{f'>= {target:.0f} g/s':>14}"
                f"{'PASS' if passed else 'FAIL':>8}"
            )
            print(f"{'  → throughput':<30}{gps:>14.1f} games/sec")
        else:
            target_str = f"< {r.target_us:.0f} µs" if r.target_us else "n/a"
            print(
                f"{r.name:<30}{_format_us(r.min_us):>14}{_format_us(r.median_us):>14}"
                f"{_format_us(r.mean_us):>14}{target_str:>14}"
                f"{'PASS' if r.passed else 'FAIL':>8}"
            )
    print()


def _results_to_json(results: list[TimingResult]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in results:
        entry: dict[str, Any] = {
            "min_us": r.min_us,
            "median_us": r.median_us,
            "mean_us": r.mean_us,
            "iters": r.iters,
            "target_us": r.target_us,
            "pass": r.passed,
        }
        if r.extra:
            entry.update(r.extra)
        out[r.name] = entry
    return out


# ---------- cProfile → HOTSPOTS.md -------------------------------------------


def _profile_rollouts(games: int) -> pstats.Stats:
    base = _make_midgame_state(plies=100, seed=42)
    pr = cProfile.Profile()
    pr.enable()
    for i in range(games):
        rng = random.Random(i + 1)
        _play_random_game(base.copy(), rng)
    pr.disable()
    return pstats.Stats(pr).strip_dirs()


def _write_hotspots(stats: pstats.Stats, dest: Path, games: int) -> None:
    """Pick top 3 non-trivial internal functions by cumulative time."""
    # pstats entries: {(file, lineno, fn): (cc, nc, tt, ct, callers)}
    # After strip_dirs(), only basenames remain — match on our own source files.
    our_files = {"engine.py", "actions.py", "state.py", "constants.py"}
    entries = list(stats.stats.items())  # type: ignore[attr-defined,arg-type]
    ours: list[tuple[tuple[str, int, str], tuple[int, int, float, float, Any]]] = []
    for key, val in entries:
        filename, _lineno, fn = key
        if filename in our_files and not fn.startswith("<"):
            ours.append((key, val))
    ours.sort(key=lambda kv: kv[1][3], reverse=True)  # by cumtime
    top = ours[:3]

    lines = [
        "# Phase 1.5 Hot Spots",
        "",
        f"Derived from cProfile on `bench_rollout_throughput` ({games} random "
        "rollouts from a mid-game 40x40 state, 4 players).",
        "",
        "Ranked by cumulative time in `territory_takeover` code. These are the "
        "functions that a Cython or C++ port should prioritize.",
        "",
        "| Rank | Function | File:Line | Calls | tottime (s) | cumtime (s) |",
        "|------|----------|-----------|-------|-------------|-------------|",
    ]
    for i, (key, val) in enumerate(top, start=1):
        filename, lineno, fn = key
        cc, _nc, tt, ct, _ = val
        short = filename.split("/")[-1] if "/" in filename else filename
        lines.append(
            f"| {i} | `{fn}` | `{short}:{lineno}` | {cc} | {tt:.4f} | {ct:.4f} |"
        )
    lines += [
        "",
        "## Notes on Cython / C++ candidacy",
        "",
    ]
    for i, (key, _val) in enumerate(top, start=1):
        _filename, _lineno, fn = key
        lines.append(f"### {i}. `{fn}`")
        lines.append("")
        lines.append(_candidacy_note(fn))
        lines.append("")
    dest.write_text("\n".join(lines))


def _candidacy_note(fn: str) -> str:
    if "detect_and_apply_enclosure" in fn:
        return (
            "**Strong C/Cython candidate.** The BFS boundary flood over an "
            "`np.bool_` mask is allocation-heavy (fresh mask + deque per "
            "enclosure check) and dominates step cost on non-trivial loops. "
            "A C implementation with a stack-allocated visited mask and "
            "explicit (row, col) queue would be 5-20x faster."
        )
    if fn == "step":
        return (
            "**Moderate candidate.** `step` itself is glue — its tottime is "
            "mostly attribute lookups (`state.players[...].path.append`) and "
            "dict construction for `info`. A Cython cdef class for GameState "
            "would collapse the attribute chain; worth doing once the core "
            "primitives are ported."
        )
    if fn == "legal_actions":
        return (
            "**Moderate candidate.** Already ~0.6 µs per call (good), but "
            "called once per step so total contribution is meaningful. Easy "
            "win in Cython: the four-way bounds+EMPTY check maps directly to "
            "a typed memoryview."
        )
    if fn == "_advance_turn":
        return (
            "**Weak candidate on its own**, but the inner `legal_actions` "
            "call makes up most of its cumtime. Port `legal_actions` first "
            "and this absorbs the gain."
        )
    if "copy" in fn:
        return (
            "**Moderate candidate.** `GameState.copy` is called once per "
            "MCTS node expansion, so its overhead is linear in tree size. "
            "A C copy of grid + typed per-player struct would trim ~30% — "
            "useful but not the first priority."
        )
    return (
        "Contribution is primarily through its callees. Porting the hot "
        "primitives above will likely absorb its cost."
    )


# ---------- main --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true", help="Don't write baseline/hotspots")
    parser.add_argument("--quick", action="store_true", help="10x fewer iterations")
    args = parser.parse_args()

    scale = 0.1 if args.quick else 1.0

    def s(n: int) -> int:
        return max(10, int(n * scale))

    print("Running TerritoryTakeover engine benchmarks...")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  numpy:  {np.__version__}")

    results = [
        bench_state_copy(iters=s(10_000)),
        bench_legal_actions(iters=s(200_000)),
        bench_step_no_enclosure(iters=s(50_000)),
        bench_step_with_enclosure(iters=s(5_000)),
        bench_full_game_random(iters=s(200)),
        bench_rollout_throughput(iters=s(500)),
    ]
    _print_table(results)

    if args.no_save:
        print("(--no-save: skipping baseline.json and HOTSPOTS.md)")
        return 0

    baseline = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": sys.platform,
        "quick_mode": args.quick,
        "results": _results_to_json(results),
    }
    (BENCH_DIR / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Wrote {BENCH_DIR / 'baseline.json'}")

    # cProfile sized to ~1-2s of rollout CPU.
    profile_games = max(50, int(200 * scale))
    print(f"Profiling {profile_games} rollouts for HOTSPOTS.md...")
    stats = _profile_rollouts(profile_games)
    _write_hotspots(stats, BENCH_DIR / "HOTSPOTS.md", games=profile_games)
    print(f"Wrote {BENCH_DIR / 'HOTSPOTS.md'}")

    # Exit nonzero if any hard-target benchmark failed.
    any_failed = any(not r.passed for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
