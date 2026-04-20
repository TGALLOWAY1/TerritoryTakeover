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

from territory_takeover import GameState, new_game, simulate_random_rollout, step
from territory_takeover.actions import legal_actions
from territory_takeover.constants import DIRECTIONS, PATH_CODES

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
    p50_us: float | None = None
    p90_us: float | None = None
    p99_us: float | None = None

    @property
    def passed(self) -> bool:
        if self.target_us is None:
            return True
        return self.min_us < self.target_us


def _percentiles_us(samples: np.ndarray) -> tuple[float, float, float]:
    """Return (p50, p90, p99) in µs from an int64 nanosecond sample array."""
    p50, p90, p99 = np.percentile(samples, [50, 90, 99])
    return float(p50) / 1000.0, float(p90) / 1000.0, float(p99) / 1000.0


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
    p50, p90, p99 = _percentiles_us(samples)
    return TimingResult(
        name=name,
        min_us=min_us,
        median_us=median_us,
        mean_us=mean_us,
        iters=iters,
        target_us=target_us,
        p50_us=p50,
        p90_us=p90,
        p99_us=p99,
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


def bench_full_game_random(iters: int, board_size: int = 40) -> TimingResult:
    """Full random game — report per-game milliseconds and turn stats.

    `board_size` is parametrized so the harness can scale across 20/30/40.
    The per-call target scales with area (O(cells) worst-case BFS).
    """
    target_us = float(50_000 * (board_size / 40) ** 2)

    # warmup
    for _ in range(3):
        rng = random.Random(0)
        _play_random_game(new_game(board_size=board_size, num_players=4, seed=0), rng)

    samples = np.empty(iters, dtype=np.int64)
    turns_arr = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        rng = random.Random(i)
        state = new_game(board_size=board_size, num_players=4, seed=i)
        t0 = time.perf_counter_ns()
        turns = _play_random_game(state, rng)
        samples[i] = time.perf_counter_ns() - t0
        turns_arr[i] = turns

    p50, p90, p99 = _percentiles_us(samples)
    mean_turns = float(turns_arr.mean())
    total_elapsed_s = float(samples.sum()) / 1e9
    total_turns = int(turns_arr.sum())
    turns_per_sec = total_turns / total_elapsed_s if total_elapsed_s > 0 else 0.0

    return TimingResult(
        name=f"bench_full_game_random_{board_size}",
        min_us=float(samples.min()) / 1000.0,
        median_us=float(np.median(samples)) / 1000.0,
        mean_us=float(samples.mean()) / 1000.0,
        iters=iters,
        target_us=target_us,
        p50_us=p50,
        p90_us=p90,
        p99_us=p99,
        extra={
            "board_size": float(board_size),
            "mean_turns_per_game": mean_turns,
            "turns_per_sec": turns_per_sec,
            "games_per_min": 60.0 * iters / total_elapsed_s if total_elapsed_s > 0 else 0.0,
        },
    )


def bench_rollout_throughput(iters: int, board_size: int = 40) -> TimingResult:
    """Rollouts (copy mid-game state, play to completion) — games per second.

    This is the MCTS-relevant op. We report games/sec as the primary metric
    in ``extra``; ``min/median/mean_us`` are per-rollout times for context.
    Scaled across board sizes via `board_size`.
    """
    # Scale plies with board area so mid-game states are comparable across sizes.
    plies = int(100 * (board_size / 40) ** 2)
    base = _make_midgame_state(board_size=board_size, plies=plies, seed=42)

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

    p50, p90, p99 = _percentiles_us(samples)

    return TimingResult(
        name=f"bench_rollout_throughput_{board_size}",
        min_us=float(samples.min()) / 1000.0,
        median_us=float(np.median(samples)) / 1000.0,
        mean_us=float(samples.mean()) / 1000.0,
        iters=iters,
        target_us=None,
        p50_us=p50,
        p90_us=p90,
        p99_us=p99,
        extra={
            "board_size": float(board_size),
            "games_per_sec": games_per_sec,
            "games_per_min": games_per_sec * 60.0,
            "target_games_per_sec": 1000.0,
            "games_per_sec_pass": float(games_per_sec >= 1000.0),
        },
    )


def bench_rollout_fastpath_throughput(
    iters: int, board_size: int = 40
) -> TimingResult:
    """Same as `bench_rollout_throughput` but uses the `simulate_random_rollout`
    fast path instead of a `step()` loop.

    `simulate_random_rollout` skips the `StepResult`/`info`/numpy mask
    allocations per half-move and inlines the legality + coord + enclosure
    path into one tight Python loop. This bench isolates that gain from the
    `step()`-based rollout cost measured by `bench_rollout_throughput`.
    """
    plies = int(100 * (board_size / 40) ** 2)
    base = _make_midgame_state(board_size=board_size, plies=plies, seed=42)

    for _ in range(5):
        rng = random.Random(0)
        simulate_random_rollout(base.copy(), rng)

    samples = np.empty(iters, dtype=np.int64)
    total_t0 = time.perf_counter_ns()
    for i in range(iters):
        rng = random.Random(i + 1)
        rollout_state = base.copy()
        t0 = time.perf_counter_ns()
        simulate_random_rollout(rollout_state, rng)
        samples[i] = time.perf_counter_ns() - t0
    total_elapsed_s = (time.perf_counter_ns() - total_t0) / 1e9
    games_per_sec = iters / total_elapsed_s

    p50, p90, p99 = _percentiles_us(samples)

    return TimingResult(
        name=f"bench_rollout_fastpath_throughput_{board_size}",
        min_us=float(samples.min()) / 1000.0,
        median_us=float(np.median(samples)) / 1000.0,
        mean_us=float(samples.mean()) / 1000.0,
        iters=iters,
        target_us=None,
        p50_us=p50,
        p90_us=p90,
        p99_us=p99,
        extra={
            "board_size": float(board_size),
            "games_per_sec": games_per_sec,
            "games_per_min": games_per_sec * 60.0,
            "target_games_per_sec": 1000.0,
            "games_per_sec_pass": float(games_per_sec >= 1000.0),
        },
    )


def bench_trigger_fire_rate(
    games: int = 50,
    board_size: int = 40,
    num_players: int = 4,
    seed_base: int = 1000,
) -> TimingResult:
    """Diagnostic: measure how often the enclosure trigger fires vs actually claims.

    The current engine runs a full-board boundary BFS every time the cheap
    trigger predicate fires (placed_cell adjacent to a same-player path tile
    other than its predecessor). On random play the trigger over-approximates
    — many fires produce 0 claimed cells — which is the core wasted-work
    hypothesis the optimization plan addresses.

    Reported via ``extra``:
      - trigger_fires: total fires summed across games
      - enclosures:    fires that claimed >= 1 cell
      - wasted_fires:  fires that claimed 0 cells (full BFS for nothing)
      - wasted_rate:   wasted_fires / trigger_fires
      - steps:         total step() calls
    """
    trigger_fires = 0
    enclosures = 0
    steps_total = 0
    wall_ns = 0

    for g in range(games):
        state = new_game(board_size=board_size, num_players=num_players, seed=seed_base + g)
        rng = random.Random(seed_base + g)
        t0 = time.perf_counter_ns()
        while not state.done:
            pid = state.current_player
            acts = legal_actions(state, pid)
            if not acts:
                step(state, 0)  # advances/marks dead
                steps_total += 1
                continue

            # Replicate the trigger-fire check around the step call so we can
            # attribute before/after. This is diagnostic only; overhead is fine.
            r, c = state.players[pid].head
            chosen = rng.choice(acts)
            dr, dc = DIRECTIONS[chosen]
            target = (r + dr, c + dc)
            # Would-be predecessor is the current head, not the placed target.
            predecessor = (r, c)
            fires = False
            p = state.players[pid]
            pset = p.path_set
            tr, tc = target
            for ddr, ddc in DIRECTIONS:
                nbr = (tr + ddr, tc + ddc)
                if nbr == predecessor:
                    continue
                if nbr in pset:
                    fires = True
                    break

            result = step(state, chosen)
            steps_total += 1
            if fires:
                trigger_fires += 1
                if result.info["claimed_this_turn"] > 0:
                    enclosures += 1
        wall_ns += time.perf_counter_ns() - t0

    wasted = trigger_fires - enclosures
    wasted_rate = (wasted / trigger_fires) if trigger_fires else 0.0
    elapsed_s = wall_ns / 1e9

    return TimingResult(
        name=f"bench_trigger_fire_rate_{board_size}",
        min_us=0.0,
        median_us=0.0,
        mean_us=0.0,
        iters=games,
        target_us=None,
        extra={
            "board_size": float(board_size),
            "num_players": float(num_players),
            "steps": float(steps_total),
            "trigger_fires": float(trigger_fires),
            "enclosures": float(enclosures),
            "wasted_fires": float(wasted),
            "wasted_rate": float(wasted_rate),
            "wall_s": elapsed_s,
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
        f"{'Benchmark':<34}{'Min':>12}{'p50':>12}{'p90':>12}{'p99':>12}"
        f"{'Mean':>12}{'Target':>14}{'Pass?':>8}"
    )
    print("-" * 116)
    for r in results:
        if r.name.startswith("bench_trigger_fire_rate"):
            assert r.extra is not None
            print(
                f"{r.name:<34}{'—':>12}{'—':>12}{'—':>12}{'—':>12}{'—':>12}"
                f"{'n/a':>14}{'INFO':>8}"
            )
            steps = int(r.extra["steps"])
            fires = int(r.extra["trigger_fires"])
            encs = int(r.extra["enclosures"])
            wasted = int(r.extra["wasted_fires"])
            wasted_rate = r.extra["wasted_rate"]
            print(
                f"{'  → steps/fires/enclosures':<34}"
                f"{steps:>12}{fires:>12}{encs:>12}{wasted:>12}"
                f"{f'{wasted_rate * 100:.1f}%':>12}"
                f"{'wasted':>14}"
            )
            continue
        if r.name.startswith(
            ("bench_rollout_throughput", "bench_rollout_fastpath_throughput")
        ):
            assert r.extra is not None
            gps = r.extra["games_per_sec"]
            target = r.extra["target_games_per_sec"]
            passed = gps >= target
            p50 = r.p50_us if r.p50_us is not None else r.median_us
            p90 = r.p90_us if r.p90_us is not None else r.median_us
            p99 = r.p99_us if r.p99_us is not None else r.median_us
            print(
                f"{r.name:<34}"
                f"{_format_us(r.min_us):>12}"
                f"{_format_us(p50):>12}"
                f"{_format_us(p90):>12}"
                f"{_format_us(p99):>12}"
                f"{_format_us(r.mean_us):>12}"
                f"{f'>= {target:.0f} g/s':>14}"
                f"{'PASS' if passed else 'FAIL':>8}"
            )
            print(f"{'  → throughput':<34}{gps:>12.1f} g/s  ({gps * 60:.0f} g/min)")
            continue

        target_str = f"< {r.target_us:.0f} µs" if r.target_us else "n/a"
        p50 = r.p50_us if r.p50_us is not None else r.median_us
        p90 = r.p90_us if r.p90_us is not None else r.median_us
        p99 = r.p99_us if r.p99_us is not None else r.median_us
        print(
            f"{r.name:<34}"
            f"{_format_us(r.min_us):>12}"
            f"{_format_us(p50):>12}"
            f"{_format_us(p90):>12}"
            f"{_format_us(p99):>12}"
            f"{_format_us(r.mean_us):>12}"
            f"{target_str:>14}"
            f"{'PASS' if r.passed else 'FAIL':>8}"
        )
        if r.extra and "mean_turns_per_game" in r.extra:
            mt = r.extra["mean_turns_per_game"]
            tps = r.extra["turns_per_sec"]
            gpm = r.extra["games_per_min"]
            print(
                f"{'  → turns/game / turns-per-sec / games-per-min':<34}"
                f"{mt:>12.1f}{tps:>12.0f}{gpm:>12.0f}"
            )
    print()


def _results_to_json(results: list[TimingResult]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in results:
        entry: dict[str, Any] = {
            "min_us": r.min_us,
            "median_us": r.median_us,
            "mean_us": r.mean_us,
            "p50_us": r.p50_us,
            "p90_us": r.p90_us,
            "p99_us": r.p99_us,
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
    parser.add_argument(
        "--board-sizes",
        type=str,
        default="20,30,40",
        help="Comma-separated board sizes for scaling benches (default 20,30,40).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="baseline.json",
        help="Output JSON filename under benchmarks/ (default baseline.json).",
    )
    parser.add_argument(
        "--hotspots-out",
        type=str,
        default="HOTSPOTS.md",
        help="Output markdown filename under benchmarks/ (default HOTSPOTS.md).",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip the cProfile pass that regenerates HOTSPOTS.md.",
    )
    args = parser.parse_args()

    scale = 0.1 if args.quick else 1.0
    board_sizes = [int(x) for x in args.board_sizes.split(",") if x.strip()]

    def s(n: int) -> int:
        return max(10, int(n * scale))

    print("Running TerritoryTakeover engine benchmarks...")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  numpy:  {np.__version__}")
    print(f"  board_sizes: {board_sizes}")

    results: list[TimingResult] = [
        bench_state_copy(iters=s(10_000)),
        bench_legal_actions(iters=s(200_000)),
        bench_step_no_enclosure(iters=s(50_000)),
        bench_step_with_enclosure(iters=s(5_000)),
    ]
    # Scale the per-game game/rollout iter counts with area so wall-clock stays
    # roughly balanced across board sizes.
    for bs in board_sizes:
        area_factor = (bs / 40) ** 2  # 20→0.25, 30→0.56, 40→1.0
        game_iters = max(30, int(s(200) / max(area_factor, 0.25)))
        rollout_iters = max(50, int(s(500) / max(area_factor, 0.25)))
        results.append(bench_full_game_random(iters=game_iters, board_size=bs))
        results.append(bench_rollout_throughput(iters=rollout_iters, board_size=bs))
        results.append(
            bench_rollout_fastpath_throughput(
                iters=rollout_iters, board_size=bs
            )
        )
        results.append(bench_trigger_fire_rate(games=max(10, s(50)), board_size=bs))

    _print_table(results)

    if args.no_save:
        print("(--no-save: skipping JSON and HOTSPOTS.md)")
        return 0

    baseline = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": sys.platform,
        "quick_mode": args.quick,
        "board_sizes": board_sizes,
        "results": _results_to_json(results),
    }
    (BENCH_DIR / args.out).write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Wrote {BENCH_DIR / args.out}")

    if args.skip_profile:
        print("(--skip-profile: skipping HOTSPOTS.md)")
    else:
        # cProfile sized to ~1-2s of rollout CPU.
        profile_games = max(50, int(200 * scale))
        print(f"Profiling {profile_games} rollouts for {args.hotspots_out}...")
        stats = _profile_rollouts(profile_games)
        _write_hotspots(stats, BENCH_DIR / args.hotspots_out, games=profile_games)
        print(f"Wrote {BENCH_DIR / args.hotspots_out}")

    # Exit nonzero if any hard-target benchmark failed.
    any_failed = any(not r.passed for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
