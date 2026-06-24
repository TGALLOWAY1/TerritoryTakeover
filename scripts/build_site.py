"""Regenerate the static Vercel site under ``public/``.

The Vercel deployment is a *zero-build* static site: the self-contained replay
HTML files in ``public/games/`` are committed to the repo and served as-is. This
script reproduces them so the gallery can be regenerated or extended without
hand-running :mod:`scripts.record_html_demo` for each matchup.

Usage::

    python scripts/build_site.py            # regenerate every replay
    python scripts/build_site.py --quick    # lighter search (faster, lower quality)

The landing page (``public/index.html``) and ``vercel.json`` are maintained by
hand; this script only (re)writes ``public/games/*.html``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = REPO_ROOT / "public" / "games"
RECORD = REPO_ROOT / "scripts" / "record_html_demo.py"


@dataclass(frozen=True)
class Match:
    """One recorded matchup. Fields map directly to record_html_demo flags."""

    out: str
    title: str
    seats: tuple[str, ...]
    board_size: int
    seed: int
    rave_iterations: int


# ``rave_iterations`` is tuned to keep generation fast while still producing a
# competent game.
MATCHES: tuple[Match, ...] = (
    Match(
        out="rave-vs-greedy.html",
        title="Territory Takeover — RAVE vs Greedy",
        seats=("rave", "greedy"),
        board_size=20,
        seed=7,
        rave_iterations=200,
    ),
    Match(
        out="four-player-ffa.html",
        title="Territory Takeover — 4-Player Free-for-All",
        seats=("rave", "rave", "greedy", "random"),
        board_size=20,
        seed=3,
        rave_iterations=80,
    ),
    Match(
        out="greedy-vs-random.html",
        title="Territory Takeover — Greedy vs Random",
        seats=("greedy", "random"),
        board_size=16,
        seed=1,
        rave_iterations=200,
    ),
    Match(
        out="rave-mirror.html",
        title="Territory Takeover — RAVE Mirror (18x18)",
        seats=("rave", "rave"),
        board_size=18,
        seed=11,
        rave_iterations=60,
    ),
)


def _build_one(match: Match, quick: bool) -> None:
    rave_iters = 40 if quick else match.rave_iterations
    cmd = [
        sys.executable,
        str(RECORD),
        "--board-size",
        str(match.board_size),
        "--num-players",
        str(len(match.seats)),
        "--seed",
        str(match.seed),
        "--rave-iterations",
        str(rave_iters),
        "--title",
        match.title,
        "--out",
        str(GAMES_DIR / match.out),
    ]
    for i, key in enumerate(match.seats):
        cmd += [f"--seat{i}", key]
    print(f"[build-site] generating {match.out} ({' vs '.join(match.seats)})")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the static Vercel replay site.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lighter search settings for faster (lower quality) generation.",
    )
    args = parser.parse_args(argv)

    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    for match in MATCHES:
        _build_one(match, quick=args.quick)
    print(f"[build-site] done — {len(MATCHES)} replays in {GAMES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
