"""Render the baseline 20x20 head-to-head matrix as a heatmap PNG.

The source data is the committed table in
``docs/baseline_report_20x20.md``. Rather than re-run the 200-game
tournament, we transcribe the matrix directly from that committed
report (cell format ``wins/ties/losses`` from the row's perspective)
and plot it with a diverging colormap centered on 0.5 win rate.

Deterministic and does not run any games. Output is checked into
``docs/assets/h2h_heatmap.png`` and embedded in the project README.

Usage::

    python scripts/render_h2h_heatmap.py

Regenerate the underlying numbers by re-running the baseline report:
``scripts/run_baseline_report.py --board-size 20 --games-per-pair 20
--parallel --seed 0`` and pasting the matrix into the HEADTOHEAD
constant below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Row label -> dict(col_label -> "wins/ties/losses" from row's POV).
# Source: docs/baseline_report_20x20.md (commit e8bd847).
HEADTOHEAD: dict[str, dict[str, str]] = {
    "random": {
        "greedy": "9/0/11", "uct": "8/1/11",
        "rave": "1/1/18", "curriculum_ref": "6/5/9",
    },
    "greedy": {
        "random": "11/0/9", "uct": "1/0/19",
        "rave": "2/0/18", "curriculum_ref": "10/0/10",
    },
    "uct": {
        "random": "11/1/8", "greedy": "19/0/1",
        "rave": "7/0/13", "curriculum_ref": "14/0/6",
    },
    "rave": {
        "random": "18/1/1", "greedy": "18/0/2",
        "uct": "13/0/7", "curriculum_ref": "12/0/8",
    },
    "curriculum_ref": {
        "random": "9/5/6", "greedy": "10/0/10",
        "uct": "6/0/14", "rave": "8/0/12",
    },
}
# Row/column order is the leaderboard ordering (strongest first).
AGENTS: list[str] = ["rave", "uct", "curriculum_ref", "greedy", "random"]
GAMES_PER_PAIR: int = 20


def build_winrate_matrix(
    h2h: dict[str, dict[str, str]], agents: list[str], games_per_pair: int
) -> np.ndarray:
    """Parse ``wins/ties/losses`` cells into a win-rate matrix.

    Win rate counts ties against the row player (matches the convention
    used by the leaderboard's Wilson CIs). Diagonal cells are ``nan``.
    """
    n = len(agents)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, row_agent in enumerate(agents):
        for j, col_agent in enumerate(agents):
            if i == j:
                continue
            cell = h2h[row_agent][col_agent]
            wins_str, _ties_str, _losses_str = cell.split("/")
            wins = int(wins_str)
            # Tie-inclusive denominator == games_per_pair.
            mat[i, j] = wins / games_per_pair
    return mat


def render_heatmap(mat: np.ndarray, agents: list[str], out: Path) -> None:
    """Draw a square heatmap with annotated cells."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Diverging colormap centered on 0.5 so >0.5 is blue (row wins more),
    # <0.5 is red (row loses more). Mask the diagonal.
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    display = np.ma.masked_invalid(mat)
    im = ax.imshow(display, cmap="RdBu", norm=norm, aspect="equal")

    # Annotate each cell with the raw win rate.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "-", ha="center", va="center", color="gray", fontsize=14)
                continue
            val = mat[i, j]
            text_color = "white" if abs(val - 0.5) > 0.25 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=text_color, fontsize=11,
            )

    ax.set_xticks(range(len(agents)))
    ax.set_yticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_yticklabels(agents)
    ax.set_xlabel("opponent (column)")
    ax.set_ylabel("agent (row)")
    ax.set_title(
        f"Head-to-head win rate matrix - 20x20, 2p, {GAMES_PER_PAIR} games/pair\n"
        "(cells: row's wins / games_per_pair; ties count against the row)"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row win rate")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render the 20x20 head-to-head matrix from baseline_report_20x20.md "
            "as a heatmap PNG."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/h2h_heatmap.png"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mat = build_winrate_matrix(HEADTOHEAD, AGENTS, GAMES_PER_PAIR)
    print(f"[heatmap] win-rate matrix (rows={AGENTS}):")
    print(np.array2string(mat, precision=2))
    print(f"[heatmap] writing PNG to {args.out}")
    render_heatmap(mat, AGENTS, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
