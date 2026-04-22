"""Tests for scripts/run_puct_scaling.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_puct_scaling as rps  # type: ignore[import-not-found]  # noqa: E402


def _fake_rows() -> list[rps.PanelRow]:
    """Synthetic sweep output that shows curriculum scaling with PUCT."""
    base = {
        "games": 20,
        "ties": 0,
        "avg_decision_time_s": 0.01,
    }
    rows: list[rps.PanelRow] = []
    for iters, (w_rand, w_greedy, w_uct) in zip(
        [4, 16, 64],
        [(10, 8, 2), (14, 12, 5), (18, 16, 10)],
        strict=True,
    ):
        for opp, w in zip(
            ["random", "greedy", "uct200"], [w_rand, w_greedy, w_uct], strict=True
        ):
            low = max(0.0, w / 20 - 0.2)
            high = min(1.0, w / 20 + 0.2)
            rows.append(
                rps.PanelRow(
                    az_iterations=iters,
                    opponent=opp,
                    wins=w,
                    losses=20 - w,
                    win_rate=w / 20,
                    ci_low=low,
                    ci_high=high,
                    **base,
                )
            )
    return rows


def _fake_result() -> rps.SweepResult:
    return rps.SweepResult(
        board_size=20,
        games_per_opponent=20,
        seed=0,
        uct_iterations=200,
        az_c_puct=1.25,
        rows=_fake_rows(),
    )


def test_group_rows_by_iters_splits_correctly() -> None:
    grouped = rps._group_rows_by_iters(_fake_rows())
    assert sorted(grouped) == [4, 16, 64]
    assert all(len(rs) == 3 for rs in grouped.values())


def test_format_report_md_headers_and_order() -> None:
    md = rps.format_report_md(
        _fake_result(), Path("docs/phase3d/net_reference.pt"), "x"
    )
    assert md.startswith("# Curriculum Reference — PUCT Scaling Sweep")
    for section in (
        "## Per-opponent breakdown",
        "## Aggregate win rate by PUCT budget",
        "## Reproducibility",
    ):
        assert section in md
    assert md.index("## Per-opponent breakdown") < md.index(
        "## Aggregate win rate by PUCT budget"
    ) < md.index("## Reproducibility")


def test_format_report_md_contains_each_iters_row() -> None:
    md = rps.format_report_md(_fake_result(), Path("x.pt"), "x")
    for iters in (4, 16, 64):
        # aggregate table row
        assert f"| {iters} | 60 |" in md  # 3 opponents * 20 games = 60


def test_format_report_md_reproducibility_footer() -> None:
    md = rps.format_report_md(
        _fake_result(),
        Path("docs/phase3d/net_reference.pt"),
        "python scripts/run_puct_scaling.py --board-size 20",
    )
    assert "- Board: 20x20, 2 players" in md
    assert "- Seed: 0" in md
    assert "- Games per (PUCT, opponent): 20" in md
    assert "- UCT reference iterations: 200" in md
    assert "`docs/phase3d/net_reference.pt`" in md
    assert "out-of-distribution" in md


def test_format_report_md_is_deterministic() -> None:
    r = _fake_result()
    p = Path("x.pt")
    assert rps.format_report_md(r, p, "cmd") == rps.format_report_md(r, p, "cmd")


def test_write_rows_csv(tmp_path: Path) -> None:
    result = _fake_result()
    path = tmp_path / "sweep.csv"
    rps.write_rows_csv(path, result)

    lines = path.read_text().splitlines()
    assert lines[0].startswith("az_iterations,opponent,games,wins")
    # 3 iters * 3 opponents = 9 data rows.
    assert len(lines) == 10
    # Spot-check first data row: iters=4, opponent=random, wins=10
    assert lines[1].startswith("4,random,20,10,")


def test_run_sweep_rejects_empty_iters() -> None:
    cfg = rps.SweepConfig(
        board_size=8,
        games_per_opponent=2,
        seed=0,
        az_iterations=(),
    )
    with pytest.raises(ValueError, match="non-empty"):
        rps.run_sweep(cfg)


def test_run_one_pairing_rejects_odd_games() -> None:
    """Contract guard: odd games_per_opponent breaks alternating-seat rotation."""
    cfg = rps.SweepConfig(board_size=8, games_per_opponent=3, seed=0)
    # Build a tiny pair of agents with no real work required.
    import numpy as np

    from territory_takeover.search.random_agent import UniformRandomAgent

    a = UniformRandomAgent(name="a", rng=np.random.default_rng(0))
    b = UniformRandomAgent(name="b", rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="must be even"):
        rps.run_one_pairing(a, b, cfg, seed=0)


def test_build_opponent_panel_names() -> None:
    cfg = rps.SweepConfig(board_size=8)
    panel = rps.build_opponent_panel(cfg, base_seed=0)
    assert [a.name for a in panel] == ["random", "greedy", "uct200"]
