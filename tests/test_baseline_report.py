"""Tests for scripts/run_baseline_report.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_baseline_report as rbr  # type: ignore[import-not-found]  # noqa: E402


def test_build_roster_classical_only_has_four_distinct_agents() -> None:
    cfg = rbr.RosterConfig(seed=0, skip_curriculum=True)
    roster = rbr.build_roster(cfg)

    names = [a.name for a in roster]
    assert names == ["random", "greedy", "uct", "rave"]
    assert len(set(names)) == 4


def test_build_roster_is_deterministic_across_calls() -> None:
    cfg = rbr.RosterConfig(seed=42, skip_curriculum=True)
    r1 = rbr.build_roster(cfg)
    r2 = rbr.build_roster(cfg)
    assert [a.name for a in r1] == [a.name for a in r2]


def test_build_roster_propagates_budgets() -> None:
    cfg = rbr.RosterConfig(
        seed=0,
        uct_iterations=17,
        rave_iterations=23,
        skip_curriculum=True,
    )
    roster = rbr.build_roster(cfg)
    uct = roster[2]
    rave = roster[3]
    assert uct._iterations == 17
    assert rave._iterations == 23


def test_build_roster_missing_checkpoint_raises() -> None:
    cfg = rbr.RosterConfig(
        seed=0,
        skip_curriculum=False,
        checkpoint_path=Path("/nonexistent/does/not/exist.pt"),
    )
    with pytest.raises(FileNotFoundError, match="Curriculum checkpoint not found"):
        rbr.build_roster(cfg)


def test_dry_run_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = rbr.main(["--skip-curriculum", "--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Roster (4 agents)" in out
    for name in ("random", "greedy", "uct", "rave"):
        assert f"- {name}" in out


def test_non_dry_run_without_logic_returns_nonzero() -> None:
    """Until the tournament runner lands, main() without --dry-run must exit non-zero."""
    rc = rbr.main(["--skip-curriculum"])
    assert rc != 0
