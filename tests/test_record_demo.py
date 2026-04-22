"""Tests for scripts/record_demo.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import record_demo as rd  # type: ignore[import-not-found]  # noqa: E402


def test_build_roster_returns_two_named_agents() -> None:
    cfg = rd.DemoConfig(seed=0)
    roster = rd.build_roster(cfg)
    assert len(roster) == 2
    assert roster[0].name == "rave"
    assert roster[1].name == "curriculum_ref"


def test_build_roster_is_deterministic_across_calls() -> None:
    cfg = rd.DemoConfig(seed=42)
    r1 = rd.build_roster(cfg)
    r2 = rd.build_roster(cfg)
    assert [a.name for a in r1] == [a.name for a in r2]


def test_build_roster_propagates_budgets() -> None:
    cfg = rd.DemoConfig(seed=0, rave_iterations=5, az_iterations=7)
    roster = rd.build_roster(cfg)
    rave = roster[0]
    assert rave._iterations == 5


def test_build_roster_missing_checkpoint_raises() -> None:
    cfg = rd.DemoConfig(
        seed=0,
        checkpoint_path=Path("/nonexistent/does/not/exist.pt"),
    )
    with pytest.raises(FileNotFoundError, match="Curriculum checkpoint not found"):
        rd.build_roster(cfg)


def test_dry_run_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = rd.main(["--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Demo config" in out
    assert "- rave" in out
    assert "- curriculum_ref" in out


def test_non_dry_run_without_logic_returns_nonzero() -> None:
    """Until trajectory capture lands, main() without --dry-run must exit non-zero."""
    rc = rd.main([])
    assert rc != 0
