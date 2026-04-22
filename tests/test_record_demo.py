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


def _tiny_config(tmp_path: Path) -> rd.DemoConfig:
    """Small-board, cheap-agent config so tests run fast without torch."""
    return rd.DemoConfig(
        board_size=6,
        num_players=2,
        seed=0,
        rave_iterations=4,
        az_iterations=2,
        checkpoint_path=tmp_path / "nonexistent.pt",
        frame_stride=1,
        fps=4,
    )


def test_play_demo_game_produces_a_terminal_trajectory(tmp_path: Path) -> None:
    import numpy as np

    from territory_takeover.search.mcts.rave import RaveAgent
    from territory_takeover.search.random_agent import UniformRandomAgent

    cfg = _tiny_config(tmp_path)
    roster = [
        RaveAgent(iterations=4, name="rave", rng=np.random.default_rng(0)),
        UniformRandomAgent(name="random", rng=np.random.default_rng(1)),
    ]
    trajectory = rd.play_demo_game(roster, cfg)

    assert len(trajectory) >= 2  # initial + at least one step
    assert trajectory[0].turn_number == 0
    assert trajectory[-1].done
    # Every non-initial snapshot is a distinct copy (no aliasing).
    assert all(
        trajectory[i] is not trajectory[i - 1] for i in range(1, len(trajectory))
    )


def test_play_demo_game_is_deterministic_on_seed(tmp_path: Path) -> None:
    import numpy as np

    from territory_takeover.search.mcts.rave import RaveAgent
    from territory_takeover.search.random_agent import UniformRandomAgent

    cfg = _tiny_config(tmp_path)

    def _build_roster() -> list:  # type: ignore[type-arg]
        return [
            RaveAgent(iterations=4, name="rave", rng=np.random.default_rng(0)),
            UniformRandomAgent(name="random", rng=np.random.default_rng(1)),
        ]

    t1 = rd.play_demo_game(_build_roster(), cfg)
    t2 = rd.play_demo_game(_build_roster(), cfg)
    assert len(t1) == len(t2)
    assert t1[-1].winner == t2[-1].winner


def test_play_demo_game_rejects_wrong_roster_size(tmp_path: Path) -> None:
    import numpy as np

    from territory_takeover.search.random_agent import UniformRandomAgent

    cfg = _tiny_config(tmp_path)
    roster = [UniformRandomAgent(name="only", rng=np.random.default_rng(0))]
    with pytest.raises(ValueError, match="roster length"):
        rd.play_demo_game(roster, cfg)


def test_downsample_keeps_first_and_terminal() -> None:
    dummy = list(range(10))
    kept = rd._downsample(dummy, 3)
    assert kept[0] == 0
    assert kept[-1] == 9  # terminal preserved even though 9 % 3 != 0


def test_downsample_stride_one_is_identity() -> None:
    dummy = list(range(5))
    kept = rd._downsample(dummy, 1)
    assert kept == dummy
    assert kept is not dummy  # must return a copy, not the input list


def test_downsample_stride_two_keeps_half_plus_terminal() -> None:
    dummy = list(range(6))
    kept = rd._downsample(dummy, 2)
    # Every 2nd frame: [0, 2, 4], plus terminal 5 appended.
    assert kept == [0, 2, 4, 5]


def test_downsample_rejects_invalid_stride() -> None:
    dummy = [1, 2, 3]
    with pytest.raises(ValueError, match="stride"):
        rd._downsample(dummy, 0)
