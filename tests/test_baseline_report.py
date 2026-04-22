"""Tests for scripts/run_baseline_report.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from territory_takeover.search.harness import AgentStats

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_baseline_report as rbr  # type: ignore[import-not-found]  # noqa: E402


def _fake_result() -> rbr.BaselineResult:
    """Two-agent result with A winning 3 of 4 head-to-head games."""
    pair = rbr.PairAggregate(
        agent_a="alpha",
        agent_b="beta",
        games=4,
        wins_a=3,
        wins_b=1,
        ties=0,
        win_rate_a=0.75,
        ci_low=0.30,
        ci_high=0.95,
    )
    per_agent = [
        AgentStats(
            name="alpha",
            games=4,
            wins=3,
            ties=0,
            losses=1,
            avg_territory=20.0,
            avg_decision_time_s=0.012,
            avg_iters_per_s=float("nan"),
            n_decisions=50,
        ),
        AgentStats(
            name="beta",
            games=4,
            wins=1,
            ties=0,
            losses=3,
            avg_territory=12.0,
            avg_decision_time_s=0.030,
            avg_iters_per_s=float("nan"),
            n_decisions=50,
        ),
    ]
    return rbr.BaselineResult(
        board_size=10,
        num_players=2,
        games_per_pair=4,
        seed=0,
        pairs=[pair],
        per_agent=per_agent,
    )


def _fake_meta() -> rbr.ReportMetadata:
    return rbr.ReportMetadata(
        uct_iterations=200,
        rave_iterations=200,
        az_iterations=4,
        az_c_puct=1.25,
        skip_curriculum=True,
        checkpoint_path=Path("docs/phase3d/net_reference.pt"),
        commit_sha="deadbee",
        command="python scripts/run_baseline_report.py --dry-run",
    )


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


def test_run_all_pairs_two_random_agents(tmp_path: Path) -> None:
    """Two Random agents, 4 games: wins + ties + losses from each perspective must balance."""
    cfg = rbr.RosterConfig(seed=0, skip_curriculum=True)
    full = rbr.build_roster(cfg)
    roster = full[:2]  # random, greedy

    run_cfg = rbr.RunConfig(games_per_pair=4, board_size=8, seed=0)
    result = rbr.run_all_pairs(roster, run_cfg)

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.games == 4
    assert pair.wins_a + pair.wins_b + pair.ties == 4
    assert 0.0 <= pair.ci_low <= pair.win_rate_a <= pair.ci_high <= 1.0
    assert pair.agent_a == "random"
    assert pair.agent_b == "greedy"

    csv_path = tmp_path / "pairs.csv"
    rbr.write_pair_csv(csv_path, result)
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("agent_a,agent_b,games")
    assert lines[1].startswith("random,greedy,4,")


def test_run_all_pairs_is_deterministic_on_seed() -> None:
    cfg = rbr.RosterConfig(seed=0, skip_curriculum=True)
    roster_a = rbr.build_roster(cfg)[:2]
    roster_b = rbr.build_roster(cfg)[:2]

    run_cfg = rbr.RunConfig(games_per_pair=4, board_size=8, seed=7)
    ra = rbr.run_all_pairs(roster_a, run_cfg)
    rb = rbr.run_all_pairs(roster_b, run_cfg)
    assert ra.pairs[0].wins_a == rb.pairs[0].wins_a
    assert ra.pairs[0].wins_b == rb.pairs[0].wins_b
    assert ra.pairs[0].ties == rb.pairs[0].ties


def test_run_all_pairs_rejects_odd_games_per_pair() -> None:
    cfg = rbr.RosterConfig(seed=0, skip_curriculum=True)
    roster = rbr.build_roster(cfg)[:2]
    bad_cfg = rbr.RunConfig(games_per_pair=3, board_size=8)
    with pytest.raises(ValueError, match="multiple of 2"):
        rbr.run_all_pairs(roster, bad_cfg)


def test_run_all_pairs_rejects_singleton_roster() -> None:
    cfg = rbr.RosterConfig(seed=0, skip_curriculum=True)
    roster = rbr.build_roster(cfg)[:1]
    with pytest.raises(ValueError, match="at least 2"):
        rbr.run_all_pairs(roster, rbr.RunConfig(games_per_pair=2, board_size=8))


def test_format_report_md_headers_and_order() -> None:
    md = rbr.format_report_md(_fake_result(), _fake_meta())

    assert md.startswith("# Territory Takeover — Baseline Report")
    assert "## Leaderboard" in md
    assert "## Head-to-head" in md
    assert "## Reproducibility" in md

    head = md.index("## Leaderboard")
    tail = md.index("## Reproducibility")
    assert head < md.index("## Head-to-head") < tail


def test_format_report_md_ranks_by_win_rate() -> None:
    md = rbr.format_report_md(_fake_result(), _fake_meta())
    # Leaderboard rank 1 line must list alpha (0.75 win rate) before beta.
    rank1_line = next(
        line for line in md.splitlines() if line.startswith("| 1 |")
    )
    rank2_line = next(
        line for line in md.splitlines() if line.startswith("| 2 |")
    )
    assert "alpha" in rank1_line
    assert "beta" in rank2_line


def test_format_report_md_head_to_head_has_em_dash_diagonal() -> None:
    md = rbr.format_report_md(_fake_result(), _fake_meta())
    # Every agent's row must place an em dash on its own column.
    for line in md.splitlines():
        if line.startswith("| alpha |") and "beta" not in line.split("| alpha |")[0]:
            # alpha row: first data cell (alpha column) should be '—'
            cells = [c.strip() for c in line.split("|")[1:-1]]
            # cells = ["alpha", "—", "<w/t/l>"]
            assert cells[1] == "—"


def test_format_report_md_footer_metadata() -> None:
    md = rbr.format_report_md(_fake_result(), _fake_meta())
    assert "- Seed: 0" in md
    assert "- UCT iterations: 200" in md
    assert "- AlphaZero iterations: 4, c_puct: 1.25" in md
    assert "- Commit: `deadbee`" in md
    assert "(curriculum agent skipped)" in md


def test_format_report_md_shows_checkpoint_when_included() -> None:
    meta = rbr.ReportMetadata(
        uct_iterations=200,
        rave_iterations=200,
        az_iterations=4,
        az_c_puct=1.25,
        skip_curriculum=False,
        checkpoint_path=Path("docs/phase3d/net_reference.pt"),
        commit_sha="abc123",
        command="x",
    )
    md = rbr.format_report_md(_fake_result(), meta)
    assert "`docs/phase3d/net_reference.pt`" in md
    assert "(curriculum agent skipped)" not in md


def test_format_report_md_is_deterministic() -> None:
    r = _fake_result()
    m = _fake_meta()
    assert rbr.format_report_md(r, m) == rbr.format_report_md(r, m)


def test_main_writes_csv_on_small_run(tmp_path: Path) -> None:
    csv_out = tmp_path / "pairs.csv"
    rc = rbr.main(
        [
            "--skip-curriculum",
            "--games-per-pair",
            "2",
            "--board-size",
            "8",
            "--seed",
            "0",
            "--uct-iterations",
            "4",
            "--rave-iterations",
            "4",
            "--csv-out",
            str(csv_out),
            "--md-out",
            str(tmp_path / "report.md"),
        ]
    )
    assert rc == 0
    assert csv_out.exists()
    lines = csv_out.read_text().splitlines()
    # 4 classical agents => 6 unique pairs => 6 data rows + 1 header.
    assert len(lines) == 7

    md_out = tmp_path / "report.md"
    assert md_out.exists()
    md = md_out.read_text()
    assert "# Territory Takeover — Baseline Report" in md
    assert "## Leaderboard" in md
    assert "## Head-to-head" in md
