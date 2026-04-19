"""CLI entry point for running tournaments from a YAML config.

Usage:
    python scripts/run_tournament.py --config configs/phase2_tournament.yaml

Writes CSV files to ``results/<config_stem>_<timestamp>/`` (or to the
path given by ``--output``). A copy of the resolved config is dropped
alongside the CSVs for provenance.

Install with ``pip install -e ".[tournament]"`` to pull in ``pyyaml``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from territory_takeover.search.harness import (
    AgentStats,
    GameLog,
    MatchResult,
    PairRow,
    Table,
    round_robin,
    run_match,
)
from territory_takeover.search.registry import AgentSpec

if TYPE_CHECKING:
    from territory_takeover.search.agent import Agent


def _require_yaml() -> ModuleType:
    """Import PyYAML with a helpful error if the extra is not installed."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise SystemExit(
            "pyyaml is required for scripts/run_tournament.py. "
            'Install with: pip install -e ".[tournament]"'
        ) from exc
    return yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a TerritoryTakeover tournament from YAML.")
    p.add_argument("--config", type=Path, required=True, help="YAML config path.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/<config_stem>_<timestamp>).",
    )
    p.add_argument(
        "--mode",
        choices=["match", "round_robin"],
        default=None,
        help="Override config.mode.",
    )
    parallel = p.add_mutually_exclusive_group()
    parallel.add_argument("--parallel", action="store_true", help="Force parallel execution.")
    parallel.add_argument(
        "--no-parallel", action="store_true", help="Force serial execution."
    )
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    yaml = _require_yaml()
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping; got {type(data).__name__}")
    return data


def _build_agents(specs: list[AgentSpec]) -> list[Agent]:
    return [spec.build() for spec in specs]


def _specs_from_config(cfg: dict[str, Any]) -> list[AgentSpec]:
    raw = cfg.get("agents")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.agents must be a non-empty list")
    specs: list[AgentSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"Each agents entry must be a mapping; got {entry!r}")
        name = str(entry["name"])
        class_name = str(entry["class"])
        kwargs = dict(entry.get("kwargs") or {})
        specs.append(AgentSpec(name=name, class_name=class_name, kwargs=kwargs))
    return specs


def _format_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    if not rows:
        return ""
    widths = [len(h) for h in headers]
    rendered: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for i, h in enumerate(headers):
            v = row.get(h, "")
            cell = (
                ("nan" if math.isnan(v) else f"{v:.3f}")
                if isinstance(v, float)
                else str(v)
            )
            cells.append(cell)
            widths[i] = max(widths[i], len(cell))
        rendered.append(cells)
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join(
        "  ".join(cells[i].ljust(widths[i]) for i in range(len(headers)))
        for cells in rendered
    )
    return "\n".join([header_line, sep_line, body])


def _agent_stats_row(s: AgentStats) -> dict[str, Any]:
    return asdict(s)


def _pair_row(r: PairRow) -> dict[str, Any]:
    return asdict(r)


def _game_row(g: GameLog) -> dict[str, Any]:
    row = asdict(g)
    # Flatten nested lists to JSON strings so they live in one CSV cell each.
    row["seat_assignment"] = json.dumps(row["seat_assignment"])
    row["actions"] = json.dumps(row["actions"])
    row["final_scores"] = json.dumps(row["final_scores"])
    row["decision_times_s"] = json.dumps(row["decision_times_s"])
    row["iterations"] = json.dumps(row["iterations"])
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _default_output_dir(config_path: Path) -> Path:
    stem = config_path.stem
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / f"{stem}_{ts}"


def _dispatch_match(cfg: dict[str, Any], specs: list[AgentSpec], parallel: bool) -> MatchResult:
    agents = _build_agents(specs)
    return run_match(
        agents=agents,
        num_games=int(cfg.get("num_games", 8)),
        board_size=int(cfg.get("board_size", 20)),
        swap_seats=bool(cfg.get("swap_seats", True)),
        seed=int(cfg.get("seed", 0)),
        parallel=parallel,
        num_players=int(cfg["num_players"]) if "num_players" in cfg else None,
    )


def _dispatch_round_robin(
    cfg: dict[str, Any], specs: list[AgentSpec], parallel: bool
) -> Table:
    agents = _build_agents(specs)
    return round_robin(
        agents=agents,
        games_per_pair=int(cfg.get("games_per_pair", 8)),
        board_size=int(cfg.get("board_size", 20)),
        seed=int(cfg.get("seed", 0)),
        parallel=parallel,
        num_players=int(cfg.get("num_players", 4)),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = _load_config(args.config)
    specs = _specs_from_config(cfg)

    mode = args.mode or str(cfg.get("mode", "match"))
    if mode not in ("match", "round_robin"):
        raise SystemExit(f"Unknown mode {mode!r}; expected 'match' or 'round_robin'.")

    if args.parallel:
        parallel = True
    elif args.no_parallel:
        parallel = False
    else:
        parallel = bool(cfg.get("parallel", False))

    out_dir = args.output or _default_output_dir(args.config)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop a copy of the config for provenance.
    shutil.copy2(args.config, out_dir / args.config.name)

    if mode == "match":
        result = _dispatch_match(cfg, specs, parallel)
        agent_rows = [_agent_stats_row(s) for s in result.per_agent]
        game_rows = [_game_row(g) for g in result.games]
        _write_csv(out_dir / "match_summary.csv", agent_rows)
        _write_csv(out_dir / "match_games.csv", game_rows)
        print(f"Match: {result.num_games} games on {result.board_size}x{result.board_size}")
        print(
            _format_table(
                agent_rows,
                [
                    "name",
                    "games",
                    "wins",
                    "ties",
                    "losses",
                    "avg_territory",
                    "avg_decision_time_s",
                    "avg_iters_per_s",
                ],
            )
        )
    else:
        table = _dispatch_round_robin(cfg, specs, parallel)
        pair_rows = [_pair_row(r) for r in table.rows]
        agent_rows = [_agent_stats_row(s) for s in table.per_agent]
        game_rows = [_game_row(g) for g in table.games]
        _write_csv(out_dir / "roundrobin_pairs.csv", pair_rows)
        _write_csv(out_dir / "roundrobin_agents.csv", agent_rows)
        _write_csv(out_dir / "roundrobin_games.csv", game_rows)
        print(f"Round-robin across {len(specs)} agents")
        print(
            _format_table(
                pair_rows,
                [
                    "agent_a",
                    "agent_b",
                    "games",
                    "wins_a",
                    "wins_b",
                    "ties",
                    "win_rate_a",
                    "ci_low",
                    "ci_high",
                ],
            )
        )
        print()
        print(
            _format_table(
                agent_rows,
                [
                    "name",
                    "games",
                    "wins",
                    "ties",
                    "losses",
                    "avg_territory",
                    "avg_decision_time_s",
                    "avg_iters_per_s",
                ],
            )
        )

    print(f"\nResults written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
