# Changelog Notes

> Curated, human-readable history of notable changes, derived from the git log. Not a
> per-commit dump — see `git log` for that. Newest themes first. Last audited 2026-05-28.

## Interactive & live visualization (PRs #38–#40)
- Added an **interactive browser frontend** for human-vs-agent and spectator play
  (`viz_interactive.py`, `scripts/play_interactive.py`).
- Added a **live HTTP viewer** that streams gameplay to a browser (`viz_live.py`,
  `scripts/serve_live_demo.py`).
- Added an **interactive HTML replay viewer** with agent strategy, Elo, and live win
  probabilities (`viz_html.py`); bundled `best_agent_demo.html` with per-seat name overrides.

## Portfolio polish (multi-phase, PRs #35–#37)
- **Phase 1:** reframed the README around evidence/outcomes; added the agent gallery and CI badge.
- **Phase 2:** added measured performance numbers, spawn clamping, Gym render, architecture sketch.
- **Phase 3:** head-to-head heatmap, territory-growth plot, MCTS compute-scaling plot/script.
- Cleaned up project metadata in `pyproject.toml`; added a License section.

## Test stabilization
- Relaxed brittle statistical thresholds for small samples (UCT-vs-random; informed-rollout vs.
  uniform → 0.55) and pinned explicit spawns in a progressive-widening test to reduce flakiness.

## CI
- Configured GitHub Actions to run on PRs and pushes to `main` (Python 3.11/3.12; pytest required,
  ruff/mypy advisory). No benchmark CI by design.

## Documentation infrastructure (this pass, 2026-05-28)
- Added the structured `docs/00-08` documentation system (snapshot, inventories, architecture,
  quality, planning, AI-context, history, visuals).
- Archived loose narrative docs into `docs/06-history/archive/` (see
  [`ARCHIVE_NOTES.md`](archive/ARCHIVE_NOTES.md)).

## Foundational (earlier history)
- Core engine, state, actions, enclosure detection; classical search stack (Random/Greedy,
  Max-N/Paranoid, UCT, RAVE); RL stack (tabular Q, PPO primitives, AlphaZero, curriculum);
  tournament harness with Wilson-CI reports; 6 ADRs; optimization/benchmark writeups.

## See also
- Decisions: [`DECISION_LOG.md`](DECISION_LOG.md) · Audits: [`AUDIT_LOG.md`](AUDIT_LOG.md)
- Committed reports: `docs/baseline_report*.md`, `docs/curriculum_puct_scaling.md`.
