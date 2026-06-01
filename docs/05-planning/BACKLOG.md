# Backlog

> Scored, actionable work items. Synthesized from [`docs/04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md)
> and the (now-archived) portfolio-readiness roadmap — with already-completed items removed.
> Last audited 2026-05-28.

## Scoring model
`Priority = user_impact + technical_risk_reduction + recruiter/demo_value − implementation_complexity`
(each sub-score 1–5; higher Priority = do sooner).

| ID | Task | Impact | Risk↓ | Demo | Complexity | **Priority** |
|---|---|---:|---:|---:|---:|---:|
| B1 | Implement AlphaZero snapshot **gating** tournament | 4 | 4 | 3 | 3 | **8** |
| B2 | Add a PPO **training driver** + smoke test + one result | 3 | 2 | 3 | 3 | **5** |
| B3 | Lightweight **perf regression** check (or scheduled benchmark CI) | 3 | 4 | 1 | 2 | **6** |
| B4 | Curriculum **value-head / 20×20 scaling** improvement | 4 | 3 | 3 | 5 | **5** |
| B5 | Run **metadata traceability** (config + git SHA + seed hash) in reports | 2 | 3 | 2 | 1 | **6** |
| B6 | Promote ruff/mypy from advisory → **required** in CI | 2 | 3 | 1 | 1 | **5** |
| B7 | Tighten `Any` usages to `TypedDict`/protocols where cheap | 1 | 2 | 1 | 2 | **2** |

---

## B1 — Implement AlphaZero gating tournament
- **Priority:** High · **Category:** RL correctness
- **User impact:** Prevents training regressions; makes AlphaZero results trustworthy.
- **Technical impact:** Closes the only deliberate stub in `src/`.
- **Why now:** It's the single most-cited limitation (README, ADR-005, KNOWN_ISSUES).
- **Relevant files:** `rl/alphazero/train.py:207-210`, `search/harness.py`, `rl/alphazero/evaluator.py`.
- **Dependencies:** seed-locked harness (exists).
- **Acceptance criteria:** new snapshot is promoted only after beating the current champion over
  N seed-locked games at a configurable win-rate threshold; behavior covered by a test.
- **Implementation sketch:** after self-play+train, play champion-vs-candidate via the harness;
  gate on threshold; keep champion if it fails.
- **Verification:** `pytest tests/test_rl_alphazero_*`; small `train_alphazero.py` run shows gating logs.

## B2 — PPO training driver
- **Priority:** Medium · **Category:** RL completeness
- **Relevant files:** new `scripts/train_ppo.py`, `rl/ppo/*`.
- **Acceptance criteria:** end-to-end PPO training runs; PPO appears as a tournament participant;
  smoke test added.
- **Verification:** `pytest tests/test_rl_ppo_*` + new smoke test.

## B3 — Performance regression guard
- **Priority:** Medium-High · **Category:** Engine reliability
- **Relevant files:** `benchmarks/bench_engine.py`, `.github/workflows/ci.yml`, `CLAUDE.md` targets.
- **Acceptance criteria:** a perf smoke test (generous thresholds) or scheduled workflow flags
  large regressions in `copy`/`step`/enclosure.
- **Verification:** intentionally slow a hot path → check fails.

## B4 — Curriculum value-head / 20×20 scaling
- **Priority:** Medium · **Category:** RL research
- **Relevant files:** `rl/curriculum/`, `rl/alphazero/network.py`, `docs/experiments/20x20_hypothesis_test.md`.
- **Acceptance criteria:** win rate improves monotonically with eval-time PUCT at 20×20, or a
  documented negative result with analysis.

## B5 — Run metadata traceability
- **Priority:** Medium · **Category:** Reproducibility
- **Relevant files:** `search/harness.py`, report writers in `scripts/run_baseline_report.py`.
- **Acceptance criteria:** each report embeds config hash + git SHA + root seed.

## B6 — Make lint/type checks required in CI
- **Priority:** Medium · **Category:** Quality gate
- **Relevant files:** `.github/workflows/ci.yml`.
- **Acceptance criteria:** ruff + mypy drop `continue-on-error` once clean.

## B7 — Reduce `Any` usage
- **Priority:** Low · **Category:** Typing debt
- **Relevant files:** `gym_env.py`, `search/harness.py` (HTTP payloads acceptable to leave).

## Already completed (do not re-do)
README rewrite around evidence; canonical baseline reports (10×10 + 20×20) with Wilson CIs;
demo GIF + agent gallery + h2h heatmap + territory-growth + MCTS-scaling plots; reproducibility
commands; ADRs (engineering-decisions note); interactive + live replay viewers; architecture
diagram (README + `docs/02-architecture/SYSTEM_MAP.md`).

## Related docs
- [`PRIORITIZED_TODO.md`](PRIORITIZED_TODO.md) · [`ROADMAP.md`](ROADMAP.md) · [`NEXT_AGENT_TASKS.md`](NEXT_AGENT_TASKS.md)
