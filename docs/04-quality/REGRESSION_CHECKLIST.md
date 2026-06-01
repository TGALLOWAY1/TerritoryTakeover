# Regression Checklist

> Run before merging changes. The repo has no automated perf/visual regression gate, so these
> manual checks matter. Last audited 2026-05-28.

## Always (every change)
- [ ] `pytest` — full suite green (405 tests). For a targeted area: `pytest tests/test_<area>.py -v`.
- [ ] `ruff check .` — no new violations (advisory in CI; treat as required locally).
- [ ] `mypy` — strict, clean (advisory in CI; treat as required locally).
- [ ] `git diff` reviewed; no stray debug prints, no committed `results/` run artifacts.

## Engine / state / actions changes
- [ ] `tests/test_engine.py test_step.py test_enclosure.py test_state.py test_actions.py` pass.
- [ ] Enclosure equivalence holds: `tests/test_engine_equivalence.py` (optimized vs. legacy BFS).
- [ ] Caches stay in lockstep — spot-check with `viz.check_invariants(state)` on a played game.
- [ ] Perf spot-check (manual): `python benchmarks/bench_engine.py` — `copy()` < 50 µs,
      `legal_actions` < 1 µs, enclosure < 200 µs on 40×40 (no large regression vs. `benchmarks/baseline.json`).

## Search / MCTS changes
- [ ] `tests/test_mcts_*.py test_search_*.py` pass.
- [ ] Determinism preserved: a fixed-seed `run_match` gives identical results serial vs. parallel.
- [ ] Sanity: `python scripts/run_baseline_report.py --board-size 10 --games-per-pair 5 --seed 0 --dry-run`
      (or small live run) still ranks RAVE/UCT above Random/Greedy.

## RL changes
- [ ] Relevant `tests/test_rl_*` pass, including the `*_train_smoke.py` for the touched track.
- [ ] AlphaZero: if touching `train.py`, confirm the gating `TODO` status is unchanged or
      properly implemented (don't silently change promotion behavior).

## Visualization changes
- [ ] `tests/test_viz.py test_viz_html.py test_viz_live.py` pass.
- [ ] Regenerate a sample artifact and eyeball it (see
      [`docs/08-visuals/VISUAL_REGRESSION_PLAN.md`](../08-visuals/VISUAL_REGRESSION_PLAN.md)):
      e.g. `python scripts/record_demo.py --seed 0 --dry-run`.

## Reproducibility / reports
- [ ] If benchmark logic changed, regenerate committed reports and diff the tables
      (`docs/baseline_report*.md`); explain any movement in the commit message.

## Documentation
- [ ] If behavior/status changed, update the matching doc and its status label
      (FEATURE_INVENTORY / CURRENT_BEHAVIOR / KNOWN_ISSUES) — see
      [`docs/07-ai-context/AGENT_WORKFLOW.md`](../07-ai-context/AGENT_WORKFLOW.md).

## Related docs
- [`docs/03-implementation/TESTING_STRATEGY.md`](../03-implementation/TESTING_STRATEGY.md) · [`RISK_REGISTER.md`](RISK_REGISTER.md)
