# Technical Debt

> Code-level debt and implementation risk. Severity reflects likelihood × cost-to-change, not
> user impact (that's KNOWN_ISSUES). The codebase is clean overall; items below are modest.
> Last audited 2026-05-28.

| Item | Severity | Evidence | Suggested mitigation |
|---|---|---|---|
| AlphaZero gating `TODO` | Medium | `rl/alphazero/train.py:207-210` | Implement gating tournament (see KNOWN_ISSUES / NEXT_AGENT_TASKS). |
| `Any` in typed surfaces (41 occurrences) | Low | `gym_env.py` (obs dict/policy), `search/harness.py` (MP work items), `viz_live.py`/`viz_interactive.py` (HTTP payloads), `engine.py` (TYPE_CHECKING) | Tighten to `TypedDict`/protocols where cheap; acceptable for HTTP JSON payloads. |
| Large UI/HTTP modules | Low–Medium | `viz_interactive.py` (~1075), `viz_live.py` (~699), `viz_html.py` (~672) — embedded HTML/CSS/JS as Python strings | Consider extracting templates/static assets if these grow further. |
| Redundant state caches | Low (by design) | `state.py` (`path_set`/`claimed_count`/`empty_count` mirror the grid) | Already mitigated by `viz.check_invariants` + equivalence tests; keep mutation paths disciplined. |
| Duplicated RL obs encoders | Low (intentional) | `rl/ppo/spaces.py` vs `rl/alphazero/spaces.py` | Different by design (rotation vs. fixed seat); documented in STATE_AND_ENCODING — leave as-is. |
| Relaxed statistical test thresholds | Low | history: "relax UCT-vs-random threshold", "informed-rollout-beats-uniform 0.55" | Acceptable for small-sample stability; revisit if flakiness recurs. |
| Phase-named modules/configs | Low | `configs/phase3*`, docstrings reference "Phase 3a/3c" | Phase labels are historical; the DECISION_LOG/ADRs carry the durable rationale. |
| No PPO training driver | Low–Medium | `rl/ppo/` has no CLI | Tracked in KNOWN_ISSUES; add `scripts/train_ppo.py`. |

## Dead code / duplication
- No dead modules found; the only intentional "reference duplicate" is
  `engine._legacy_detect_and_apply_enclosure_full_bfs` (kept for equivalence tests).

## Notes
- `mypy --strict` and ruff (E,F,I,B,UP,N,SIM,RUF,ANN,TID) keep typing/lint debt low, though both
  are **advisory in CI** (see KNOWN_ISSUES) — so debt can creep without a hard gate.

## Related docs
- [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) · [`RISK_REGISTER.md`](RISK_REGISTER.md) · [`docs/05-planning/BACKLOG.md`](../05-planning/BACKLOG.md)
