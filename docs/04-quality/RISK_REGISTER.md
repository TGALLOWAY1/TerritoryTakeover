# Risk Register

> Risks ranked by impact. Severity: Low | Medium | High | Critical. Last audited 2026-05-28.

---

## Risk: Hot-path performance regression goes undetected
- **Severity:** Medium
- **Area:** Engine core / CI
- **Description:** `copy()`, `step`, and enclosure have tight perf targets that gate MCTS
  viability, but no CI check enforces them.
- **Why it matters:** A silent regression would make 200-sim MCTS slow, undermining the project's
  central claim.
- **Evidence:** `CLAUDE.md` targets; `benchmarks/`; `.github/workflows/ci.yml` (no perf step).
- **Suggested mitigation:** Lightweight perf smoke test with generous thresholds, or a scheduled
  benchmark workflow.
- **Owner:** unassigned · **Status:** Open

## Risk: Untrusted checkpoint deserialization (`torch.load` / pickle)
- **Severity:** Medium
- **Area:** RL persistence
- **Description:** `.pt` checkpoints load via `torch.load` (pickle), which can execute arbitrary
  code if a file is malicious.
- **Why it matters:** Loading a checkpoint from an untrusted source is a code-execution vector.
- **Evidence:** `rl/alphazero/`, `rl/curriculum/`, `eval_*` scripts; `docs/phase3d/net_reference.pt`.
- **Suggested mitigation:** Only load trusted checkpoints; document this (done in SECURITY notes);
  consider `weights_only=True` loading where supported.
- **Owner:** unassigned · **Status:** Open (documented)

## Risk: Demo HTTP servers bound to a reachable interface
- **Severity:** Low–Medium
- **Area:** Visualization demos
- **Description:** `viz_live`/`viz_interactive` start stdlib HTTP servers; `--host` could bind to
  a non-loopback interface, exposing an unauthenticated server.
- **Why it matters:** No auth/validation layer; intended for local use only.
- **Evidence:** `viz_live.py`, `viz_interactive.py`; `--host`/`--port` flags.
- **Suggested mitigation:** Default to localhost; document local/dev-only usage (done).
- **Owner:** unassigned · **Status:** Open (documented)

## Risk: State-cache desync corrupts games silently
- **Severity:** Medium (low likelihood)
- **Area:** Engine core
- **Description:** If a future mutation path updates the grid but not `path_set`/counters, games
  could diverge without an obvious error.
- **Why it matters:** Caches are correctness-critical and trade safety for speed.
- **Evidence:** `state.py`, ADR-002.
- **Suggested mitigation:** Keep mutations centralized in `engine`; run `viz.check_invariants` in
  debug; the equivalence tests guard enclosure logic.
- **Owner:** unassigned · **Status:** Mitigated

## Risk: AlphaZero training regresses without gating
- **Severity:** Medium
- **Area:** RL pipeline
- **Description:** Latest snapshot always becomes champion, so a bad self-play iteration can
  degrade the agent unchecked.
- **Why it matters:** Limits trust in AlphaZero results.
- **Evidence:** `rl/alphazero/train.py:207-210`, ADR-005.
- **Suggested mitigation:** Implement gating (see KNOWN_ISSUES).
- **Owner:** unassigned · **Status:** Open

## Risk: Advisory-only lint/type gate lets quality drift
- **Severity:** Low
- **Area:** CI
- **Description:** ruff/mypy run with `continue-on-error`, so violations don't block merges.
- **Evidence:** `.github/workflows/ci.yml`.
- **Suggested mitigation:** Promote to required once consistently clean.
- **Owner:** unassigned · **Status:** Open

## Related docs
- [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) · [`SECURITY_AND_PRIVACY_NOTES.md`](SECURITY_AND_PRIVACY_NOTES.md) · [`REGRESSION_CHECKLIST.md`](REGRESSION_CHECKLIST.md)
