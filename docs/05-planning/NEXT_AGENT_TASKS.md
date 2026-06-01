# Next Agent Tasks

> Ready-to-run task prompts for a future AI agent. Each is self-contained: read the listed
> context, edit the listed files, satisfy the acceptance criteria, run the checks, commit.
> Load only the listed docs (see [`docs/07-ai-context/CONTEXT_LOADING_PROTOCOL.md`](../07-ai-context/CONTEXT_LOADING_PROTOCOL.md)).
> Last audited 2026-05-28.

---

## Task B1 — Implement the AlphaZero gating tournament

**Context to read:**
- `docs/02-architecture/ARCHITECTURE.md`, `docs/02-architecture/DATA_MODEL.md`
- `docs/04-quality/KNOWN_ISSUES.md` (gating issue), `docs/adr/ADR-005-alphazero-gating.md`
- Source: `rl/alphazero/train.py`, `rl/alphazero/evaluator.py`, `rl/alphazero/mcts.py`, `search/harness.py`

**Goal:** Replace the `TODO` at `rl/alphazero/train.py:207-210` with a gating tournament: after
each train iteration, play the candidate snapshot against the current champion over N seed-locked
games; promote the candidate to champion only if its win rate clears a configurable threshold.

**Non-goals:** Don't refactor the MCTS or network; don't change the obs encoding; don't add new
dependencies; don't touch unrelated RL tracks.

**Acceptance criteria:**
- `TrainConfig` gains gating knobs (e.g. `gating_games`, `gating_threshold`); defaults preserve a
  sensible behavior and are documented.
- Promotion is gated; a deliberately-worse candidate is **not** promoted (covered by a test).
- Self-play continues to use the current champion.

**Checks:** `pytest tests/test_rl_alphazero_*`, `ruff check .`, `mypy`; small run of
`scripts/train_alphazero.py --config configs/phase3c_alphazero_8x8_2p.yaml --num-iterations 2`.

**Commit:** `rl(alphazero): gate snapshot promotion on a champion tournament`. Update
`docs/04-quality/KNOWN_ISSUES.md`, `docs/01-product/FEATURE_INVENTORY.md` (status → Implemented),
and `docs/adr/` (supersede ADR-005 with a new ADR rather than editing it).

---

## Task B3 — Add a performance regression guard

**Context to read:**
- `CLAUDE.md` (perf targets), `docs/02-architecture/ARCHITECTURE.md`, `docs/04-quality/RISK_REGISTER.md`
- Source: `benchmarks/bench_engine.py`, `benchmarks/baseline.json`, `.github/workflows/ci.yml`

**Goal:** Catch large hot-path regressions automatically. Either (a) a `pytest`-marked perf smoke
test with **generous** thresholds (e.g. 3× the documented target to avoid CI flakiness), or
(b) a scheduled GitHub Actions workflow that runs `benchmarks/bench_engine.py` and compares to
`baseline.json`.

**Non-goals:** Don't make perf a hard PR gate with tight thresholds (shared runners are noisy);
don't add benchmark runs to the existing per-PR job.

**Acceptance criteria:** intentionally slowing `GameState.copy()` makes the check fail; normal code
passes comfortably. Document the chosen approach in `docs/03-implementation/TESTING_STRATEGY.md`.

**Checks:** `pytest` (incl. the new check if option a), `ruff check .`, `mypy`.

**Commit:** `ci/bench: add hot-path performance regression guard`.

---

## Task B5 — Embed run metadata in benchmark reports

**Context to read:** `docs/02-architecture/ARCHITECTURE.md` (reproducibility), `docs/adr/ADR-006-seedsequence-reproducibility.md`;
source `search/harness.py`, `scripts/run_baseline_report.py`, `scripts/run_puct_scaling.py`.

**Goal:** Each committed report footer records the root seed, git SHA, and a config hash so any
table is traceable to exact inputs.

**Non-goals:** Don't change result computation or table format beyond the footer.

**Acceptance criteria:** regenerated reports include the metadata footer; a test asserts the
footer fields are present.

**Checks:** `pytest tests/test_baseline_report.py`, `ruff check .`, `mypy`.

**Commit:** `harness: stamp reports with seed + git SHA + config hash`.

---

## Task B2 — Add a PPO training driver

**Context to read:** `docs/01-product/FEATURE_INVENTORY.md` (PPO Partial), `docs/02-architecture/STATE_AND_ENCODING.md` (PPO encoder);
source `rl/ppo/{spaces,network,ppo_core,vec_env}.py`, an existing trainer like `scripts/train_tabular_q.py` as a CLI template.

**Goal:** Add `scripts/train_ppo.py` wiring the existing primitives into an end-to-end training
loop driven by a YAML config; produce one committed result and register PPO as a tournament agent.

**Non-goals:** Don't rewrite the PPO primitives; match the existing CLI conventions.

**Acceptance criteria:** training runs end-to-end on a small config; a `*_train_smoke.py` test
passes; PPO is selectable via `search/registry.py`.

**Checks:** `pytest tests/test_rl_ppo_*` + new smoke test, `ruff check .`, `mypy`.

**Commit:** `rl(ppo): add end-to-end PPO training CLI and smoke test`.

---

## Reminder for any task
After code changes, update the matching docs and status labels, run the
[`REGRESSION_CHECKLIST`](../04-quality/REGRESSION_CHECKLIST.md), and add an
[`AUDIT_LOG`](../06-history/AUDIT_LOG.md) / [`DECISION_LOG`](../06-history/DECISION_LOG.md) entry
if the change is architecturally significant.
