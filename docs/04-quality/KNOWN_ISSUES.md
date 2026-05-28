# Known Issues

> Concrete, evidence-backed issues with current behavior. These are deliberate scope cuts or
> documented limitations, not hidden bugs. No functional defects were found in the core engine
> at audit time. Last audited 2026-05-28.

---

## Issue: AlphaZero snapshot gating is stubbed
- **Status:** Stubbed (deliberate Phase-3c scope cut).
- **Severity:** Medium (limits AlphaZero training quality, not correctness).
- **User impact:** The self-play "champion" is always the latest snapshot; there is no
  evaluation-gated promotion, so training can regress without a guardrail.
- **Technical cause:** The gating tournament is a `TODO` in the train loop.
- **Relevant files:** `src/territory_takeover/rl/alphazero/train.py:207-210` (and docstring
  `:12-14`); rationale in `docs/adr/ADR-005-alphazero-gating.md`.
- **Suggested fix:** Implement a gating tournament (new snapshot vs. current champion over N
  seed-locked games); promote only on a win-rate threshold. See
  [`docs/05-planning/NEXT_AGENT_TASKS.md`](../05-planning/NEXT_AGENT_TASKS.md).
- **Verification steps:** Add a test asserting a deliberately-worse snapshot is *not* promoted;
  run `scripts/train_alphazero.py` on a small config and confirm gating logs.

## Issue: Curriculum checkpoint is out-of-distribution above 10×10
- **Status:** Partial (documented limitation).
- **Severity:** Medium.
- **User impact:** `curriculum_ref` underperforms compute-matched MCTS at 20×20 and its strength
  does not scale monotonically with eval-time PUCT iterations.
- **Technical cause:** Trained on a 6×6→8×8→10×10 schedule; the conv head accepts arbitrary
  `H×W` but the value head is OOD at larger boards.
- **Relevant files:** `rl/curriculum/`, checkpoint `docs/phase3d/net_reference.pt`;
  evidence in `docs/experiments/20x20_hypothesis_test.md`.
- **Suggested fix:** Extend the curriculum schedule (add 20×20) and/or improve value-head
  accuracy (flagged as the pipeline's weakest link).
- **Verification steps:** Re-run `scripts/run_puct_scaling.py` at 20×20 and check for monotonic
  win-rate vs. iterations.

## Issue: PPO has no end-to-end training driver
- **Status:** Partial.
- **Severity:** Low–Medium (primitives are sound; the track is incomplete).
- **User impact:** Cannot train or benchmark a PPO agent out of the box.
- **Technical cause:** `rl/ppo/` provides network/buffer/update/vec-env primitives but no
  orchestrated `train_ppo.py` CLI or committed result.
- **Relevant files:** `rl/ppo/{spaces,network,ppo_core,vec_env}.py`; no `scripts/train_ppo.py`.
- **Suggested fix:** Add a PPO training script wiring the primitives, plus a smoke test.
- **Verification steps:** New `scripts/train_ppo.py` runs a few updates; PPO appears as a
  tournament participant.

## Issue: No automated performance / benchmark regression in CI
- **Status:** Partial (deliberate).
- **Severity:** Low–Medium.
- **User impact:** A perf regression in the hot path (`copy`, `step`, enclosure) would not be
  caught automatically; only the local benchmark harness catches it.
- **Technical cause:** Benchmarks are expensive/flaky on shared runners; reports are run locally
  and committed instead.
- **Relevant files:** `.github/workflows/ci.yml`, `benchmarks/`, `CLAUDE.md` perf targets.
- **Suggested fix:** A lightweight perf smoke test with generous thresholds, or a manual,
  scheduled benchmark workflow. See [`RISK_REGISTER.md`](RISK_REGISTER.md).
- **Verification steps:** Intentionally slow a hot path locally and confirm the benchmark flags it.

## Issue: ruff/mypy are advisory in CI
- **Status:** Partial (by design).
- **Severity:** Low.
- **User impact:** Lint/type regressions can merge without failing CI (`continue-on-error: true`).
- **Relevant files:** `.github/workflows/ci.yml`.
- **Suggested fix:** Once the codebase is consistently clean, flip ruff/mypy to required.
- **Verification steps:** Run `ruff check .` and `mypy` locally before merging.

## Related docs
- Technical debt (code-level): [`TECHNICAL_DEBT.md`](TECHNICAL_DEBT.md)
- Risks (impact-ranked): [`RISK_REGISTER.md`](RISK_REGISTER.md)
- Backlog items derived from these: [`docs/05-planning/BACKLOG.md`](../05-planning/BACKLOG.md)
