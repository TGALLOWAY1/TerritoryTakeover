# Testing Strategy

> How the project is tested, the conventions, and where the gaps are.
> Last audited 2026-05-28.

## At a glance
- **405 tests across 45 files** in `tests/`, run by `pytest` (`testpaths=["tests"]`, `addopts="-ra"`).
- **CI runs the full suite** on Python 3.11 and 3.12 as a **required** step (ruff/mypy are advisory).
- `mypy --strict` covers `tests/` too (annotations relaxed only via ruff `ANN` per-file ignore).

## Conventions (deliberate, see CLAUDE.md)
- **Function-based tests only** — no test classes, no `conftest.py`, fixtures limited to trivial helpers.
- **No `@pytest.mark.parametrize`** — mypy strict + missing pytest stubs flag the decorator as
  untyped; instead tests loop over `range(N)` with `f"trial={i}"` assertion messages.
- Tests carry docstrings explaining the scenario; setup is inline per test.

## Coverage map (representative)
| Area | Files |
|---|---|
| Engine / state / actions / step / enclosure | `test_engine.py`, `test_state.py`, `test_actions.py`, `test_step.py`, `test_enclosure.py`, `test_engine_equivalence.py` |
| Rollout / rollout API | `test_mcts_rollouts.py`, `test_rollout_api.py` |
| Classical search | `test_search_agents.py`, `test_search_maxn.py`, `test_mcts_uct.py`, `test_mcts_rave.py`, `test_mcts_node.py`, `test_mcts_progressive_widening.py`, `test_puct_scaling.py` |
| Eval / tuning | `test_eval_features.py`, `test_eval_heuristic.py`, `test_tuning.py` |
| Harness / reporting / Elo | `test_harness.py`, `test_baseline_report.py`, `test_rl_eval_elo.py` |
| Gym | `test_gym_env.py` |
| RL tabular | `test_rl_tabular_*` (agent, reward, state_encoder, train_smoke) |
| RL PPO | `test_rl_ppo_*` (core, network, spaces, vec_env) |
| RL AlphaZero | `test_rl_alphazero_*` (network, variable_size, mcts, evaluator, replay, nstep_target, spaces, train_smoke) |
| RL curriculum | `test_rl_curriculum_*` (schedule, trainer_smoke, transfer) |
| Visualization | `test_viz.py`, `test_viz_html.py`, `test_viz_live.py` |
| Demo recording | `test_record_demo.py` |

## Test types in use
- **Unit / property** — engine invariants, action legality, evaluator features, encoders.
- **Equivalence** — optimized enclosure BFS vs. the legacy full-BFS reference (`test_engine_equivalence.py`).
- **Determinism** — serial vs. multiprocessing tournament runs are bit-identical (harness).
- **Smoke** — RL training loops (`*_train_smoke.py`) run a few steps to prove the pipeline executes.
- **Statistical (relaxed)** — some win-rate thresholds were loosened historically to avoid
  flakiness from small samples (e.g. UCT-vs-random, informed-rollout-beats-uniform).

## Performance "tests"
Hot-path targets (`copy()` < 50 µs, `legal_actions` < 1 µs, enclosure < 200 µs on 40×40) are
documented in `CLAUDE.md`, loosely asserted in `tests/test_state.py`/`test_actions.py`, and
measured by `benchmarks/`. They are **not** enforced as strict CI gates.

## Gaps / what's under-tested
- **PPO has only primitive-level tests** — no end-to-end training/eval (no PPO training driver exists).
- **AlphaZero gating** is unimplemented, so there is nothing to test there (Stubbed).
- **No automated benchmark/perf regression CI** — perf relies on manual benchmark runs.
- **HTTP demo servers** are smoke-tested (`test_viz_live.py`) but interactive browser flows
  (`viz_interactive.py`) are exercised by hand, not automated.

## Running
```bash
pytest                         # full suite
pytest tests/test_enclosure.py -v
pytest -k test_simple_3x3_loop
ruff check . ; mypy
```

## Related docs
- Regression checklist before changes: [`docs/04-quality/REGRESSION_CHECKLIST.md`](../04-quality/REGRESSION_CHECKLIST.md)
- Known issues: [`docs/04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md)
