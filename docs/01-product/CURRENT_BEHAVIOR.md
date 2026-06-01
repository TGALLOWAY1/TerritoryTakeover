# Current Behavior

> What the project actually does **today**, grounded in the test suite, committed
> artifacts, and source. Evidence is cited. Anything not directly verified is marked
> *inferred*. Last audited 2026-05-28.

## Verified working (test-backed)

The following run end-to-end and are covered by the 405-test suite:

- **Play a full game.** `new_game` → repeated `step` → terminal state with a winner (or tie).
  Illegal moves mark the player dead and advance the turn (RL-friendly default) or raise
  `IllegalMoveError` under `strict=True`. Evidence: `tests/test_engine.py`, `test_step.py`.
- **Enclosure claims.** Closing a loop floods the enclosed empty region to the triggering
  player; claimed counts stay in lockstep with the grid. Evidence: `tests/test_enclosure.py`,
  `engine.py::detect_and_apply_enclosure` (validated against `_legacy_..._full_bfs`).
- **Legal-action queries and masking.** `legal_action_mask`/`legal_actions` return the four
  N/S/W/E legality bits/indices. Evidence: `tests/test_actions.py`.
- **All classical agents select moves.** Random, Greedy, Max-N, Paranoid, UCT, RAVE.
  Evidence: `tests/test_search_*.py`, `test_mcts_*.py`.
- **Tournaments produce Wilson-CI tables.** `run_match`/`round_robin` with seat rotation and
  optional multiprocessing; serial and parallel runs are bit-identical for a fixed seed.
  Evidence: `tests/test_harness.py`, `test_baseline_report.py`, ADR-006.
- **Gym env steps.** `TerritoryTakeoverEnv` resets/steps with action masking and auto-opponents;
  `MultiAgentEnv` AEC-style cycling. Evidence: `tests/test_gym_env.py`.
- **Tabular Q training (smoke).** Train loop runs and improves vs. baselines on small boards.
  Evidence: `tests/test_rl_tabular_train_smoke.py`, artifacts in `docs/phase3a/`.
- **AlphaZero pipeline (smoke).** Self-play → replay → SGD train step runs end-to-end on a
  tiny config. Evidence: `tests/test_rl_alphazero_train_smoke.py`, `test_rl_alphazero_mcts.py`.
- **PPO primitives.** Network forward pass, GAE, clipped update, vectorized env stepping.
  Evidence: `tests/test_rl_ppo_*.py`.
- **Curriculum training (smoke) + weight transfer.** Evidence:
  `tests/test_rl_curriculum_trainer_smoke.py`, `test_rl_curriculum_transfer.py`.
- **Visualization.** ASCII/matplotlib/GIF render; HTML replay generation; live + interactive
  HTTP viewers. Evidence: `tests/test_viz.py`, `test_viz_html.py`, `test_viz_live.py`.

## Verified by committed artifacts

- **20×20 leaderboard** (RAVE@200 first at 0.762 [0.659, 0.842]) — `docs/baseline_report_20x20.md`.
- **10×10 sanity baseline** — `docs/baseline_report.md`.
- **Curriculum PUCT scaling sweep** — `docs/curriculum_puct_scaling.md`.
- **Demo media** — `docs/assets/demo.gif`, `agent_gallery.png`, `h2h_heatmap.png`,
  `territory_growth.png`, `mcts_scaling.png`, `best_agent_demo.html`.
- **Reference curriculum checkpoint** — `docs/phase3d/net_reference.pt` (+ config + Elo logs).

## Partial / deferred behavior

- **AlphaZero gating is stubbed.** The latest self-play snapshot is always promoted to
  champion; no evaluation-gated promotion exists. The pipeline trains, but this branch is a
  documented TODO (`rl/alphazero/train.py`, ADR-005, KNOWN_ISSUES).
- **PPO has no training driver.** Primitives are tested, but there is no `scripts/train_ppo.py`
  or committed PPO result.
- **Curriculum checkpoint is out-of-distribution above 10×10** — accepts arbitrary `H×W` (conv
  head) but does not scale monotonically with eval-time PUCT compute at 20×20
  (`docs/experiments/20x20_hypothesis_test.md`).

## Performance behavior (targets, checked manually — not in CI)

| Operation | Target | Source |
|---|---|---|
| `GameState.copy()` | < 50 µs | `CLAUDE.md`, `tests/test_state.py` (loose baseline) |
| `legal_actions()` | < 1 µs | `CLAUDE.md`, `tests/test_actions.py` |
| `detect_and_apply_enclosure` (40×40, post-trigger) | < 200 µs | `CLAUDE.md`, `benchmarks/` |

*Inferred:* these targets are asserted loosely or documented rather than enforced on every CI
run; treat them as design intent verified by the benchmark harness, not continuous guarantees.

## Not present
No persistent user data, accounts, network services beyond local demo viewers, scheduled jobs,
or external API calls. See [`docs/03-implementation/CODEBASE_INVENTORY.md`](../03-implementation/CODEBASE_INVENTORY.md).
