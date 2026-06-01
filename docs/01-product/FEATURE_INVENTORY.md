# Feature Inventory

> Canonical feature list with honest status labels. Evidence is cited as source paths.
> Status labels: **Implemented / Partial / Stubbed / Broken / Designed only / Deprecated / Unknown**.
> Maturity framing (Production/Reference/Experimental) comes from the README legend.
> Last audited 2026-05-28.

## Summary table

| Feature | Status | Module(s) |
|---|---|---|
| Core game engine (step, enclosure, winner) | Implemented | `engine.py` |
| Game state + cheap copy for tree search | Implemented | `state.py` |
| Legal actions / policy masking | Implemented | `actions.py` |
| Random-rollout fast path | Implemented | `rollout.py` |
| Heuristic evaluator + Voronoi + features | Implemented | `eval/` |
| Evaluator weight tuning | Implemented | `eval/tuning.py` |
| Search: Random / Greedy agents | Implemented | `search/random_agent.py` |
| Search: Max-N / Paranoid | Implemented | `search/maxn.py` |
| Search: UCT MCTS (+ progressive widening) | Implemented | `search/mcts/uct.py` |
| Search: RAVE MCTS | Implemented | `search/mcts/rave.py` |
| Rollout policies (uniform/informed/voronoi) | Implemented | `search/mcts/rollout.py` |
| Tournament harness (Wilson-CI, seat rotation, MP) | Implemented | `search/harness.py` |
| Elo rating | Implemented | `rl/eval/elo.py` |
| Gym environment (single + multi-agent) | Implemented | `gym_env.py` |
| RL: Tabular Q-learning | Implemented (Reference) | `rl/tabular/` |
| RL: Curriculum learning (checkpoint shipped) | Implemented (Reference) | `rl/curriculum/` |
| RL: PPO | **Partial** | `rl/ppo/` |
| RL: AlphaZero self-play + training | **Partial** | `rl/alphazero/` |
| AlphaZero snapshot **gating** tournament | **Stubbed** | `rl/alphazero/train.py` |
| Visualization: ASCII / matplotlib / GIF | Implemented | `viz.py` |
| Visualization: self-contained HTML replay | Implemented | `viz_html.py` |
| Live HTTP gameplay viewer | Implemented | `viz_live.py` |
| Interactive browser play (human-vs-agent + spectator) | Implemented | `viz_interactive.py` |
| Committed benchmark reports | Implemented (data) | `docs/baseline_report*.md`, `scripts/run_baseline_report.py` |

---

## Core game engine
- **Status:** Implemented (Production).
- **User value:** Deterministic, fast simulation kernel that every agent plays against.
- **Primary flow:** `new_game()` → `step(state, action)` → `StepResult(state, reward, done, info)`;
  enclosure claims resolved via `detect_and_apply_enclosure`; winner via `_compute_winner`.
- **Entry points:** library API; exercised by every script and the harness.
- **Modules:** `engine.py`, `state.py`, `actions.py`, `constants.py`.
- **Known issues:** none functional. Performance targets are checked manually, not in CI.
- **Evidence:** `engine.py` (517 LOC), `tests/test_engine.py`, `test_step.py`, `test_enclosure.py`.

## Classical search agents
- **Status:** Implemented (Production) — Random, Greedy, Max-N, Paranoid, UCT, RAVE.
- **User value:** Strong baselines and the head-to-head leaders (RAVE@200, UCT@200).
- **Primary flow:** construct an agent (directly or via `search/registry.py`), call
  `select_action(state, player_id, time_budget_s, max_iterations)`.
- **Entry points:** `scripts/run_tournament.py`, `scripts/run_baseline_report.py`.
- **Modules:** `search/random_agent.py`, `search/maxn.py`, `search/mcts/{uct,rave,rollout,node}.py`.
- **Known issues:** none; a few win-rate test thresholds were relaxed historically for stability.
- **Evidence:** `tests/test_search_*.py`, `test_mcts_*.py`; leaderboard `docs/baseline_report_20x20.md`.

## Tournament harness & evaluation
- **Status:** Implemented (Production).
- **User value:** Turns "which agent is better" into Wilson-CI-bounded numbers, reproducibly.
- **Primary flow:** `run_match` / `round_robin` over agents → `Table` with win rates + CIs;
  `compute_elo` for ratings. Seeds derive from one root via `SeedSequence`.
- **Entry points:** `scripts/run_tournament.py`, `run_baseline_report.py`, `compute_elo.py`.
- **Modules:** `search/harness.py`, `eval/heuristic.py`, `eval/voronoi.py`, `eval/features.py`, `rl/eval/elo.py`.
- **Known issues:** no automated benchmark CI (deliberate — expensive/flaky on shared runners).
- **Evidence:** `tests/test_harness.py`, `test_baseline_report.py`, `test_rl_eval_elo.py`.

## Gym environment
- **Status:** Implemented (Production).
- **User value:** RL-ecosystem compatibility; auto-plays opponents between agent turns.
- **Primary flow:** `TerritoryTakeoverEnv` (`Discrete(4)`, Dict obs with `action_mask`),
  `reset()`/`step()`/`render()`; `MultiAgentEnv` for AEC-style use.
- **Modules:** `gym_env.py` (lazy-imported from package root).
- **Known issues:** uses `Any` in obs/policy typing (debt, not a defect).
- **Evidence:** `tests/test_gym_env.py`.

## RL — Tabular Q-learning
- **Status:** Implemented (Reference; validated baseline).
- **User value:** Simplest learning agent; sanity baseline for the RL track.
- **Primary flow:** `scripts/train_tabular_q.py` (config in `configs/phase3a_*`) → trained
  agent; eval via `scripts/eval_tabular_q.py`. Artifacts in `docs/phase3a/`.
- **Modules:** `rl/tabular/{config,state_encoder,q_agent,reward,eval,train}.py`.
- **Known issues:** state encoding is lossy/compressible above small boards (by design).
- **Evidence:** `tests/test_rl_tabular_*.py`, `docs/phase3a/*summary.yaml`.

## RL — Curriculum learning
- **Status:** Implemented (Reference; checkpoint shipped).
- **User value:** Trains an AlphaZero-style net across a 6×6→8×8→10×10 schedule; shipped
  reference checkpoint is a tournament participant (`curriculum_ref`).
- **Primary flow:** `scripts/train_curriculum.py` (config `configs/phase3d_curriculum*.yaml`).
- **Modules:** `rl/curriculum/{schedule,trainer,transfer}.py`.
- **Known issues:** checkpoint is **out-of-distribution above 10×10**; strength does not scale
  monotonically with eval-time PUCT compute at 20×20. See `docs/experiments/20x20_hypothesis_test.md`.
- **Evidence:** `tests/test_rl_curriculum_*.py`, `docs/phase3d/net_reference.pt`.

## RL — PPO
- **Status:** **Partial** (Experimental). Primitives are implemented and unit-tested
  (`ActorCritic`, `RolloutBuffer`/GAE, `ppo_update_step`, `VectorizedEnv`, masked spaces),
  but there is **no orchestrated training/eval CLI driver** (`scripts/` has `train_tabular_q`,
  `train_alphazero`, `train_curriculum` — no `train_ppo`).
- **User value:** Reusable on-policy RL building blocks for the environment.
- **Modules:** `rl/ppo/{spaces,network,ppo_core,vec_env}.py`.
- **Known issues:** no end-to-end PPO training entry point or committed result.
- **Evidence:** `tests/test_rl_ppo_*.py` (core/network/spaces/vec_env).

## RL — AlphaZero (self-play + training)
- **Status:** **Partial** (Experimental). The self-play → replay → train loop runs end-to-end
  (`train_loop`, smoke-tested), but one pipeline component is deliberately deferred:
  - **AlphaZero snapshot gating: Stubbed.** The latest self-play snapshot always becomes the
    champion; there is no evaluation-gated promotion. TODO marker in `rl/alphazero/train.py`;
    decision recorded in ADR-005.
- **User value:** Demonstrates a working AlphaZero pipeline (PUCT MCTS + ResNet value/policy +
  cached/batched NN evaluator with virtual loss + replay buffer).
- **Primary flow:** `scripts/train_alphazero.py` (config `configs/phase3c_*`) → snapshots;
  eval via `scripts/eval_alphazero.py`.
- **Modules:** `rl/alphazero/{spaces,network,evaluator,mcts,replay,selfplay,train}.py`.
- **Known issues:** gating deferral (above); value-head quality flagged as the weakest link
  in `docs/experiments/20x20_hypothesis_test.md`.
- **Evidence:** `tests/test_rl_alphazero_*.py` (network/mcts/evaluator/replay/nstep/spaces/train_smoke).

## Visualization & demo surfaces
- **Status:** Implemented (Production) across five surfaces:
  - **ASCII / matplotlib / GIF** — `viz.py` (`render_ascii`, `render_matplotlib`, `save_game_gif`).
  - **Self-contained HTML replay** — `viz_html.py` (`save_game_html`, embedded CSS/JS, win-prob bars).
  - **Live HTTP viewer** — `viz_live.py` (`LiveServer`; streams each move; Reset button).
  - **Interactive browser play** — `viz_interactive.py` (`InteractiveServer`; human-vs-agent +
    spectator; agent presets).
- **User value:** Inspect games and regressions visually; recruiter-facing demo path.
- **Entry points:** `scripts/serve_live_demo.py`, `play_interactive.py`, `record_demo.py`,
  `record_html_demo.py`, `record_agent_gallery.py`.
- **Known issues:** HTTP servers are local/dev-only (see SECURITY notes); `Any` in payload typing.
- **Evidence:** `tests/test_viz.py`, `test_viz_html.py`, `test_viz_live.py`; `docs/assets/*`.

## Committed benchmark reports
- **Status:** Implemented (data artifact).
- **User value:** Reviewer-scannable, reproducible-from-seed leaderboards.
- **Entry points:** `scripts/run_baseline_report.py`, `run_puct_scaling.py`.
- **Evidence:** `docs/baseline_report.md`, `docs/baseline_report_20x20.md`,
  `docs/curriculum_puct_scaling.md` (+ matching CSVs).
