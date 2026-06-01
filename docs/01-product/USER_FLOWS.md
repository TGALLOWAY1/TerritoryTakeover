# User Flows

> The main workflows a developer/researcher runs, expressed end-to-end. "User" here means
> a developer using the library/CLI, not an end-user of an app. Last audited 2026-05-28.

## Flow 1 — Regenerate the canonical baseline leaderboard
- **Goal:** Reproduce the committed 20×20 head-to-head report from a single seed.
- **Entry point:** `scripts/run_baseline_report.py`.
- **Steps:** `python scripts/run_baseline_report.py --board-size 20 --games-per-pair 20 --parallel --seed 0`
- **Expected behavior:** 5-way round-robin (Random/Greedy/UCT@200/RAVE@200/curriculum_ref),
  seat-rotated, Wilson-CI table written to markdown + CSV (`--md-out`/`--csv-out`). ~30 min on 16 cores.
- **Failure states:** missing curriculum checkpoint → use `--skip-curriculum`; missing `torch`
  → curriculum/AZ participants unavailable.
- **Relevant modules:** `search/harness.py`, `search/registry.py`, `eval/`, `rl/curriculum/`.
- **Open questions:** wall-clock varies by machine; numbers in `docs/baseline_report_20x20.md`.

## Flow 2 — Watch a live game in the browser
- **Goal:** Visually inspect agent behavior in real time.
- **Entry point:** `scripts/serve_live_demo.py`.
- **Steps:** `python scripts/serve_live_demo.py` → open the printed URL. Choose agents with
  `--seat0/--seat1` (`random`/`greedy`/`rave`); press **Reset** in the page for a new game.
- **Expected behavior:** Page streams each move (`GET /state`), shows per-seat win-probability
  bars; `POST /reset` restarts. `--once` plays a single game; `--no-browser` skips auto-open.
- **Failure states:** port in use → `--port`; headless host → use `--no-browser` and tunnel.
- **Relevant modules:** `viz_live.py` (`LiveServer`, `play_and_serve`), agents via registry.

## Flow 3 — Play against an agent (human-vs-agent)
- **Goal:** Drive a seat with the keyboard against a chosen agent; or spectate.
- **Entry point:** `scripts/play_interactive.py`.
- **Steps:** launch → open URL → `GET /agents` lists presets → `POST /start` begins → arrow
  keys send `POST /action` (direction `0..3`).
- **Expected behavior:** interactive board with agent presets (random/greedy/MCTS easy/medium/hard).
- **Relevant modules:** `viz_interactive.py` (`InteractiveServer`, `AGENT_PRESETS`).

## Flow 4 — Record a shareable replay
- **Goal:** Produce a demo artifact (GIF or self-contained HTML).
- **Entry points:** `scripts/record_demo.py` (GIF), `scripts/record_html_demo.py` (HTML).
- **Steps (HTML):** `python scripts/record_html_demo.py --seed 0 --board-size 20 --seat0 rave
  --seat1 greedy --win-prob heuristic --out demo.html`.
- **Expected behavior:** a single self-contained file (embedded CSS/JS, frame-by-frame replay,
  win-prob bars). GIF path uses matplotlib + pillow.
- **Failure states:** missing `matplotlib`/`pillow` for GIF → `pip install -e ".[viz]"`.
- **Relevant modules:** `viz.py` (GIF), `viz_html.py` (HTML).

## Flow 5 — Train an RL agent
- **Goal:** Train a learning agent from a YAML config.
- **Entry points:** `train_tabular_q.py`, `train_alphazero.py`, `train_curriculum.py`.
- **Steps (tabular):** `python scripts/train_tabular_q.py --config configs/phase3a_tabular_8x8_2p.yaml --seed 0 --out-dir results/myrun`.
- **Expected behavior:** training loop runs to the configured budget, periodically snapshots;
  eval via the matching `eval_*` script. AlphaZero trains end-to-end but **always promotes the
  latest snapshot** (gating stubbed — see KNOWN_ISSUES).
- **Failure states:** neural tracks require `torch` (`pip install -e ".[rl_deep]"`).
- **Relevant modules:** `rl/tabular/`, `rl/alphazero/`, `rl/curriculum/`.

## Flow 6 — Use the engine as a library (incl. Gym)
- **Goal:** Drive the simulation directly or via Gymnasium for custom agents/RL.
- **Steps:** `from territory_takeover import new_game, step` → loop on `step`; or
  `from territory_takeover import TerritoryTakeoverEnv` for the Gym API (`Discrete(4)`,
  Dict obs with `action_mask`).
- **Expected behavior:** deterministic transitions; illegal moves kill the player by default.
- **Relevant docs:** [`docs/02-architecture/PUBLIC_API.md`](../02-architecture/PUBLIC_API.md),
  [`STATE_AND_ENCODING.md`](../02-architecture/STATE_AND_ENCODING.md).

## Related docs
- All entry points + flags: [`docs/03-implementation/ENTRYPOINTS_AND_SCRIPTS.md`](../03-implementation/ENTRYPOINTS_AND_SCRIPTS.md).
- Visual outputs of flows 2–4: [`docs/08-visuals/`](../08-visuals/SCREENSHOT_MANIFEST.md).
