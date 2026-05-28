# Product Brief

> What this project is, who it's for, and what it demonstrates. Last audited 2026-05-28.

## One-liner

**A reproducible multi-agent decision-systems AI lab.** Six search and RL algorithm
families — Max-N / Paranoid, UCT, RAVE, Tabular Q-learning, AlphaZero self-play, and
curriculum learning — are implemented against one shared turn-based grid environment and
benchmarked head-to-head under a seed-locked tournament harness with committed
Wilson-CI leaderboards.

## The product is the *method*, not the game

The territory-control game (snakes claim territory by enclosing regions on an `N×N` grid)
is a **testbed**, deliberately small enough to hold in your head. The actual deliverable is
the discipline around it:

- a deterministic, performant simulation core (tuned so 200-simulation-per-move MCTS is cheap),
- a layered algorithm stack (classical tree search + modern RL) all sharing one environment,
- a harness that turns "which agent is better?" into a concrete Wilson-CI-bounded number,
- committed benchmark reports any reviewer can regenerate from a single integer seed.

## Audiences

| Audience | What they need | Where to look |
|---|---|---|
| Recruiter / hiring manager | 3-minute proof of depth | root `README.md` (headline leaderboard, demo GIF, subsystem table) |
| Engineer evaluating the code | architecture + API + quality | [`02-architecture/`](../02-architecture/ARCHITECTURE.md), [`PUBLIC_API.md`](../02-architecture/PUBLIC_API.md), [`04-quality/`](../04-quality/KNOWN_ISSUES.md) |
| Researcher / collaborator | reproducible experiments | `docs/baseline_report*.md`, `docs/experiments/`, `docs/adr/` |
| Future AI agent | minimal correct context to continue work | [`07-ai-context/CONTEXT_LOADING_PROTOCOL.md`](../07-ai-context/CONTEXT_LOADING_PROTOCOL.md) |

## What it demonstrates (claims, each backed by code/reports)

- **Simulation engine design & performance engineering** — single int8-grid state with
  cheap copy semantics; documented hot-path targets (`copy()` < 50 µs, `legal_actions` < 1 µs,
  enclosure < 200 µs on 40×40). Evidence: `engine.py`, `state.py`, `benchmarks/`, `docs/OPTIMIZATION_ANALYSIS.md`.
- **Algorithmic breadth** — classical search and deep RL in one domain with cross-comparison
  infrastructure. Evidence: `search/`, `rl/`.
- **Measurement discipline** — seed-locked tournaments, Wilson 95% CIs, head-to-head matrices,
  committed reports. Evidence: `search/harness.py`, `docs/baseline_report_20x20.md`.
- **Reproducibility** — every per-game seed and per-agent RNG derives from one root integer
  via `numpy.random.SeedSequence`; serial and multiprocessing runs are bit-identical.
  Evidence: `search/harness.py::run_match`, ADR-006.

## Scope / non-goals

- **Not** a deployable application, service, or game product. There is no UI app, server
  backend, database, or user accounts — only optional local demo viewers for inspection.
- **Not** a from-scratch deep-learning framework — it builds on `numpy` (core) and optional
  `torch`/`gymnasium`.
- The AlphaZero track is **experimental**: the self-play→train loop runs end-to-end, but the
  snapshot-gating tournament is deliberately deferred (see [`FEATURE_INVENTORY.md`](FEATURE_INVENTORY.md)).

## Headline result (current, 20×20 / 2 players)

5-way round-robin, 20 games/pair. RAVE@200 leads (win rate 0.762, CI [0.659, 0.842]),
ahead of UCT@200, then the curriculum AlphaZero reference (compute-asymmetric), then
Greedy and Random. Full table + reproducibility footer in `docs/baseline_report_20x20.md`.
