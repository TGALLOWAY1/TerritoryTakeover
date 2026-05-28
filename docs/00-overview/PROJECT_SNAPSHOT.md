# Project Snapshot

> One-screen orientation. For the full map see
> [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md). Last audited: 2026-05-28.

## What it is

`territory_takeover` is a **decision-systems AI research library** (not a product app).
A turn-based, multi-agent, grid territory-control game is the shared testbed; six
search/RL algorithm families are implemented against it and benchmarked head-to-head
under a seed-locked tournament harness with committed Wilson-CI leaderboards.

- **Package:** `territory_takeover` (src/ layout), version `0.1.0`, MIT licensed.
- **Language/runtime:** Python **3.11+** (CI matrix: 3.11, 3.12).
- **Only runtime dependency:** `numpy>=1.26`. Everything else (`torch`, `gymnasium`,
  `matplotlib`, `pillow`, `pyyaml`, `tensorboardX`) is an **optional extra**.
- **Size:** ~13,000 LOC in `src/`, 54 Python modules; **405 tests across 45 files**.

## Stack at a glance

| Concern | Choice |
|---|---|
| Build backend | `setuptools.build_meta` (`pyproject.toml`, src/ layout) |
| Package manager | `pip` (editable install + extras) |
| Test runner | `pytest` (`testpaths=["tests"]`, `addopts="-ra"`) |
| Lint | `ruff` (E,F,I,B,UP,N,SIM,RUF,ANN,TID; line length 100) |
| Type check | `mypy` **strict** (src/ + tests/) |
| CI | GitHub Actions (`.github/workflows/ci.yml`): pytest **required**, ruff + mypy **advisory** (`continue-on-error`) |
| Web / DB / auth | **None** — this is a library; only optional stdlib `http.server` demo viewers exist |

## Commands

```bash
pip install -e ".[dev]"   # dev install (pulls torch/gym/viz/yaml/tensorboardX)
pytest                    # full suite (405 tests)
pytest tests/test_enclosure.py -v
ruff check .
mypy                      # strict, configured in pyproject.toml
```

## Subsystem status at a glance

Maturity labels follow the README legend; see
[`docs/01-product/FEATURE_INVENTORY.md`](../01-product/FEATURE_INVENTORY.md) for the
per-subsystem status using the project's standard status labels.

| Subsystem | Maturity | Key modules |
|---|---|---|
| Core engine (step, enclosure, winner) | Production | `engine.py` |
| Game state (cheap-copy for tree search) | Production | `state.py` |
| Actions / legal-move masking | Production | `actions.py` |
| Rollout fast path | Production | `rollout.py` |
| Search: Random / Greedy | Production | `search/random_agent.py` |
| Search: Max-N / Paranoid | Production | `search/maxn.py` |
| Search: UCT / RAVE MCTS | Production | `search/mcts/` |
| Tournament harness (Wilson-CI, seat rotation) | Production | `search/harness.py` |
| Evaluation / heuristics / Voronoi | Production | `eval/` |
| Gym environment | Production | `gym_env.py` |
| Visualization (ASCII/matplotlib/GIF/HTML/HTTP) | Production | `viz*.py` |
| RL: Tabular Q-learning | Reference | `rl/tabular/` |
| RL: Curriculum (checkpoint shipped) | Reference | `rl/curriculum/` |
| RL: PPO primitives | Experimental | `rl/ppo/` |
| RL: AlphaZero (**gating deferred/stubbed**) | Experimental / **Partial** | `rl/alphazero/` |

The single known deliberate stub is the AlphaZero snapshot-**gating** tournament
(`rl/alphazero/train.py`): the latest self-play snapshot always becomes the champion;
there is no evaluation-gated promotion. See
[`docs/04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md).

## Repo layout

```
src/territory_takeover/   engine, state, actions, rollout, gym_env, viz*, eval/, search/, rl/
tests/                    45 pytest files (~405 tests)
scripts/                  19 CLI entry points (training, tournaments, demos, recordings)
benchmarks/               perf harnesses + committed JSON baselines + *FINDINGS.md
configs/                  10 YAML training/experiment configs
docs/                     this documentation system + ADRs, baselines, experiments, assets
experiments/              curriculum ablation script
results/                  run artifacts (git-ignored except force-added references)
```

## Where to start
- New contributor: [`docs/00-overview/README.md`](README.md) → root `README.md`.
- AI agent: [`docs/07-ai-context/CONTEXT_LOADING_PROTOCOL.md`](../07-ai-context/CONTEXT_LOADING_PROTOCOL.md).
- Architecture: [`docs/02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md).
- What works / what's deferred: [`docs/01-product/CURRENT_BEHAVIOR.md`](../01-product/CURRENT_BEHAVIOR.md).
