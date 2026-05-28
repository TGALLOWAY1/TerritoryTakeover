# Entry Points and Scripts

> The library equivalent of a route inventory. Two kinds of entry points exist:
> **CLI scripts** (`scripts/*.py`) and **local HTTP demo endpoints** (stdlib `http.server`
> in `viz_live.py` / `viz_interactive.py`). No web framework or REST API. Last audited 2026-05-28.

## CLI scripts (`scripts/`)

All are `python scripts/<name>.py [flags]`; pass `--help` for full usage. Status is
**Implemented** unless noted. Flags below are the notable ones (not exhaustive).

### Training
| Script | Purpose | Key flags |
|---|---|---|
| `train_tabular_q.py` | Phase-3a tabular Q-learning training. | `--config --seed --num-episodes --out-dir --tag` |
| `train_alphazero.py` | Phase-3c AlphaZero training (self-play → train loop; **gating stubbed**). | `--config --seed --num-iterations --games-per-iteration --out-dir --tag --device` |
| `train_curriculum.py` | Phase-3d curriculum trainer. | `--config --seed --out-dir --device --tag` |
| `tune_weights.py` | Two-stage evaluator-weight tuner (evolutionary). | `--board-size --seed --stage-a/-b-* --validation-games --parallel` |

> No `train_ppo.py` exists — PPO is **Partial** (primitives only). See FEATURE_INVENTORY.

### Evaluation & reporting
| Script | Purpose | Key flags |
|---|---|---|
| `run_tournament.py` | Run a tournament from a YAML config. | `--config --output --mode --parallel/--no-parallel` |
| `run_baseline_report.py` | Generate the canonical baseline report (markdown + CSV). | `--games-per-pair --board-size --seed --uct/rave/az-iterations --checkpoint --parallel --dry-run` |
| `run_puct_scaling.py` | Sweep curriculum eval-time PUCT iters vs. a fixed panel. | `--board-size --games-per-opponent --az-iters --uct-iterations --parallel` |
| `compute_elo.py` | Round-robin Elo evaluator (Phase 3d). | `--config --board-size --games-per-pair --anchor --seed --out` |
| `eval_alphazero.py` | Evaluate an AlphaZero checkpoint vs. baselines. | `--checkpoint --config --games --uct-iters --mcts-iters --device` |
| `eval_tabular_q.py` | Evaluate a TabularQAgent checkpoint vs. baselines. | `--checkpoint --board-size --games --uct-iters --plot` |

### Recording / rendering (produce the `docs/assets/` media)
| Script | Purpose | Key flags |
|---|---|---|
| `record_demo.py` | Deterministic demo game → animated GIF. | `--seed --board-size --rave-iterations --az-iterations --frame-stride --fps --out --dry-run` |
| `record_agent_gallery.py` | 4-panel agent-behavior gallery PNG. | `--seed --board-size --target-turn --uct/rave-iterations --out --dry-run` |
| `record_html_demo.py` | Game → self-contained HTML viewer (win-prob bars). | `--seed --board-size --num-players --seat0..3 --win-prob --elo-csv --fps --out` |
| `record_territory_growth.py` | Territory-growth plot for one game. | `--seed --board-size --rave-iterations --out --dry-run` |
| `record_mcts_scaling.py` | UCT-vs-random win-rate-vs-compute plot. | `--seed --board-size --games --iterations --out --dry-run` |
| `render_h2h_heatmap.py` | 20×20 head-to-head matrix heatmap PNG. | `--out` |

Most recording scripts support `--dry-run` (compute without writing) — useful in CI/tests.

### Interactive demos (launch HTTP servers)
| Script | Purpose | Key flags |
|---|---|---|
| `serve_live_demo.py` | Stream a live agent-vs-agent game to a browser. | `--host --port --seed --board-size --num-players --seat0..3 --rave-iterations --fps --no-browser --once` |
| `play_interactive.py` | Launch the human-vs-agent / spectator frontend. | `--host --port --title --no-browser` |

## Local HTTP demo endpoints

These are **local development/inspection servers** (stdlib `http.server`), not a deployed
API. See [`docs/04-quality/SECURITY_AND_PRIVACY_NOTES.md`](../04-quality/SECURITY_AND_PRIVACY_NOTES.md).

### `viz_live.py` — `LiveServer` (via `serve_live_demo.py`)
| Method | Path | Purpose |
|---|---|---|
| GET | `/` (or `/index.html`) | Serve the live-viewer HTML page. |
| GET | `/state` | Current frame as JSON (board, heads, win probs). |
| POST | `/reset` | Start a fresh game. |

### `viz_interactive.py` — `InteractiveServer` (via `play_interactive.py`)
| Method | Path | Purpose |
|---|---|---|
| GET | `/` (or `/index.html`) | Serve the interactive UI. |
| GET | `/state` | Current game state JSON. |
| GET | `/agents` | Available agent presets (`AGENT_PRESETS`). |
| POST | `/start` | Begin a game with a chosen agent/config. |
| POST | `/action` | Submit a human move (direction `0..3`). |

## Related docs
- Flows that chain these together: [`docs/01-product/USER_FLOWS.md`](../01-product/USER_FLOWS.md).
- Reproducibility commands: root `README.md` "Reproducibility" + `docs/baseline_report*.md`.
- Visual artifacts produced: [`docs/08-visuals/SCREENSHOT_MANIFEST.md`](../08-visuals/SCREENSHOT_MANIFEST.md).
