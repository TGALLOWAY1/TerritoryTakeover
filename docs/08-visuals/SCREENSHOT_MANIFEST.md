# Screenshot Manifest

> Inventory of visual artifacts: newly generated ones under `docs/08-visuals/screenshots/` and the
> pre-existing committed media under `docs/assets/`. All are reproducible from a seed.
> Last captured: 2026-05-28.

## Newly generated (`docs/08-visuals/screenshots/`)

### Mid-game board snapshot
- **What:** Single matplotlib render of a 10×10 RAVE-vs-Greedy game at turn 22 — shows blue/red
  paths, a claimed (light-pink) enclosed pocket, and boxed heads.
- **Viewport / size:** 533×556 PNG.
- **State:** mid-game (~2/3 through), seed 0.
- **Path:** `docs/08-visuals/screenshots/board_midgame_10x10.png`
- **How to reproduce:** see [`VISUAL_REGRESSION_PLAN.md`](VISUAL_REGRESSION_PLAN.md) §"Board snapshot".
- **Known visual issues:** none.

### Full-game animated replay (GIF)
- **What:** 33-frame animation of the same RAVE@60-vs-Greedy game (seat 0 RAVE wins).
- **Size:** 330×353 GIF, 6 fps.
- **Path:** `docs/08-visuals/screenshots/game_rave_vs_greedy_10x10.gif`
- **How to reproduce:** `VISUAL_REGRESSION_PLAN.md` §"GIF".

### Self-contained HTML replay
- **What:** Interactive single-file replay (embedded CSS/JS, frame stepping, heuristic win-prob bars).
- **Path:** `docs/08-visuals/screenshots/replay_rave_vs_greedy_10x10.html`
- **How to reproduce:** `python scripts/record_html_demo.py --seed 0 --board-size 10 --num-players 2
  --seat0 rave --seat1 greedy --rave-iterations 60 --win-prob heuristic --out <path>`.

### ASCII board render
- **What:** Text render (turn 8 + final) demonstrating the `render_ascii` surface.
- **Path:** `docs/08-visuals/screenshots/sample_board_ascii.txt`
- **How to reproduce:** `VISUAL_REGRESSION_PLAN.md` §"ASCII".

## Pre-existing committed media (`docs/assets/`)
These were produced by the repo's recording scripts and are referenced from the root README.

| Artifact | What | Reproduce with |
|---|---|---|
| `docs/assets/demo.gif` | RAVE@200 vs curriculum_ref @4, 20×20, seed 0 | `scripts/record_demo.py --seed 0` (needs torch for the AZ side) |
| `docs/assets/agent_gallery.png` | 4-panel Random/Greedy/UCT/RAVE self-play at turn 100 | `scripts/record_agent_gallery.py --seed 0` |
| `docs/assets/h2h_heatmap.png` | 20×20 head-to-head win-rate heatmap | `scripts/render_h2h_heatmap.py` |
| `docs/assets/territory_growth.png` | RAVE@200 vs Greedy territory growth, 20×20 | `scripts/record_territory_growth.py --seed 0` |
| `docs/assets/mcts_scaling.png` | UCT-vs-random win rate vs sims/move, 10×10 | `scripts/record_mcts_scaling.py --seed 0` |
| `docs/assets/best_agent_demo.html` | bundled interactive HTML demo | `scripts/record_html_demo.py` |
| `docs/phase3a/*.png` | Phase-3a tabular-Q game-state screenshots | training/eval scripts (Phase 3a) |

## Notes
- The newly generated artifacts use **torch-free** classical agents (RAVE/Greedy) so they
  reproduce with just `pip install -e ".[viz]"`. AlphaZero/curriculum visuals (`demo.gif`) need
  the `rl_deep` extra and the reference checkpoint.
- Update the **Last captured** date and the relevant row when regenerating.

## Related docs
- Reproduce/diff process: [`VISUAL_REGRESSION_PLAN.md`](VISUAL_REGRESSION_PLAN.md)
- Architecture/flow diagrams: [`FLOW_DIAGRAMS.md`](FLOW_DIAGRAMS.md)
