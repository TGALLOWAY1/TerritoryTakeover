# Visual Regression Plan

> A lightweight, repeatable process for visually auditing the game/agents over time. There is no
> automated visual-diff harness; this documents how to regenerate artifacts deterministically and
> compare them by eye / by file. Last audited 2026-05-28.

## Principle
Every artifact is a pure function of `(seed, board_size, agents, iterations)`. Fix those and the
output is deterministic (engine + harness are seed-locked — see ADR-006). To check for a visual
regression: regenerate with the **same** inputs and compare to the committed file.

## Setup
```bash
pip install -e ".[viz]"     # numpy + matplotlib + pillow (torch-free path)
# AlphaZero/curriculum visuals additionally need: pip install -e ".[rl_deep]"
```

## Regenerate the committed canonical media (existing scripts)
```bash
python scripts/record_agent_gallery.py --seed 0        # docs/assets/agent_gallery.png
python scripts/render_h2h_heatmap.py                   # docs/assets/h2h_heatmap.png
python scripts/record_territory_growth.py --seed 0     # docs/assets/territory_growth.png
python scripts/record_mcts_scaling.py --seed 0         # docs/assets/mcts_scaling.png
python scripts/record_demo.py --seed 0                 # docs/assets/demo.gif  (needs rl_deep + checkpoint)
```
Most recording scripts accept `--dry-run` to validate configuration without writing.

## Regenerate the docs/08-visuals artifacts

### HTML replay
```bash
python scripts/record_html_demo.py --seed 0 --board-size 10 --num-players 2 \
  --seat0 rave --seat1 greedy --rave-iterations 60 --win-prob heuristic \
  --out docs/08-visuals/screenshots/replay_rave_vs_greedy_10x10.html
```

### Board snapshot (PNG), GIF, and ASCII (torch-free, via the library)
```python
import numpy as np, territory_takeover as tt
from territory_takeover.actions import legal_actions
from territory_takeover.viz import render_ascii, render_matplotlib, save_game_gif
from territory_takeover.search import RaveAgent, HeuristicGreedyAgent

ss = np.random.SeedSequence(0)
a_rng, g_rng = (np.random.default_rng(s) for s in ss.spawn(2))
agents = {0: RaveAgent(iterations=60, rng=a_rng, name="rave"),
          1: HeuristicGreedyAgent(rng=g_rng, name="greedy")}
s = tt.new_game(board_size=10, num_players=2, seed=0)
states = [s.copy()]
while not s.done and len(states) < 400:
    pid = s.current_player; la = legal_actions(s, pid)
    a = agents[pid].select_action(s, pid, time_budget_s=10.0, max_iterations=60) if la else 0
    s = tt.step(s, int(a)).state; states.append(s.copy())

render_matplotlib(states[len(states)*2//3]).get_figure().savefig(
    "docs/08-visuals/screenshots/board_midgame_10x10.png", dpi=110, bbox_inches="tight")
save_game_gif(states, "docs/08-visuals/screenshots/game_rave_vs_greedy_10x10.gif", fps=6)
open("docs/08-visuals/screenshots/sample_board_ascii.txt", "w").write(render_ascii(states[-1]))
```

## Comparing over time
- **Deterministic artifacts** (PNG/GIF/ASCII at fixed seed): regenerate and `git diff` /
  byte-compare. Note that matplotlib/pillow version bumps can change pixels even when the game is
  identical — diff the **ASCII** render for a version-independent game-logic check.
- **Game-logic regression check (most robust):** compare `render_ascii(final_state)` for a fixed
  seed; any change means the engine/agent behavior changed, independent of rendering.
- **Reports:** for tournament tables, regenerate `docs/baseline_report*.md` and diff the numbers
  (see [`docs/04-quality/REGRESSION_CHECKLIST.md`](../04-quality/REGRESSION_CHECKLIST.md)).

## When to refresh artifacts
- After engine/enclosure changes (board appearance may change) → regenerate ASCII + PNG.
- After agent/search changes (play differs) → regenerate GIF/HTML and update the manifest date.
- Update [`SCREENSHOT_MANIFEST.md`](SCREENSHOT_MANIFEST.md) "Last captured" whenever you regenerate.
