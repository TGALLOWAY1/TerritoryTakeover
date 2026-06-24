# Deploying to Vercel

Territory Takeover ships to Vercel as a **static replay site** (Phase 1).

## Why static?

The interactive *Arena* (`territory_takeover.viz_interactive`) is a **stateful**
server: it runs the game loop in a background thread and holds game state in
memory while clients poll `/state`. Vercel is **serverless** — every request is
an isolated, ephemeral function with no persistent threads or shared in-memory
state — so the live Arena cannot run on it unchanged.

Instead, Phase 1 serves pre-recorded, self-contained HTML replays. Each replay
embeds a full game trajectory plus a small JS player (scrub, play/pause,
per-frame win-probability bars) and has **no external assets or backend**, which
makes it ideal for static hosting.

## Layout

```
public/
  index.html          landing page / replay gallery
  games/*.html        self-contained replays (committed, served as-is)
vercel.json           static config: outputDirectory=public, no build step
```

`vercel.json` sets `framework: null` and `buildCommand: null`, so Vercel
performs **no build** — it just uploads and serves `public/`.

## Deploy

Connect the repo in the Vercel dashboard (or `vercel` CLI). No environment
variables, install step, or build command are required. The production output
directory is `public/`.

## Regenerating replays

The replays are committed so the deploy needs no Python at build time. To
regenerate or extend the gallery locally:

```
pip install -e ".[dev]"
python scripts/build_site.py          # all matchups
python scripts/build_site.py --quick  # faster, lighter search
```

Edit the `MATCHES` table in `scripts/build_site.py` to add a matchup, then add a
matching card to `public/index.html`.

## Phase 2 (planned): serverless interactive Arena

To bring live agent-vs-agent play to Vercel, the Arena endpoints would be
re-implemented as stateless Python functions under `api/`: because the engine is
deterministic and copies cheaply, each request can recompute board state from a
seed + move history rather than relying on a persistent process. MCTS latency
runs per request, so search budgets would need tuning for serverless timeouts.
