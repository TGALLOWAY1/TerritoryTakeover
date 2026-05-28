# Prioritized Roadmap

## A) Must-Have for Portfolio Readiness (Minimum Viable Showcase)

1. **Rewrite top-level README around evidence and outcomes.**
   - Include: project thesis, architecture sketch, quickstart, benchmark command, headline results table, and known limitations.
   - Remove stale “not wired up” language and replace with current capabilities.

2. **Define one canonical benchmark suite and publish fixed baseline results.**
   - Required metrics (minimum): win rate, tie rate, average decision time, turns to first enclosure.
   - Required baselines: Random, Greedy, UCT, best RL checkpoint.
   - Pin seeds/configs and produce one `results/baseline_report.md`.

3. **Create one recruiter-friendly demo path.**
   - A short replay GIF/video + concise explanation of what the viewer is seeing.
   - Show at least one interesting strategy difference (e.g., random vs greedy vs trained agent).

4. **Add reproducibility contract.**
   - A single command sequence to reproduce benchmark table from clean checkout.
   - Include environment versions, expected runtime, output file paths.

5. **Publish an “engineering decisions” note.**
   - Why enclosure BFS was optimized this way.
   - Why curriculum was chosen.
   - Why current value-target design changed and what tradeoffs remain.

---

## B) High-Leverage Enhancements (Big Recruiter Upside)

1. **Simulation Inspector / Replay Viewer (lightweight web or notebook UI).**
   - Step-through timeline, per-turn territory counts, action overlays.
   - This dramatically improves demo value and comprehension.

2. **Controlled experiment dashboard artifact.**
   - Static HTML/markdown report with plots for:
     - win-rate trajectories
     - value loss/policy loss
     - enclosure timing distribution
     - decision latency by agent type

3. **Agent comparison harness hardening.**
   - Standardize experiment schema and outputs (JSON/CSV contract).
   - Add run metadata hash (config + git SHA + seed) for traceability.

4. **Finalize/de-risk deferred training flow pieces.**
   - Replace stubs (e.g., gating policy) where they affect credibility.
   - Add one “productionized” training path that is clearly complete.

5. **Performance story with before/after evidence.**
   - A concise table showing optimization impact (e.g., enclosure routine, rollout throughput).
   - Keep this measurable and reproducible.

---

## C) Nice-to-Have Polish

1. **Architecture diagram (clean and minimal).**
2. **Public project page (GitHub Pages/Notion) linking to demo + report.**
3. **Short narrated video walkthrough (2–4 minutes).**
4. **Automated benchmark CI job on reduced settings.**
5. **Ablation write-up for one key design choice (e.g., terminal vs n-step target).**

---

## Practical 4-Week Execution Plan

### Week 1 — Narrative + baseline proof
- README rewrite
- canonical benchmark run
- baseline results table and reproducibility instructions

### Week 2 — Demo surface
- replay/inspector output
- screenshots/GIFs
- “how to interpret this run” section

### Week 3 — Experiment credibility
- one polished experiment report with limitations and follow-ups
- benchmark schema + metadata traceability

### Week 4 — Recruiter polish
- architecture diagram
- concise project page
- final trimming for coherence

