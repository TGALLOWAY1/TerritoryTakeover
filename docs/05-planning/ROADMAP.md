# Roadmap

> Thematic direction, not dated commitments. The "minimum viable showcase" milestone
> (README/baselines/demo/decisions) is **already met**; what remains is depth and rigor.
> Last audited 2026-05-28.

## Where the project is
A mature, presented decision-systems lab: polished README with headline leaderboard,
reproducible baselines (10×10 + 20×20), demo media, ADRs, and interactive/live viewers. The
core engine and classical search are Production; the RL stack is the growth area.

## Theme 1 — Finish the RL pipeline (credibility)
- **B1** AlphaZero gating tournament (remove the last stub).
- **B2** PPO training driver + result (make all three RL tracks runnable end-to-end).
- Outcome: every advertised RL track trains and benchmarks without caveats.

## Theme 2 — Lock in the performance story (reliability)
- **B3** perf regression guard / scheduled benchmark CI.
- **B5** run-metadata traceability (config + git SHA + seed) in every report.
- Outcome: the "fast, reproducible engine" claim is continuously protected, not just asserted.

## Theme 3 — Push RL research depth (differentiation)
- **B4** improve curriculum value-head accuracy / scale to 20×20 (the flagged weakest link).
- A follow-up ablation write-up (e.g. terminal vs. n-step target — ADR-004) once gating exists.
- Outcome: a genuine learning-systems result, not just infrastructure.

## Theme 4 — Quality gates (hygiene)
- **B6** promote ruff/mypy to required in CI.
- **B7** reduce `Any` usage where cheap.
- Outcome: the strict-typing discipline is enforced, not optional.

## Sequencing
Themes 1 and 2 first (highest leverage, lowest risk), then Theme 3 (longest), with Theme 4 as
ongoing hygiene. See [`PRIORITIZED_TODO.md`](PRIORITIZED_TODO.md) and [`BACKLOG.md`](BACKLOG.md).
