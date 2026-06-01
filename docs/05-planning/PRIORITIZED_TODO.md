# Prioritized TODO

> The backlog as an ordered, do-next list. Full detail per item in [`BACKLOG.md`](BACKLOG.md).
> Last audited 2026-05-28.

## Now (highest leverage)
1. **B1 — AlphaZero gating tournament.** Removes the only stub; biggest credibility win.
   `rl/alphazero/train.py:207-210`.
2. **B3 — Perf regression guard.** Protects the central "fast engine" claim. `benchmarks/`, CI.
3. **B5 — Run metadata traceability.** Cheap; strengthens reproducibility. Report writers.

## Next
4. **B2 — PPO training driver.** Completes the third RL track. `scripts/train_ppo.py`.
5. **B6 — Make ruff/mypy required in CI.** Cheap quality gate once clean.

## Later
6. **B4 — Curriculum value-head / 20×20 scaling.** Research-y; higher complexity.
7. **B7 — Reduce `Any` usage.** Incremental typing polish.

## Sequencing notes
- B1, B3, B5 are independent and can proceed in parallel.
- B6 should follow a clean `ruff check .` / `mypy` pass (do B7 partially first if needed).
- B4 depends on no other item but is the longest-running.

## Related docs
- Ready-to-run prompts: [`NEXT_AGENT_TASKS.md`](NEXT_AGENT_TASKS.md)
- Themes/sequencing over time: [`ROADMAP.md`](ROADMAP.md)
