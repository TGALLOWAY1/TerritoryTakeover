# ADR-004: Terminal vs. n-step bootstrapped value target

**Status:** Accepted (terminal as default; n-step opt-in)
**Date:** 2026-04-22

## Context

AlphaZero trains its value head on a per-state target. Two standard
choices:

- **Terminal target:** use the final game outcome `{-1, 0, +1}` for
  every state in that game. Classic AlphaZero / MuZero choice;
  unbiased but high-variance on long games.
- **N-step bootstrapped target:** use the discounted sum of `n` future
  rewards plus `V(s_{t+n})` from the current network. Lower variance,
  but introduces bootstrap bias whenever the value net is miscalibrated.

Phase 3d's curriculum training used the terminal target exclusively
and showed the curriculum arm beating the direct arm by +27pp win
rate vs. random on 10×10 (see `KEY_FINDINGS.md` Phase-3d entry). That
result was strong enough that switching defaults was not worth the
risk.

At the same time, PR 2's 20×20 hypothesis study
(`docs/experiments/20x20_hypothesis_test.md`) showed that curriculum's
out-of-distribution value head is the pipeline's weakest component
above 10×10 — more eval-time PUCT compute did *not* rescue win rate.
Whether an n-step target would materially change the value-head
quality on larger boards is the obvious next ablation.

## Decision

- Keep the terminal value target as the default.
- Add a flag-gated n-step bootstrapped target as an alternative path
  in `rl/alphazero/train.py` (commit `a1f1254`).
- Default behaviour is unchanged; n-step is opt-in via config.

## Consequences

- **Reference checkpoint stays reproducible.** Every committed
  Phase-3d artifact was trained under terminal targets; default
  behaviour continues to reproduce them.
- **N-step is available for follow-up studies.** A value-head
  quality ablation at 20×20 (motivated by the PR 2 writeup) can
  flip the flag without a branch change.
- **Trade-off accepted:** two code paths in `train.py` mean a slightly
  larger maintenance surface. Tests
  (`tests/test_rl_alphazero_nstep_target.py`) pin the bootstrap math
  so a future refactor can't silently break the opt-in path.
- **Open question:** no committed evidence yet on which target is
  better for boards above 10×10. Recorded as a follow-up in the PR 2
  writeup.

## References

- `src/territory_takeover/rl/alphazero/train.py` — both targets.
- `tests/test_rl_alphazero_nstep_target.py` — n-step math tests.
- `KEY_FINDINGS.md`, `PHASE3_SUMMARY.md` — terminal-target headline
  results.
- `docs/experiments/20x20_hypothesis_test.md` — motivation for a future
  value-head ablation.
- Commit `a1f1254` — introduces the flag-gated n-step path.
