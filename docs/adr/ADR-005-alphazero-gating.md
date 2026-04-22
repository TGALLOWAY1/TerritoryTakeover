# ADR-005: Promote-latest snapshot (no gating tournament) for AlphaZero

**Status:** Accepted (for Phase 3c scope); revisit motivated by PR 2
**Date:** 2026-04-22

## Context

Canonical AlphaZero runs a *gating tournament* every N training
iterations: the candidate network plays a fixed head-to-head against
the current champion, and the candidate is promoted only if its win
rate exceeds a threshold (classically 55%). Gating prevents a single
bad training cycle from poisoning self-play and gives a clean signal
that "the model is actually improving."

Phase 3c was scoped to finish in a compressed timeline on a single
box. A full gating tournament per iteration would add ~2–5× of
evaluation overhead on top of self-play, which was unacceptable
against the Phase-3c deadline. The curriculum (Phase 3d) adds stage
promotion gates (`promotion.max_self_play_steps`,
`elo_gain_threshold`) that provide a *coarser* form of gating between
stages — but not between individual snapshots within a stage.

## Decision

- No per-iteration gating tournament. The latest snapshot always
  becomes the self-play champion.
- `rl/alphazero/train.py` documents the stub explicitly in its module
  docstring (lines 12–14) and keeps an in-code TODO at the promotion
  point (line 207) referencing this ADR.
- Curriculum training stays in lockstep: stage-level promotion is the
  only gating signal.

## Consequences

- **Training throughput preserved.** Phase 3c finished on time.
- **Noisier training.** A bad update can contaminate self-play until
  the next stage gate (in curriculum) or the end of the run
  (in `direct` arm).
- **Reference checkpoint reproducible.** `docs/phase3d/net_reference.pt`
  was produced under this policy; fixing it now would invalidate that
  reproducibility.
- **Subsystem marked Experimental in README.** The `What's in the box`
  table flags AlphaZero as "Experimental — gating stubbed" with a link
  to this ADR.

## Motivated follow-up

PR 2's 20×20 hypothesis study
(`docs/experiments/20x20_hypothesis_test.md`) found that the
curriculum reference's *value head* is the weakest component of the
pipeline at out-of-distribution board sizes: more eval-time PUCT
compute did not raise win rate and in aggregate moved it slightly in
the wrong direction. That result turns "finish the gating tournament"
from a generic to-do into a directed investment:

- Gate on **value-head accuracy** against a held-out evaluation set
  (not just head-to-head win rate), since the diagnosis is that the
  policy head already generalizes and the value head does not.
- Make the gating threshold / sample size visible in config so
  ablations can quantify the value-head vs. search-compute trade-off.

## References

- `src/territory_takeover/rl/alphazero/train.py` — docstring + `:207`
  TODO.
- `docs/experiments/20x20_hypothesis_test.md` — H(a) supported,
  implications for value-head quality.
- `configs/phase3d_curriculum_fast.yaml` — stage-level `promotion`
  knobs that provide coarse gating in the curriculum arm.
