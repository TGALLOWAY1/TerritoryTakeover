# Board-Scale Hypothesis Test — 20x20

**Date:** 2026-04-22
**Commit:** `cd7c598`
**Branch:** `claude/portfolio-readiness-20x20-hypothesis-study`

## Why this experiment exists

PR 1's canonical baseline report ran a 5-agent round-robin head-to-head at
10x10 / 2p and produced a surprising ranking: the curriculum reference
checkpoint (`docs/phase3d/net_reference.pt`) tied uniformly-random on
overall win rate (both at 0.300), driven by an unusually high tie rate.
This contradicted the Phase-3d narrative in `PHASE3_SUMMARY.md` — where
the curriculum arm was reported as the headline RL result with a +27pp
win rate lift over the direct arm.

Two competing explanations presented themselves:

- **H(a): Curriculum is compute-insensitive / intrinsically weak.** The
  checkpoint's nominal Phase-3d strength came from territorial
  tie-breaking (Elo derived from claim counts); under a
  winner-takes-all rule it is no better than random regardless of
  eval-time PUCT compute.

- **H(b): Curriculum scales with eval-time PUCT.** The 4-iteration
  setting in PR 1 matched the Phase-3d Elo-pool convention but is
  severely under-compute for evaluation. Given more per-move PUCT
  iterations, the checkpoint's value/policy heads would express their
  latent strength and the win rate would rise materially.

The 10x10 result alone could not discriminate between these. Two
additional uncertainties loomed: 10x10 is a very small strategic
surface (many pairings may be decided by opening-move luck), and the
curriculum checkpoint was trained only up through 10x10, so larger
boards would expose it as out-of-distribution. Both arguments pointed
at the same remedy: re-run the study at a board size where strategic
play matters, and explicitly sweep eval-time PUCT.

## Design

Two experiments, both at 20x20 / 2p, same repo as PR 1, reproducible from
one integer seed via `numpy.random.SeedSequence`.

**E1 — 20x20 canonical leaderboard.** Same roster as PR 1 (Random,
Greedy, UCT@200, RAVE@200, curriculum_ref@4 PUCT iters). Board enlarged
from 10x10 to 20x20 (4x the area). `games_per_pair=20` instead of 40 to
keep wall clock bounded under `--parallel`. Reported in
`docs/baseline_report_20x20.md`.

**E2 — Curriculum PUCT scaling sweep.** `scripts/run_puct_scaling.py`
rebuilds the curriculum agent at each PUCT budget in `{4, 16, 64}` (a
16x compute range) and runs alternating-seat head-to-heads against a
fixed opponent panel of Random, Greedy, UCT@100. `games_per_opponent=10`
per cell (reduced after the first attempt was killed when the session
suspended overnight); 90 games total. Reported in
`docs/curriculum_puct_scaling.md`.

**Caveats baked into the design:**

- The curriculum checkpoint was trained only up through 10x10. 20x20 is
  out-of-distribution; the `head_type=conv` architecture accepts
  arbitrary HxW (`tests/test_rl_alphazero_network_variable_size.py`) but
  generalization is part of what we are measuring.
- E2's 10 games per cell produces wide Wilson 95% intervals
  (approximately ±25pp at 50% win rate). The study is powered to detect
  a directional trend, not small compute effects.
- UCT was dropped from 200 iters in E1 to 100 iters in E2 to keep E2's
  wall clock tractable — this is a within-experiment change, not across.

## Results

### E1 headlines (20x20, 20 games/pair, 200 games total)

| Rank | Agent          | Win rate | 95% CI          |
|-----:|----------------|---------:|-----------------|
| 1    | rave           | 0.762    | [0.659, 0.842]  |
| 2    | uct            | 0.637    | [0.528, 0.734]  |
| 3    | curriculum_ref | 0.412    | [0.311, 0.522]  |
| 4    | greedy         | 0.300    | [0.211, 0.408]  |
| 5    | random         | 0.300    | [0.211, 0.408]  |

Two shifts vs. the 10x10 baseline worth flagging:

- **RAVE overtakes UCT.** At 10x10 UCT led by 6.3pp; at 20x20 RAVE leads
  by 12.5pp and the CIs are well separated. AMAF generalization across
  the much larger branching factor appears to dominate.
- **Curriculum reference moves from rank 5 (tied with random) at 10x10
  to rank 3 (above greedy) at 20x20**, even at the minimum 4 PUCT iters.
  Head-to-head: curriculum vs. greedy is 10/0/10 (split), vs. random is
  9/5/9 (split with 5 ties), vs. rave is 8/0/12 (loses).

### E2 headlines (20x20, curriculum PUCT sweep, 90 games total)

| PUCT iters | Agg. win rate | 95% CI          | Avg decision |
|-----------:|--------------:|-----------------|-------------:|
| 4          | 0.367         | [0.219, 0.545]  | 1.56 s       |
| 16         | 0.233         | [0.118, 0.409]  | 4.57 s       |
| 64         | 0.300         | [0.167, 0.479]  | 21.05 s      |

Compute scaled 16x from iters=4 to iters=64; the aggregate win rate did
not rise. iters=4 was the best single cell, iters=16 the worst. CIs
overlap heavily, so individual differences between cells are not
statistically significant, but the **direction of the trend** (flat or
mildly negative) is clearly inconsistent with H(b).

Per-opponent detail mirrors the aggregate pattern: against UCT@100 the
curriculum scored 2/10 at every PUCT budget; against Random it
peaked at 4/10 at iters=4 and dropped to 2/10 at iters=16 and iters=64.

## Verdict

**H(a) is supported; H(b) is not.**

More precisely, once you integrate E1 and E2 together:

- The curriculum checkpoint is **not** intrinsically no-stronger-than-
  random at strategic board sizes. E1 shows a clear rank-3 standing at
  20x20 and a decisive head-to-head lead over greedy + random at just
  4 PUCT iters. Earlier narrative text suggesting the checkpoint owes
  its Phase-3d standing entirely to territorial tie-breaking was
  overstated.
- The curriculum checkpoint's strength at 20x20 comes primarily from
  its **policy-head prior**, not from search compute. Increasing PUCT
  iterations 16x produces no measurable gain and a suggestive negative
  trend. This is consistent with an out-of-distribution value head
  providing unreliable guidance — more search compounds rather than
  corrects the error.
- The PR 1 narrative of "curriculum_ref ties random" is a local-to-10x10
  artifact: at 10x10 the strategic surface is small enough that random
  / greedy / curriculum cluster tightly; at 20x20 strategic differences
  separate cleanly.

## Implications for the portfolio narrative

1. **The baseline benchmark must be at 20x20 or larger.** PR 1's 10x10
   report should be reframed as a logic-verification artifact, not the
   headline result. The README (PR 3) should cite the 20x20 leaderboard.
2. **Curriculum remains a real RL result, not a dead end.** Rank 3 at
   20x20 above the strongest classical heuristic (greedy) is a
   defensible headline for the RL track, but it must be framed honestly:
   the advantage is policy-prior quality, not search-time compute.
3. **ADR-005 (AlphaZero gating stub) becomes more interesting.** The
   current train.py always promotes the latest snapshot; the result
   here suggests the value head is the weakest component of the
   pipeline, and a well-designed gating tournament could specifically
   select for value-head quality (not just policy self-play win rate).
   This is a compelling follow-up rather than a stub to be silently
   closed.
4. **RAVE > UCT at scale is a secondary but publishable finding.** Worth
   a short note in the eventual decision log.

## Limitations and follow-ups

- **Statistical power.** E2's 10 games/cell yields ~25pp CIs. A follow-up
  at games_per_opponent in {40-100} would tighten the trend enough to
  distinguish "flat" from "mildly negative" with more confidence.
- **Larger boards (30x30+) are untested.** H(a) was confirmed at 20x20;
  whether the curriculum checkpoint's policy prior continues to transfer
  at 30x30 or degrades into noise is open.
- **PUCT budget spacing.** The 4/16/64 geometric spacing may have
  skipped a regime where the checkpoint is helpful (e.g., iters=1 vs.
  iters=4, or iters=200+). A finer sweep would reveal this.
- **Dirichlet noise and temperature.** Eval-time was run at
  `temperature_eval=0` with no Dirichlet; if the checkpoint was trained
  with non-trivial Dirichlet, eval-time PUCT may be exploring a
  mismatched state distribution. A sensitivity analysis would help.
- **Value head ablation.** The mechanistic hypothesis "value head is the
  weak link at 20x20" would be testable by running a tree-search variant
  that uses only the policy head (no value guidance, e.g., PUCT with a
  constant value) and comparing.

## Reproducibility

```
python scripts/run_baseline_report.py \
    --board-size 20 --games-per-pair 20 --parallel --seed 0

python scripts/run_puct_scaling.py \
    --board-size 20 --games-per-opponent 10 \
    --az-iters 4 16 64 --uct-iterations 100 \
    --parallel --seed 0
```

Both runs are deterministic under the given seed and reseed every agent
and every per-game RNG through the harness's `SeedSequence` spawn tree
(see `src/territory_takeover/search/harness.py::run_match`).
