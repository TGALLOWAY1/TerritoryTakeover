# Architecture Decision Records

Short, one-page-ish records of architecturally significant decisions.
Each follows a context / decision / consequences template. New ADRs
are added at the next number; existing ADRs are not edited in place
once committed (supersede with a new ADR).

| # | Title | Scope |
|--:|-------|-------|
| 001 | [Int8 grid encoding with dual PATH/CLAIMED codes](ADR-001-int8-grid-encoding.md) | Engine representation |
| 002 | [State split — grid + redundant per-player caches](ADR-002-state-split.md) | Engine representation |
| 003 | [Enclosure detection — trigger check + iterative BFS](ADR-003-enclosure-bfs.md) | Engine perf |
| 004 | [Terminal value target by default, n-step opt-in](ADR-004-value-target.md) | RL training |
| 005 | [Promote-latest AlphaZero snapshot; gating deferred](ADR-005-alphazero-gating.md) | RL training |
| 006 | [SeedSequence + seat rotation for reproducibility](ADR-006-seedsequence-reproducibility.md) | Harness |

## Why this directory exists

Design rationale for this project was previously distributed across
`KEY_FINDINGS.md`, `PHASE3_SUMMARY.md`, the `docs/optimization_analysis/`
notes, and in-code docstrings. That's fine for an active lab notebook
but hard to consume as a portfolio artifact — a reader has to
reconstruct the reasoning chain from scattered evidence.

The ADRs here centralize the *why* for decisions that are still live
in the codebase today. They reference the narrative docs for
experimental context rather than duplicating them.
