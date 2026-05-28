# Archive Notes

> This folder holds historical/narrative documents relocated from the repo top level during the
> documentation-infrastructure pass (2026-05-28). They are preserved verbatim (only relative
> image paths were adjusted) for provenance, not actively maintained. For current information,
> use the live docs under `docs/00-overview` … `docs/08-visuals`.

## What moved here and why

| Was (top level) | Now | Why archived |
|---|---|---|
| `KEY_FINDINGS.md` | `docs/06-history/archive/KEY_FINDINGS.md` | Running lab notebook (Phase 3a/3c/3d). Historical narrative; superseded for navigation by the structured docs + ADRs. Image links rewritten (`docs/phase3a/` → `../../phase3a/`). |
| `PHASE3_SUMMARY.md` | `docs/06-history/archive/PHASE3_SUMMARY.md` | Cross-phase synthesis + deferrals. Same rationale. |
| `portfolio-readiness-audit/` | `docs/06-history/archive/portfolio-readiness-audit/` | A 6-file portfolio audit that **predates the current README**. |

## Important: the portfolio-readiness-audit is partly superseded

The portfolio-readiness-audit was written when the top-level README was stale. Its central
recommendation — "rewrite the README around evidence/outcomes" and several "must-have"
items (canonical baselines, demo GIF, decision notes, architecture diagram) — **have since been
completed**. Specifically, these claims in that audit are **no longer accurate**:

- "The top-level README is stale and directly contradicts the actual implementation depth."
  → The current `README.md` is polished and evidence-led (headline leaderboard, demo media,
  subsystem status table, known limitations).
- "No single proof surface / fragmented evidence."
  → Baselines (`docs/baseline_report*.md`), ADRs (`docs/adr/`), and interactive viewers exist.

Its still-valid forward-looking items (AlphaZero gating, PPO driver, perf CI, value-head scaling)
were extracted into the live planning docs:
[`docs/05-planning/BACKLOG.md`](../../05-planning/BACKLOG.md) and
[`docs/05-planning/NEXT_AGENT_TASKS.md`](../../05-planning/NEXT_AGENT_TASKS.md).

## Plain-text references elsewhere
Some committed ADRs and `docs/optimization_analysis/` notes mention `KEY_FINDINGS.md` /
`PHASE3_SUMMARY.md` as plain inline-code text (not clickable links). Those were intentionally
**not edited** — ADRs are immutable once committed (superseded, not edited). When you see such a
mention, the file now lives in this archive folder.

## Related
- Decision history: [`../DECISION_LOG.md`](../DECISION_LOG.md)
- Change history: [`../CHANGELOG_NOTES.md`](../CHANGELOG_NOTES.md)
- Audit history: [`../AUDIT_LOG.md`](../AUDIT_LOG.md)
