# Documentation Index

> The complete map of project documentation — the structured `docs/00-08` system plus preserved
> existing docs. New here? See [`README.md`](README.md). Last audited 2026-05-28.

## Start Here
- [Project snapshot](PROJECT_SNAPSHOT.md) — stack, commands, subsystem status at a glance.
- [Product brief](../01-product/PRODUCT_BRIEF.md) — what it is, audiences, claims.
- [Current behavior](../01-product/CURRENT_BEHAVIOR.md) — what works / is deferred, with evidence.
- [Architecture](../02-architecture/ARCHITECTURE.md) — layers, runtime/data flow, boundaries.
- [Known issues](../04-quality/KNOWN_ISSUES.md) — limitations and stubs.
- [Next agent tasks](../05-planning/NEXT_AGENT_TASKS.md) — ready-to-run work prompts.

## Product Docs (`01-product/`)
- [Product brief](../01-product/PRODUCT_BRIEF.md)
- [Feature inventory](../01-product/FEATURE_INVENTORY.md) — every feature with a status label.
- [Current behavior](../01-product/CURRENT_BEHAVIOR.md)
- [User flows](../01-product/USER_FLOWS.md) — baseline report, live viewer, play, replay, train, library use.

## Architecture Docs (`02-architecture/`)
- [Architecture](../02-architecture/ARCHITECTURE.md) · [System map](../02-architecture/SYSTEM_MAP.md)
- [Data model](../02-architecture/DATA_MODEL.md) · [State & encoding](../02-architecture/STATE_AND_ENCODING.md)
- [Public API](../02-architecture/PUBLIC_API.md) · [Integrations](../02-architecture/INTEGRATIONS.md)

## Implementation Docs (`03-implementation/`)
- [Codebase inventory](../03-implementation/CODEBASE_INVENTORY.md) — module-by-module.
- [Entry points & scripts](../03-implementation/ENTRYPOINTS_AND_SCRIPTS.md) — CLIs + HTTP demo endpoints.
- [Config & environment](../03-implementation/CONFIG_AND_ENVIRONMENT.md)
- [Testing strategy](../03-implementation/TESTING_STRATEGY.md)

## Quality Docs (`04-quality/`)
- [Known issues](../04-quality/KNOWN_ISSUES.md) · [Technical debt](../04-quality/TECHNICAL_DEBT.md)
- [Risk register](../04-quality/RISK_REGISTER.md) · [Regression checklist](../04-quality/REGRESSION_CHECKLIST.md)
- [Security & privacy notes](../04-quality/SECURITY_AND_PRIVACY_NOTES.md)

## Planning Docs (`05-planning/`)
- [Backlog](../05-planning/BACKLOG.md) · [Prioritized TODO](../05-planning/PRIORITIZED_TODO.md)
- [Roadmap](../05-planning/ROADMAP.md) · [Next agent tasks](../05-planning/NEXT_AGENT_TASKS.md)

## AI Agent Docs (`07-ai-context/`)
- [Context loading protocol](../07-ai-context/CONTEXT_LOADING_PROTOCOL.md) — minimal bundles per task.
- [Agent workflow](../07-ai-context/AGENT_WORKFLOW.md) — inspect → change → verify → document → commit.

## Visual Docs (`08-visuals/`)
- [Screenshot manifest](../08-visuals/SCREENSHOT_MANIFEST.md) · [Visual regression plan](../08-visuals/VISUAL_REGRESSION_PLAN.md)
- [Flow diagrams](../08-visuals/FLOW_DIAGRAMS.md) — engine/harness/AlphaZero Mermaid flows.

## Historical Docs (`06-history/`)
- [Decision log](../06-history/DECISION_LOG.md) — synthesis + ADR index.
- [Changelog notes](../06-history/CHANGELOG_NOTES.md) · [Audit log](../06-history/AUDIT_LOG.md)
- [Archive](../06-history/archive/ARCHIVE_NOTES.md) — relocated `KEY_FINDINGS.md`, `PHASE3_SUMMARY.md`,
  `portfolio-readiness-audit/` (with supersession notes).

## Preserved existing docs (not part of `00-08`)
- ADRs: [`../adr/`](../adr/README.md) — 6 architecture decision records.
- Benchmarks: [`../baseline_report_20x20.md`](../baseline_report_20x20.md),
  [`../baseline_report.md`](../baseline_report.md),
  [`../curriculum_puct_scaling.md`](../curriculum_puct_scaling.md).
- Experiments: [`../experiments/20x20_hypothesis_test.md`](../experiments/20x20_hypothesis_test.md).
- Performance: [`../OPTIMIZATION_ANALYSIS.md`](../OPTIMIZATION_ANALYSIS.md),
  [`../optimization_analysis/`](../optimization_analysis/), `../../benchmarks/*FINDINGS.md`.
- Reference checkpoint + Phase-3a/3d artifacts: `../phase3a/`, `../phase3d/`.

## Status labels (used throughout)
| Label | Meaning |
|---|---|
| **Implemented** | Present and appears functional. |
| **Partial** | Some behavior works; important pieces missing. |
| **Stubbed** | Placeholder exists; no real behavior. |
| **Broken** | Intended behavior fails or appears unsafe. |
| **Designed only** | Mentioned in docs/plans; not found in code. |
| **Deprecated** | Superseded path/feature. |
| **Unknown** | Insufficient evidence. |

## Conventions
- Docs document **actual** behavior; inferred statements are labeled *inferred*.
- Status claims cite source paths. ADRs are immutable (supersede, don't edit).
- Keep docs in sync with code — see [`../07-ai-context/AGENT_WORKFLOW.md`](../07-ai-context/AGENT_WORKFLOW.md).
