# Context Loading Protocol

> Do **not** load all docs for every task. Load the smallest relevant bundle for the task type,
> then open specific source files as needed. This avoids context bloat and "lost in the middle"
> failures. Last audited 2026-05-28.

## Always start with
- `CLAUDE.md` (root) — commands, tile encoding, engine entry points, conventions.
- [`docs/00-overview/PROJECT_SNAPSHOT.md`](../00-overview/PROJECT_SNAPSHOT.md) — one-screen orientation.

## Bundles by task type

### Engine / state / actions / enclosure work
Read:
- `CLAUDE.md`
- [`docs/02-architecture/STATE_AND_ENCODING.md`](../02-architecture/STATE_AND_ENCODING.md)
- [`docs/02-architecture/DATA_MODEL.md`](../02-architecture/DATA_MODEL.md)
- [`docs/04-quality/REGRESSION_CHECKLIST.md`](../04-quality/REGRESSION_CHECKLIST.md)
- ADRs 001–003 (`docs/adr/`)
Don't read: RL docs, planning docs, history/archive.

### Classical search / MCTS work
Read:
- [`docs/02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md)
- [`docs/02-architecture/PUBLIC_API.md`](../02-architecture/PUBLIC_API.md) (Agents + Harness)
- [`docs/03-implementation/CODEBASE_INVENTORY.md`](../03-implementation/CODEBASE_INVENTORY.md) (search section)
- [`docs/04-quality/REGRESSION_CHECKLIST.md`](../04-quality/REGRESSION_CHECKLIST.md)
Don't read: viz docs, product brief, archive.

### RL / training work (tabular / PPO / AlphaZero / curriculum)
Read:
- [`docs/02-architecture/STATE_AND_ENCODING.md`](../02-architecture/STATE_AND_ENCODING.md) (obs encoders)
- [`docs/01-product/FEATURE_INVENTORY.md`](../01-product/FEATURE_INVENTORY.md) (RL status)
- [`docs/04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md)
- the relevant ADR (004 value target, 005 gating) + the matching `docs/experiments/` writeup
Don't read: unrelated RL tracks, viz docs.

### Visualization / demo work
Read:
- [`docs/03-implementation/ENTRYPOINTS_AND_SCRIPTS.md`](../03-implementation/ENTRYPOINTS_AND_SCRIPTS.md)
- [`docs/01-product/USER_FLOWS.md`](../01-product/USER_FLOWS.md)
- [`docs/08-visuals/SCREENSHOT_MANIFEST.md`](../08-visuals/SCREENSHOT_MANIFEST.md) + [`VISUAL_REGRESSION_PLAN.md`](../08-visuals/VISUAL_REGRESSION_PLAN.md)
- [`docs/04-quality/SECURITY_AND_PRIVACY_NOTES.md`](../04-quality/SECURITY_AND_PRIVACY_NOTES.md) (HTTP servers)
Don't read: RL internals, decision log.

### Bug-fix work
Read:
- [`docs/04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md)
- [`docs/04-quality/REGRESSION_CHECKLIST.md`](../04-quality/REGRESSION_CHECKLIST.md)
- only the feature/architecture/API doc for the affected area
Don't read: planning docs, history/archive, unrelated subsystems.

### Benchmark / reproducibility work
Read:
- [`docs/02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md) (data-flow + reproducibility)
- [`docs/03-implementation/ENTRYPOINTS_AND_SCRIPTS.md`](../03-implementation/ENTRYPOINTS_AND_SCRIPTS.md)
- `docs/baseline_report*.md`, ADR-006
Don't read: viz internals, product brief.

### Documentation work
Read:
- [`docs/00-overview/DOCUMENTATION_INDEX.md`](../00-overview/DOCUMENTATION_INDEX.md)
- [`docs/06-history/AUDIT_LOG.md`](../06-history/AUDIT_LOG.md)
- [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md)
Don't read: deep source unless a specific claim must be verified.

## Picking up a planned task
For backlog items, read the matching prompt in
[`docs/05-planning/NEXT_AGENT_TASKS.md`](../05-planning/NEXT_AGENT_TASKS.md) — it already lists the
exact context bundle and files to edit.

## Anti-patterns
- Don't load the whole `docs/06-history/archive/` (KEY_FINDINGS / PHASE3_SUMMARY / portfolio
  audit) for routine work — it's historical narrative.
- Don't read every ADR; read the one(s) for your subsystem.
- Don't read all RL tracks when touching one.
