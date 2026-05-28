# Agent Workflow

> How an AI agent (or human) should work in this repo so docs stay trustworthy and changes stay
> safe. Last audited 2026-05-28.

## The loop: inspect → change → verify → document → commit

### 1. Inspect before coding
- Load the minimal context bundle for your task
  ([`CONTEXT_LOADING_PROTOCOL.md`](CONTEXT_LOADING_PROTOCOL.md)).
- Check `git status` first. If there are uncommitted changes you didn't make, **do not overwrite
  them** — report what's dirty and stop if unsafe.
- Reuse existing utilities (`engine`, `eval`, `search/registry`, `viz`) before writing new code.

### 2. Make the change
- Don't modify `src/` product logic for doc-only tasks.
- Follow conventions in `CLAUDE.md`: full type annotations (mypy strict), ruff rules, function-style
  tests (no `parametrize`/fixtures), extend `GameState`/`PlayerState` in place (cheap-copy depends
  on it), keep state caches in lockstep with the grid.

### 3. Verify
- Run the [`REGRESSION_CHECKLIST`](../04-quality/REGRESSION_CHECKLIST.md): `pytest`, `ruff check .`,
  `mypy`, plus the area-specific checks. Perf-sensitive changes: spot-check `benchmarks/`.

### 4. Update documentation (same change, same PR)
- If behavior or status changed, update the matching doc **and its status label**:
  - capability/status → [`docs/01-product/FEATURE_INVENTORY.md`](../01-product/FEATURE_INVENTORY.md), [`CURRENT_BEHAVIOR.md`](../01-product/CURRENT_BEHAVIOR.md)
  - new limitation/fix → [`docs/04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md)
  - new API/module → [`docs/02-architecture/PUBLIC_API.md`](../02-architecture/PUBLIC_API.md), [`docs/03-implementation/CODEBASE_INVENTORY.md`](../03-implementation/CODEBASE_INVENTORY.md)
  - new script/endpoint → [`docs/03-implementation/ENTRYPOINTS_AND_SCRIPTS.md`](../03-implementation/ENTRYPOINTS_AND_SCRIPTS.md)
- Use the **exact status labels** (below). Never mark planned/stubbed work as Implemented.
- Cite evidence (source path) for status claims; mark guesses *inferred*.

### 5. Log decisions & audits
- Architecturally significant choice → add a new ADR in `docs/adr/` (never edit a committed ADR;
  supersede it) and a [`DECISION_LOG`](../06-history/DECISION_LOG.md) entry.
- A documentation/audit pass → append an [`AUDIT_LOG`](../06-history/AUDIT_LOG.md) entry.

### 6. Commit
- Small, focused commits; conventional-style messages (`docs:`, `rl(alphazero):`, `ci/bench:`).
- Don't bundle unrelated changes. Don't commit `results/` run artifacts. Don't skip hooks.

## Status labels (use these exact words)
`Implemented` · `Partial` · `Stubbed` · `Broken` · `Designed only` · `Deprecated` · `Unknown`.
Definitions in [`docs/00-overview/DOCUMENTATION_INDEX.md`](../00-overview/DOCUMENTATION_INDEX.md).

## Updating visuals
- Regenerate artifacts with the existing scripts and the documented commands
  ([`docs/08-visuals/VISUAL_REGRESSION_PLAN.md`](../08-visuals/VISUAL_REGRESSION_PLAN.md)), then
  update [`SCREENSHOT_MANIFEST.md`](../08-visuals/SCREENSHOT_MANIFEST.md) (path + last-captured).

## Avoiding context bloat
- Load bundles, not the whole `docs/` tree. Don't read the `06-history/archive/` for routine work.
- Prefer `PUBLIC_API.md` / `CODEBASE_INVENTORY.md` summaries over reading every source file.

## Protecting user work
- Never run destructive git (`reset --hard`, force-push, `checkout .`) without explicit request.
- Push only to the working branch you were told to use.
