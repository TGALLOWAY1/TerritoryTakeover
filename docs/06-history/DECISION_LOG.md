# Decision Log

> Index and synthesis of architecturally significant decisions. The canonical, immutable records
> are the ADRs in [`docs/adr/`](../adr/README.md) (never edited once committed — superseded by a
> new ADR). This log summarizes them and records decisions made outside the ADR process.
> Last audited 2026-05-28.

## ADR index (canonical source: `docs/adr/`)

| # | Decision | Scope | Status |
|--:|---|---|---|
| [001](../adr/ADR-001-int8-grid-encoding.md) | Int8 grid with dual PATH/CLAIMED codes | Engine representation | Accepted |
| [002](../adr/ADR-002-state-split.md) | State split — grid + redundant per-player caches | Engine representation | Accepted |
| [003](../adr/ADR-003-enclosure-bfs.md) | Enclosure detection — trigger check + iterative BFS | Engine perf | Accepted |
| [004](../adr/ADR-004-value-target.md) | Terminal value target by default, n-step opt-in | RL training | Accepted |
| [005](../adr/ADR-005-alphazero-gating.md) | Promote-latest AlphaZero snapshot; gating deferred | RL training | Accepted (deferral) |
| [006](../adr/ADR-006-seedsequence-reproducibility.md) | SeedSequence + seat rotation for reproducibility | Harness | Accepted |

**One-line rationale each:** (001) one int8 array keeps state memcpy-cheap and dense for the net;
(002) caches like `path_set`/`claimed_count` buy O(1) ops while the grid stays canonical;
(003) a ~4-lookup trigger then bounded BFS hits the < 200 µs target without recursion;
(004) terminal target is simplest/most stable, n-step is opt-in; (005) gating is deliberately
deferred — latest snapshot is champion (the one active limitation); (006) one root seed via
`SeedSequence` makes serial and multiprocessing runs bit-identical.

---

## Decision: Lean, project-shaped documentation structure
- **Date:** 2026-05-28
- **Status:** Accepted
- **Context:** A documentation-infrastructure task prescribed a web-app `docs/00-08` template
  (routes, screens, components, REST API, DB, auth, LLM prompts). This repo is a Python research
  library with none of those — only optional stdlib HTTP demo viewers.
- **Decision:** Build the numbered `docs/00-08` tree but **only the docs that fit a library**.
  Web-specific files were dropped or renamed to library equivalents: ROUTE_INVENTORY →
  `ENTRYPOINTS_AND_SCRIPTS`, SCREEN_INVENTORY → folded into `08-visuals`, COMPONENT_INVENTORY →
  `CODEBASE_INVENTORY`, API_INVENTORY → `PUBLIC_API`, STATE_MANAGEMENT → `STATE_AND_ENCODING`;
  PROMPT_INVENTORY and auth/DB docs dropped (no such code).
- **Why:** Honesty over template-completeness — no empty "Not applicable" stubs; documents the
  actual system.
- **Alternatives considered:** Faithful full template with N/A markers (rejected: low-value
  stubs). 
- **Consequences:** ~30 docs instead of ~40; future agents must map web terms to the adapted
  names (captured in [`AUDIT_LOG.md`](AUDIT_LOG.md) and the index).
- **Related files:** all of `docs/00-08`.

## Decision: Reorganize loose narrative docs into `06-history/archive/`
- **Date:** 2026-05-28
- **Status:** Accepted
- **Context:** `KEY_FINDINGS.md`, `PHASE3_SUMMARY.md`, and `portfolio-readiness-audit/` lived at
  the repo top level; the portfolio audit predated (and is partly contradicted by) the current README.
- **Decision:** Move them under [`archive/`](archive/ARCHIVE_NOTES.md), preserve content verbatim
  (only fixed relative image paths), update the README links, and note what is superseded.
- **Why:** Cleaner top level; clear separation of historical narrative from live docs.
- **Consequences:** Plain-text mentions in immutable ADRs now point at the archive location
  (documented in `ARCHIVE_NOTES.md` rather than editing the ADRs).

## How to add a decision
- Architecturally significant → add a new ADR in `docs/adr/` (next number; never edit a committed
  one — supersede it) and add a one-line row here.
- Process/structure decisions (like docs layout) → record here in the format above.
