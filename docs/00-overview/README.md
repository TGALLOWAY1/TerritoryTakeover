# Documentation — Start Here

This is the documentation system for **TerritoryTakeover**, a reproducible multi-agent
decision-systems AI lab (a turn-based grid engine + classical search + RL + a seed-locked
tournament harness). For the project itself, see the root [`README.md`](../../README.md).

## Read in this order
1. [`PROJECT_SNAPSHOT.md`](PROJECT_SNAPSHOT.md) — one-screen orientation (stack, commands, status).
2. [`../01-product/PRODUCT_BRIEF.md`](../01-product/PRODUCT_BRIEF.md) — what it is and who it's for.
3. [`../01-product/CURRENT_BEHAVIOR.md`](../01-product/CURRENT_BEHAVIOR.md) — what actually works today.
4. [`../02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md) — how it's built.
5. [`../04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md) — limitations and deferrals.
6. [`../05-planning/NEXT_AGENT_TASKS.md`](../05-planning/NEXT_AGENT_TASKS.md) — what to do next.

## Full map
See [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md).

## If you're an AI agent
Don't load everything. Start with [`../07-ai-context/CONTEXT_LOADING_PROTOCOL.md`](../07-ai-context/CONTEXT_LOADING_PROTOCOL.md)
and follow [`../07-ai-context/AGENT_WORKFLOW.md`](../07-ai-context/AGENT_WORKFLOW.md).

## How this documentation is organized
```
00-overview/      orientation + this index
01-product/       what it does, features (status-labeled), behavior, user flows
02-architecture/  architecture, system map, data model, public API, encodings, integrations
03-implementation/ codebase inventory, entry points/scripts, config, testing strategy
04-quality/       known issues, technical debt, risks, regression checklist, security
05-planning/      backlog, prioritized TODO, roadmap, next-agent tasks
06-history/       decision log, changelog, audit log, archive/ (old narrative docs)
07-ai-context/    context-loading protocol, agent workflow
08-visuals/       screenshot manifest, visual-regression plan, flow diagrams, screenshots/
```
Pre-existing docs (`../adr/`, `../baseline_report*.md`, `../experiments/`,
`../optimization_analysis/`, `../OPTIMIZATION_ANALYSIS.md`) are preserved and linked from the index.

## Maintaining these docs
Documentation is a first-class artifact: when you change behavior, update the matching doc and use
the exact status labels (defined in [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md)). See
[`../07-ai-context/AGENT_WORKFLOW.md`](../07-ai-context/AGENT_WORKFLOW.md).
