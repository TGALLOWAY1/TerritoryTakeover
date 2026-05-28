# System Map

> Module dependency map. Arrows mean "depends on / imports". Edges are simplified to the
> architecturally significant ones. Last audited 2026-05-28.

## Dependency graph

```mermaid
graph TD
    constants[constants.py]
    state[state.py]
    actions[actions.py]
    engine[engine.py]
    rollout[rollout.py]

    state --> constants
    actions --> constants
    actions --> state
    engine --> constants
    engine --> state
    engine --> actions
    rollout --> actions
    rollout --> engine

    subgraph eval
      features[features.py]
      voronoi[voronoi.py]
      heuristic[heuristic.py]
      tuning[tuning.py]
    end
    features --> actions
    voronoi --> state
    heuristic --> features
    heuristic --> voronoi
    tuning --> heuristic

    subgraph search
      agent[agent.py Protocol]
      randoma[random_agent.py]
      maxn[maxn.py]
      mcts_node[mcts/node.py]
      mcts_rollout[mcts/rollout.py]
      uct[mcts/uct.py]
      rave[mcts/rave.py]
      registry[registry.py]
      harness[harness.py]
    end
    randoma --> agent
    randoma --> heuristic
    maxn --> heuristic
    mcts_rollout --> rollout
    mcts_rollout --> heuristic
    uct --> mcts_node
    uct --> mcts_rollout
    rave --> uct
    registry --> randoma
    registry --> maxn
    registry --> uct
    registry --> rave
    harness --> registry
    harness --> engine

    subgraph rl
      tabular[rl/tabular/*]
      ppo[rl/ppo/*]
      az[rl/alphazero/*]
      curriculum[rl/curriculum/*]
      elo[rl/eval/elo.py]
    end
    tabular --> engine
    ppo --> gym
    az --> engine
    az --> heuristic
    curriculum --> az
    elo --> harness

    gym[gym_env.py] --> engine
    gym --> actions

    subgraph viz
      vz[viz.py]
      vhtml[viz_html.py]
      vlive[viz_live.py]
      vinter[viz_interactive.py]
    end
    vz --> state
    vhtml --> heuristic
    vlive --> registry
    vinter --> registry
```

## Layering rules (read top-to-bottom = allowed dependency direction)

| Layer | Modules | May depend on |
|---|---|---|
| Core | `constants`, `state`, `actions`, `engine`, `rollout` | only core |
| Eval | `eval/*` | core |
| Agents | `search/*`, `rl/*`, `gym_env` | core, eval |
| Benchmark | `search/harness`, `rl/eval/elo` | core, eval, agents |
| Viz | `viz*` | any of the above (read-only views) |

No upward dependencies: the engine core knows nothing about agents, RL, or visualization.

## Hotspots (where most of the complexity/LOC lives)
- `viz_interactive.py` (~1075), `viz_live.py` (~699), `viz_html.py` (~672) — UI/HTTP surface area.
- `search/harness.py` (~623), `search/mcts/uct.py` (~556), `search/mcts/rave.py` (~531) — algorithm core.
- `rl/alphazero/*` (~1751 total) — the deepest RL track.

## Related docs
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layer responsibilities and runtime flow.
- [`docs/08-visuals/FLOW_DIAGRAMS.md`](../08-visuals/FLOW_DIAGRAMS.md) — sequence/flow Mermaid diagrams.
- [`docs/03-implementation/CODEBASE_INVENTORY.md`](../03-implementation/CODEBASE_INVENTORY.md) — per-module symbols + LOC.
