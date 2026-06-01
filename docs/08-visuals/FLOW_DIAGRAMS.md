# Flow Diagrams

> Sequence/flow diagrams for the core runtime paths. Mermaid blocks render natively on GitHub and
> were validated during authoring. For the module dependency graph see
> [`docs/02-architecture/SYSTEM_MAP.md`](../02-architecture/SYSTEM_MAP.md). Last audited 2026-05-28.

## 1. Engine: one move (`step` + enclosure)

```mermaid
flowchart TD
    A[Agent.select_action] --> B[engine.step state, action]
    B --> C{In bounds and target EMPTY?}
    C -- no --> D[mark player alive=False]
    D --> T[advance turn]
    C -- yes --> E[write PATH code, append path/path_set, advance head, empty_count--]
    E --> F[detect_and_apply_enclosure]
    F --> G{trigger: adjacent same-player path other than predecessor?}
    G -- no --> H[claimed_this_turn = 0]
    G -- yes --> I[boundary BFS over EMPTY, non-empty = wall]
    I --> J[claim enclosed cells for triggering player, update grid + claimed_count]
    J --> K[reward = 1.0 + claimed_this_turn]
    H --> K
    K --> T
    T --> L{alive_count <= 1?}
    L -- yes --> M[_compute_winner, done=True]
    L -- no --> N[return StepResult]
    M --> N
```
Source: `engine.py` (`step`, `detect_and_apply_enclosure`, `_advance_turn`, `_compute_winner`).

## 2. Tournament harness (seed-locked, reproducible)

```mermaid
flowchart TD
    R[root seed integer] --> SS[numpy SeedSequence spawn tree]
    SS --> PS[per-game seeds + per-agent RNGs]
    PS --> RM[run_match / round_robin]
    RM --> PG[play_game: agents drive engine.step, seats rotated]
    PG --> TS[terminal GameState per game]
    TS --> ST[AgentStats: wins/losses/ties + timing]
    ST --> TB[Table with Wilson 95% CI]
    TB --> RP[committed markdown + CSV report]
    RM -. optional .-> MP[multiprocessing workers]
    MP -. bit-identical to serial .-> PG
```
Source: `search/harness.py`; reproducibility rationale in ADR-006.

## 3. AlphaZero self-play → training loop

```mermaid
flowchart TD
    INIT[initialize AlphaZeroNet] --> SP[self-play: puct_search with NNEvaluator]
    SP --> SAMP[Samples: grid, scalars, mask, MCTS visits, terminal value]
    SAMP --> RB[ReplayBuffer]
    RB --> TR[train_step: masked policy CE + value MSE + L2]
    TR --> SNAP[snapshot net]
    SNAP --> GATE{gating tournament: candidate beats champion?}
    GATE -. STUBBED: always promote latest .-> SP
    GATE -. planned: promote only if win-rate >= threshold .-> SP
    SNAP --> SP
```
Source: `rl/alphazero/{selfplay,train,mcts,evaluator,replay}.py`. The gating branch is the one
deliberate stub (`train.py:207-210`, ADR-005) — see
[`docs/04-quality/KNOWN_ISSUES.md`](../04-quality/KNOWN_ISSUES.md).

## Related docs
- [`docs/02-architecture/ARCHITECTURE.md`](../02-architecture/ARCHITECTURE.md) — prose runtime/data flow.
- [`SCREENSHOT_MANIFEST.md`](SCREENSHOT_MANIFEST.md) — rendered game artifacts.
