# Phase-2 Evaluation Harness — Key Findings

Notes from building `run_match` / `round_robin` / `scripts/run_tournament.py`
in `src/territory_takeover/search/harness.py`. Written so the next person
tuning the tournament pipeline doesn't re-discover the same constraints.

## Acceptance result

All six `tests/test_harness.py` tests green:

- 4-way 8-game match on 20×20 finishes in **under 2 seconds** (budget was
  120s). Random agents on a small board don't stress the harness; the
  budget is mostly insurance for when MCTS agents are swapped in.
- Seat rotation hand-check: with 4 unique-name agents over 4 games,
  every agent visits every seat exactly once.
- `parallel=True` and `parallel=False` produce **bit-identical**
  `GameLog.actions`, `final_scores`, `winner_seat`, and `seat_assignment`
  for the same seed. Two trial seeds (7 and 11) covered.
- `_wilson_ci(50, 100)` matches closed-form Wilson 95% CI to within 1e-3
  (center exactly 0.5 by symmetry).

`ruff check` clean, `mypy --strict` clean (on `src/` + `tests/` per
project config). CLI smoke test with 4 random agents on 12×12 writes
well-formed `match_summary.csv` + `match_games.csv`.

## Design decisions (and their trade-offs)

### 1. Seat rotation: cyclic single-shift, not full permutation

- **Choice**: for game `i`, seat `k` is assigned agent `(k + i) mod N`.
- **Alternative considered**: iterate all `N!` permutations. For 4-player
  games that's 24 distinct orderings, exploding game budgets.
- **Why cyclic is enough**: spawn corners are rotationally symmetric under
  the default spawn layout (`engine._default_spawns`). The only
  statistically interesting covariate is "which corner did this agent
  start in" — cyclic covers that orbit exactly. Head-to-head
  permutations matter for game trees with opening-move asymmetries
  (e.g. chess), not for this spawn geometry.
- **Constraint**: `num_games % N == 0` is validated up front. Silently
  accepting a non-multiple would have each agent playing some seats
  more often, biasing `avg_territory` systematically.

### 2. Round-robin layout: turn-alternating, not adjacent-grouped

- **Choice**: for pair `(A, B)` the two 2v2 layouts are `[A, B, A, B]`
  and `[B, A, B, A]`.
- **Alternative considered**: `[A, A, B, B]`. Symmetric on paper but
  **not** on the board — the default spawns place players at the four
  corners with adjacent seats on the same board edge. `[A, A, B, B]`
  makes the A-pair share an edge while the B-pair shares the opposite
  edge, which changes the enclosure geometry materially. Turn-
  alternating diagonalizes the layout.
- **Cost**: the spec said "ordered pair" but that's a red herring —
  once both layouts are played, (A-first, B-first) and (B-first,
  A-first) are the same match set, so we use unordered
  `combinations` and halve the game count.

### 3. Parallel determinism: `SeedSequence.spawn()`, not `(base, worker, game)` hashing

- **Choice**: per-game seeds come from `np.random.SeedSequence(seed).spawn(num_games)`.
  Each game's sequence is *further* spawned into `1 + num_players`
  children (1 for `new_game`, N for per-agent `_rng`). `parallel=True`
  dispatches the same pre-computed jobs across a `Pool`.
- **Rejected**: hashing `(base_seed, worker_id, game_index)` as the
  task spec initially suggested. Worker ID is a scheduling artifact;
  if the OS assigns game 3 to worker 1 on run A and worker 2 on run B,
  seeds diverge and the parallel-equals-serial test fails. Dropping
  the worker dimension from the seed tree makes determinism depend
  only on `game_index`, which is the correct equivalence.
- **Bonus**: agents get a fresh seeded `_rng` per game via
  `_reseed_agent(agent, seed)`, regardless of whether the agent's
  `reset()` clears the RNG. This defensively closes a UCT-specific
  footgun: `UCTAgent.reset()` clears the tree but intentionally
  preserves `self._rng` across games in the same tournament.

### 4. Agent isolation: pickle every game, even in serial mode

- **Choice**: `_run_one_game` accepts `agents_pickled: bytes` and
  unpickles a fresh copy for every game. The serial path calls
  `_worker(args)` — same codepath as parallel.
- **Why**: ensures the serial loop cannot accidentally rely on agent
  state leaking across games. The parallel-equals-serial test is
  much cheaper to pass when both paths share semantics; otherwise
  every subtle in-place mutation (tree reuse cross-game, feature
  cache, transposition tables) becomes a potential divergence.
- **Cost**: ~one `pickle.loads` per game. For MCTS agents with
  500 iterations that's <1% of game time. For 10k+ iteration agents
  the pickle cost might be worth revisiting, but only for deep agents
  that make pickling expensive.

### 5. Iterations/sec: separate field, NaN for non-counting agents

- **Choice**: `AgentStats.avg_iters_per_s` is `NaN` for agents without
  an iteration counter (random, greedy). MCTS agents use
  `last_search_stats["iterations"]`; Max-N/Paranoid fall back to
  `last_nodes`.
- **Rejected**: a normalized "decisions/sec" metric that treats all
  agents uniformly. Tempting — but Max-N nodes and MCTS iterations
  are non-comparable (one tree descent vs one minimax recursion). A
  single metric would be interpretable only among agents of the same
  family; NaN forces the consumer to acknowledge that.
- **Consequence**: CSV renderers must treat `nan` as a valid cell,
  which the hand-rolled `_format_table` does explicitly.

### 6. Data model: flat dataclasses, lists serialized as JSON in CSV

- **Choice**: `GameLog.actions` / `seat_assignment` / `final_scores` /
  `decision_times_s` / `iterations` are lists. In `match_games.csv`
  they're written as JSON strings (one cell per game).
- **Alternative considered**: fully normalized schema — one row per
  (game, turn) in a `turns.csv`. Strictly more queryable but needs a
  join to get per-game metadata, and the analysis use cases so far
  are "aggregate per agent" (fully served by `match_summary.csv`)
  and "spot-check one game" (served by loading the row and running
  `json.loads`).
- **Pragmatic win**: one CSV per result type, each loadable with
  `pandas.read_csv(..., converters={"actions": json.loads, ...})`.

### 7. YAML config: optional extra, registry in `search/`

- **Choice**: `pyyaml` is a `[tournament]` optional extra; the core
  package stays numpy-only. Registry lives in
  `src/territory_takeover/search/registry.py`, importable from the
  harness itself and from tests (not only the CLI script).
- **Rationale**: tests can build `AgentSpec` directly without pulling
  in YAML parsing. The CLI does `import yaml` inside `_require_yaml`
  and translates `ImportError` into a helpful `SystemExit` pointing
  at the extra.

## Things that will bite the next person

1. **`num_games` must be divisible by `len(agents)`** when
   `swap_seats=True`. This is validated, but the error at
   `run_match(num_games=10, agents=[a, b, c, d])` surprises people
   who expected "round down to the nearest multiple".
2. **`GameState.copy()` is cheap but `pickle.dumps(agent)` for UCT
   with a 10k-node tree is not.** After the first `select_action`,
   UCTAgent carries a live `MCTSNode` tree in `self._root`. Our
   harness pickles agents *before* any game runs, so this is fine —
   but if you add a pre-warmed agent path later, the pickle cost
   grows linearly with tree size. Consider snapshotting
   `AgentSpec`s instead.
3. **`multiprocessing.Pool` uses `fork` on Linux.** The
   parallel-equals-serial test relies on deterministic spawn of
   child processes; macOS/Windows default to `spawn`, which
   re-imports the module and can change behavior if the test runs
   in a non-`__main__` context. The test passes on Linux CI; if you
   see drift on another platform, set `mp.set_start_method("fork")`
   or gate the test.
4. **Wilson CI with `n == 0`** returns `(0.0, 1.0)`. Round-robin will
   only hit this if `games_per_pair == 0`, which is already rejected,
   but it's still the right fallback for a pair row that saw no
   games.
5. **`reset()` plus `_reseed_agent` plus a fresh unpickled instance
   is overkill** on paper — each of the three gives the same
   guarantee alone. Keeping all three is cheap insurance: when the
   next agent variant forgets to clear one kind of state, the other
   two still hold the determinism contract.

## Meta-insights

- **Determinism is a single invariant with multiple sources of truth.**
  Three independent guards (fresh agent instance, `reset()`, RNG
  re-seed) each cost <5 µs but protect against entirely different
  failure modes. The cost budget for harness correctness is large
  enough to afford belt-and-suspenders defenses.
- **Parallel-equals-serial is the strongest single test you can
  write.** It exercises pickling, RNG seeding, seat rotation, game
  driving, and aggregation all at once, and the equality check is
  trivially decidable. Any implementation bug that can silently
  break determinism gets surfaced here, not in production analysis
  after 1000 games.
- **Resist the urge to generalize.** An earlier draft of
  `round_robin` accepted arbitrary `num_players` with a "we'll
  figure out the layouts later" TODO. Killing that and hard-coding
  `num_players == 4` was the right call: the only 2v2 semantics
  that make sense are at 4 players, and anything else deserves a
  different function with a different name.
- **The harness is the instrument, not the experiment.** Resist
  folding per-experiment knobs (agent-specific time budgets, custom
  reward schemes) into `run_match`. Each addition is a new
  interaction surface for bugs. Keep the core `run_match` signature
  stable and layer experimental variants on top.
