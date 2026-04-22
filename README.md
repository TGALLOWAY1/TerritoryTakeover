# Territory Takeover

A turn-based multi-agent grid-game engine with a full decision-systems
AI stack — classical tree search (Random, Greedy, Max-N, Paranoid, UCT,
RAVE) and modern RL (Tabular Q, PPO primitives, AlphaZero self-play,
curriculum learning) — evaluated under one reproducible tournament
harness.

The project is set up as a lab: a deterministic, testable simulation
core at the bottom, a layered algorithm stack on top, and a benchmark
suite that emits committed markdown reports every agent can be
compared against.

## Headline result

5-way round-robin head-to-head at 20×20 / 2 players, 20 games per pair
(200 games total). Overall win rate, Wilson 95% CI, ties counted
against the win rate:

| Rank | Agent          | Win rate | 95% CI          |
|-----:|----------------|---------:|-----------------|
| 1    | rave @ 200     | 0.762    | [0.659, 0.842]  |
| 2    | uct @ 200      | 0.637    | [0.528, 0.734]  |
| 3    | curriculum_ref | 0.412    | [0.311, 0.522]  |
| 4    | greedy         | 0.300    | [0.211, 0.408]  |
| 5    | random         | 0.300    | [0.211, 0.408]  |

Full report (including the head-to-head matrix and reproducibility
footer) at [`docs/baseline_report_20x20.md`](docs/baseline_report_20x20.md).
A 10×10 sanity-check baseline is also maintained at
[`docs/baseline_report.md`](docs/baseline_report.md).

## What's in the box

| Subsystem                  | Status                              | LOC   | Key modules |
|----------------------------|-------------------------------------|------:|-------------|
| Core engine                | Production                          |   509 | [`engine.py`](src/territory_takeover/engine.py) |
| Game state                 | Production                          |   152 | [`state.py`](src/territory_takeover/state.py) |
| Actions / legal moves      | Production                          |    89 | [`actions.py`](src/territory_takeover/actions.py) |
| Rollout fast path          | Production                          |   114 | [`rollout.py`](src/territory_takeover/rollout.py) |
| Search: Random / Greedy    | Production                          |   115 | [`search/random_agent.py`](src/territory_takeover/search/random_agent.py) |
| Search: Max-N / Paranoid   | Production                          |   354 | [`search/maxn.py`](src/territory_takeover/search/maxn.py) |
| Search: UCT MCTS           | Production                          |   556 | [`search/mcts/uct.py`](src/territory_takeover/search/mcts/uct.py) |
| Search: RAVE MCTS          | Production                          |   531 | [`search/mcts/rave.py`](src/territory_takeover/search/mcts/rave.py) |
| Tournament harness         | Production                          |   623 | [`search/harness.py`](src/territory_takeover/search/harness.py) |
| RL: Tabular Q-learning     | Reference                           | 1,241 | [`rl/tabular/`](src/territory_takeover/rl/tabular/) |
| RL: PPO primitives         | Experimental                        | 1,028 | [`rl/ppo/`](src/territory_takeover/rl/ppo/) |
| RL: AlphaZero              | Experimental — *gating stubbed*     | 1,751 | [`rl/alphazero/`](src/territory_takeover/rl/alphazero/) |
| RL: Curriculum             | Reference — *checkpoint shipped*    |   732 | [`rl/curriculum/`](src/territory_takeover/rl/curriculum/) |
| Evaluation / heuristics    | Production                          | 1,056 | [`eval/`](src/territory_takeover/eval/) |
| Visualization              | Production                          |   306 | [`viz.py`](src/territory_takeover/viz.py) |
| Gym environment            | Production                          |   433 | [`gym_env.py`](src/territory_takeover/gym_env.py) |

Legend:

- **Production** — typed, tested, stable public API; safe to build on.
- **Reference** — complete implementation with a shipped checkpoint or
  baseline result, but not every hyperparameter is exhaustively tuned.
- **Experimental** — complete enough to run end-to-end, but at least
  one pipeline-relevant component is deliberately deferred. Today's one
  such deferral is the AlphaZero snapshot-gating tournament in
  [`rl/alphazero/train.py`](src/territory_takeover/rl/alphazero/train.py)
  (the latest snapshot always becomes the self-play champion; no
  evaluation-gated promotion). `docs/experiments/20x20_hypothesis_test.md`
  argues this is where the next meaningful RL investment should go.

## Install (dev)

```
pip install -e ".[dev]"
```

Requires Python 3.11+. `numpy` is the only runtime dependency; `torch`
is pulled in by the `rl_deep` extra (auto-included by `dev`) for the
AlphaZero / curriculum agents.

## Test / lint / typecheck

```
pytest             # 369 tests across 42 files (~8k LOC)
ruff check .
mypy               # strict mode, configured in pyproject.toml
```
