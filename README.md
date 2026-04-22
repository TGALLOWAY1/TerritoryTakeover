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

## Install (dev)

```
pip install -e ".[dev]"
```

Requires Python 3.11+. `numpy` is the only runtime dependency; `torch`
is pulled in by the `rl_deep` extra (auto-included by `dev`) for the
AlphaZero / curriculum agents.

## Test / lint / typecheck

```
pytest             # 344 tests across 41 files
ruff check .
mypy               # strict mode, configured in pyproject.toml
```
