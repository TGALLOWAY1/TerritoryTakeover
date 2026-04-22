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
