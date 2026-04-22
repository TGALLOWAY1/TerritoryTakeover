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

![Demo: RAVE@200 vs curriculum_ref@4 at 20×20, seed 0 — 70 frames at 4 fps](docs/assets/demo.gif)

*One deterministic 20×20 / 2-player game between RAVE (at 200 sims/move)
and the curriculum AlphaZero reference checkpoint (at 4 PUCT iters).
Regenerate with `python scripts/record_demo.py --seed 0`.*

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

## Reproducibility

The tournament harness derives every per-game seed and every per-agent
RNG from a single root integer via `numpy.random.SeedSequence`'s spawn
tree. Serial and multiprocessing runs from the same seed produce
bit-identical game logs (see
[`search/harness.py::run_match`](src/territory_takeover/search/harness.py)).

To regenerate the committed benchmark reports from scratch:

```
# 20x20 canonical leaderboard (the headline above) — ~30 min on 16 cores
python scripts/run_baseline_report.py \
    --board-size 20 --games-per-pair 20 --parallel --seed 0

# 10x10 sanity-check baseline — ~15 min serial
python scripts/run_baseline_report.py --games-per-pair 40 --seed 0

# Curriculum PUCT scaling sweep (H(a) vs H(b) test at 20x20)
python scripts/run_puct_scaling.py \
    --board-size 20 --games-per-opponent 10 \
    --az-iters 4 16 64 --uct-iterations 100 --parallel --seed 0
```

Reference curriculum checkpoint at
[`docs/phase3d/net_reference.pt`](docs/phase3d/net_reference.pt); its
training config is mirrored at
[`docs/phase3d/reference_config.yaml`](docs/phase3d/reference_config.yaml).

## Further reading

Architecture and conventions:

- [`CLAUDE.md`](CLAUDE.md) — tile encoding, state split, engine entry
  points, performance targets, coding conventions.
- [`docs/adr/`](docs/adr/) — architecture decision records (6 initial
  ADRs: int8 grid encoding, state split, enclosure BFS, value-target
  choice, AlphaZero gating deferral, reproducibility).

Benchmark reports (committed markdown tables):

- [`docs/baseline_report_20x20.md`](docs/baseline_report_20x20.md) — headline 20×20 leaderboard.
- [`docs/baseline_report.md`](docs/baseline_report.md) — 10×10 sanity-check baseline.
- [`docs/curriculum_puct_scaling.md`](docs/curriculum_puct_scaling.md) — curriculum scaling sweep.

Experiment writeups:

- [`docs/experiments/20x20_hypothesis_test.md`](docs/experiments/20x20_hypothesis_test.md) — H(a) vs H(b) study on curriculum scaling.

Phase-level research narrative:

- [`KEY_FINDINGS.md`](KEY_FINDINGS.md) — running lab notebook (Phase 3a / 3c / 3d).
- [`PHASE3_SUMMARY.md`](PHASE3_SUMMARY.md) — cross-phase synthesis + deferrals.

Performance engineering:

- [`docs/OPTIMIZATION_ANALYSIS.md`](docs/OPTIMIZATION_ANALYSIS.md) — hotspot / cost-breakdown analysis.
- [`benchmarks/OPTIMIZATION_REPORT.md`](benchmarks/OPTIMIZATION_REPORT.md) — before/after optimization writeup.
- [`benchmarks/TUNING_FINDINGS.md`](benchmarks/TUNING_FINDINGS.md), [`benchmarks/HARNESS_FINDINGS.md`](benchmarks/HARNESS_FINDINGS.md), [`benchmarks/ROLLOUT_FINDINGS.md`](benchmarks/ROLLOUT_FINDINGS.md) — subsystem-specific perf notes.

## Known limitations and open questions

- **AlphaZero snapshot-gating is stubbed** (`rl/alphazero/train.py:12-14`,
  `:207-209`). The latest self-play snapshot always becomes the
  self-play champion. This was a deliberate Phase-3c scope cut; the
  20×20 study in `docs/experiments/20x20_hypothesis_test.md` argues
  that value-head quality is the pipeline's weakest link and motivates
  finishing the gating tournament with that remit.
- **Curriculum checkpoint is out-of-distribution above 10×10.** It was
  trained on a 6×6 → 8×8 → 10×10 schedule. Its `head_type=conv`
  architecture accepts arbitrary `H×W` but strength does not scale
  monotonically with eval-time PUCT compute at 20×20 — see the writeup
  for the evidence.
- **No automated benchmark CI.** The baseline reports are run locally
  and checked into `docs/`; re-running on every PR would be expensive
  and flaky on shared runners. Reproducibility is provided via the
  commands above plus the seed-locked harness, not CI.

