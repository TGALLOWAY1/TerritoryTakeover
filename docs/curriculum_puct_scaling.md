# Curriculum Reference — PUCT Scaling Sweep

Curriculum reference vs. a fixed panel (Random, Greedy, UCT@100) at 20x20, 2 players, 10 games per (PUCT, opponent) pairing.

Tests whether the curriculum checkpoint's playing strength scales with eval-time PUCT compute. Note: the checkpoint was trained only up to 10x10 — its 20x20 behaviour is out-of-distribution.

## Per-opponent breakdown

| PUCT iters | Opponent | Games | Wins | Ties | Losses | Win rate | 95% CI |
|---:|---|---:|---:|---:|---:|---:|---|
| 4 | random | 10 | 4 | 2 | 4 | 0.400 | [0.168, 0.687] |
| 4 | greedy | 10 | 5 | 0 | 5 | 0.500 | [0.237, 0.763] |
| 4 | uct100 | 10 | 2 | 0 | 8 | 0.200 | [0.057, 0.510] |
| 16 | random | 10 | 2 | 1 | 7 | 0.200 | [0.057, 0.510] |
| 16 | greedy | 10 | 3 | 0 | 7 | 0.300 | [0.108, 0.603] |
| 16 | uct100 | 10 | 2 | 0 | 8 | 0.200 | [0.057, 0.510] |
| 64 | random | 10 | 2 | 1 | 7 | 0.200 | [0.057, 0.510] |
| 64 | greedy | 10 | 5 | 0 | 5 | 0.500 | [0.237, 0.763] |
| 64 | uct100 | 10 | 2 | 0 | 8 | 0.200 | [0.057, 0.510] |

## Aggregate win rate by PUCT budget

| PUCT iters | Aggregate games | Wins | Ties | Losses | Aggregate win rate | 95% CI | Avg decision (s) |
|---:|---:|---:|---:|---:|---:|---|---:|
| 4 | 30 | 11 | 2 | 17 | 0.367 | [0.219, 0.545] | 1.5615 |
| 16 | 30 | 7 | 1 | 22 | 0.233 | [0.118, 0.409] | 4.5715 |
| 64 | 30 | 9 | 1 | 20 | 0.300 | [0.167, 0.479] | 21.0530 |

## Reproducibility

- Board: 20x20, 2 players
- Seed: 0
- Games per (PUCT, opponent): 10
- UCT reference iterations: 100
- AlphaZero c_puct: 1.25
- Checkpoint: `docs/phase3d/net_reference.pt`
- Commit: `6c6a66b`
- Command: `scripts/run_puct_scaling.py --board-size 20 --games-per-opponent 10 --az-iters 4 16 64 --uct-iterations 100 --parallel --seed 0`

Wilson 95% intervals are computed on curriculum win rate (ties count against the win rate).
