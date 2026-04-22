# Territory Takeover — Baseline Report

Round-robin head-to-head on 10x10, 2 players, 40 games per pair (alternating seats, `SeedSequence`-derived per-game seeds).

## Leaderboard

| Rank | Agent | Games | Wins | Ties | Losses | Win rate | 95% CI | Avg decision (s) |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 1 | uct | 160 | 112 | 5 | 43 | 0.700 | [0.625, 0.766] | 0.0753 |
| 2 | rave | 160 | 102 | 6 | 52 | 0.637 | [0.561, 0.708] | 0.0948 |
| 3 | greedy | 160 | 55 | 13 | 92 | 0.344 | [0.275, 0.420] | 0.0006 |
| 4 | curriculum_ref | 160 | 48 | 24 | 88 | 0.300 | [0.234, 0.375] | 0.0063 |
| 5 | random | 160 | 48 | 22 | 90 | 0.300 | [0.234, 0.375] | 0.0000 |

## Head-to-head

| vs | random | greedy | uct | rave | curriculum_ref |
|---|:---:|:---:|:---:|:---:|:---:|
| random | — | 22/3/15 | 7/3/30 | 5/3/32 | 14/13/13 |
| greedy | 15/3/22 | — | 5/1/34 | 4/0/36 | 31/9/0 |
| uct | 30/3/7 | 34/1/5 | — | 24/1/15 | 24/0/16 |
| rave | 32/3/5 | 36/0/4 | 15/1/24 | — | 19/2/19 |
| curriculum_ref | 13/13/14 | 0/9/31 | 16/0/24 | 19/2/19 | — |

*(cell format: wins / ties / losses from row's perspective, out of games_per_pair)*

## Reproducibility

- Board: 10x10, 2 players
- Seed: 0
- UCT iterations: 200
- RAVE iterations: 200
- AlphaZero iterations: 4, c_puct: 1.25
- Checkpoint: `docs/phase3d/net_reference.pt`
- Commit: `5d9e218`
- Command: `scripts/run_baseline_report.py --games-per-pair 40 --seed 0`

Wilson 95% intervals are computed on overall win rate (ties counted against the win rate).
