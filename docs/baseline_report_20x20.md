# Territory Takeover — Baseline Report

Round-robin head-to-head on 20x20, 2 players, 20 games per pair (alternating seats, `SeedSequence`-derived per-game seeds).

## Leaderboard

| Rank | Agent | Games | Wins | Ties | Losses | Win rate | 95% CI | Avg decision (s) |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 1 | rave | 80 | 61 | 1 | 18 | 0.762 | [0.659, 0.842] | 0.7560 |
| 2 | uct | 80 | 51 | 1 | 28 | 0.637 | [0.528, 0.734] | 0.7384 |
| 3 | curriculum_ref | 80 | 33 | 5 | 42 | 0.412 | [0.311, 0.522] | 1.4122 |
| 4 | greedy | 80 | 24 | 0 | 56 | 0.300 | [0.211, 0.408] | 0.0095 |
| 5 | random | 80 | 24 | 7 | 49 | 0.300 | [0.211, 0.408] | 0.0002 |

## Head-to-head

| vs | random | greedy | uct | rave | curriculum_ref |
|---|:---:|:---:|:---:|:---:|:---:|
| random | — | 9/0/11 | 8/1/11 | 1/1/18 | 6/5/9 |
| greedy | 11/0/9 | — | 1/0/19 | 2/0/18 | 10/0/10 |
| uct | 11/1/8 | 19/0/1 | — | 7/0/13 | 14/0/6 |
| rave | 18/1/1 | 18/0/2 | 13/0/7 | — | 12/0/8 |
| curriculum_ref | 9/5/6 | 10/0/10 | 6/0/14 | 8/0/12 | — |

*(cell format: wins / ties / losses from row's perspective, out of games_per_pair)*

## Reproducibility

- Board: 20x20, 2 players
- Seed: 0
- UCT iterations: 200
- RAVE iterations: 200
- AlphaZero iterations: 4, c_puct: 1.25
- Checkpoint: `docs/phase3d/net_reference.pt`
- Commit: `e8bd847`
- Command: `scripts/run_baseline_report.py --board-size 20 --games-per-pair 20 --parallel --seed 0 --md-out docs/baseline_report_20x20.md --csv-out docs/baseline_report_20x20.csv`

Wilson 95% intervals are computed on overall win rate (ties counted against the win rate).
