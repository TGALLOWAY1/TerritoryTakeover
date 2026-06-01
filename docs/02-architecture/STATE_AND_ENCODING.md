# State and Encoding

> How game state is represented in memory and encoded for the neural agents. This is the
> library equivalent of "state management". Last audited 2026-05-28.

## Tile encoding (one int8 grid)

The board is a single `np.int8` array of shape `(H, W)` (`state.grid`). Values:

| Range | Meaning |
|---|---|
| `0` | EMPTY |
| `1..4` | `PATH_CODES[player_id]` — a player's snake path |
| `5..8` | `CLAIMED_CODES[player_id]` — a player's claimed territory |

Constants expose both scalar names (`PLAYER_1_PATH`, …) and the indexed tuples
`PATH_CODES`/`CLAIMED_CODES` — prefer the tuples when a `player_id` is in scope so code stays
symmetric across players. `DIRECTIONS = (N, S, W, E)`; the action space is **fixed at 4**
regardless of board size (keeps the policy-head shape constant for MCTS/RL). Rationale: ADR-001.

## State split: grid + per-player caches

The grid is the **source of truth**. `PlayerState` carries redundant caches for speed:
`path` (ordered list), `path_set` (O(1) membership), `head`, `claimed_count`. Global counters
`empty_count` and `alive_count` are likewise caches. Every mutation must keep all caches in
lockstep with the grid (ADR-002). `GameState.copy()` is tuned for tree search (memcpy grid,
shallow-copy path list/set, value-copy scalars). Details in [`DATA_MODEL.md`](DATA_MODEL.md).

## Neural observation encodings

The two neural tracks encode `GameState` into tensors **differently, on purpose**.

### PPO encoder (`rl/ppo/spaces.py`) — active-player-rotated
- `grid_planes`: shape **`(2N + 2, H, W)`**
  - channels `0..N-1`: per-player PATH, **active player rotated to channel 0** (symmetric input).
  - channels `N..2N-1`: per-player CLAIMED, same rotation.
  - channel `2N`: EMPTY mask.
  - channel `2N+1`: one-hot head of the active player.
- `scalar_features`: shape **`(3 + N,)`** — normalized turn; per-player claimed counts (rotated);
  board fill ratio; active player's cell fraction.
- Action masking uses `LOGIT_MASK_VALUE = -1e9` (finite, fp16-safe), not `-inf`.
- Rotation is a pure permutation, invertible at inference.

### AlphaZero encoder (`rl/alphazero/spaces.py`) — fixed seat order
- `grid_planes`: shape **`(3N + 2, H, W)`**
  - channels `0..N-1`: per-seat PATH (native order).
  - channels `N..2N-1`: per-seat CLAIMED.
  - channel `2N`: EMPTY mask.
  - channel `2N+1`: one-hot head of active player.
  - channels `2N+2..3N+1`: per-seat turn one-hot (all-ones on active seat).
- `scalar_features`: shape `(3 + N,)`, **no rotation** (claimed counts in native seat order).
- **Why fixed order:** the value head outputs a length-`N` vector where `value[i]` = expected
  normalized score for seat `i`. That only learns if seat identity is stable in the input — so
  AlphaZero keeps seats fixed and adds the turn one-hot, instead of rotating like PPO.
- Helpers: `grid_channel_count(N) = 3N+2`, `scalar_feature_dim(N) = 3+N`.

### Gym observation (`gym_env.py`)
Dict space: `grid`, `current_player`, `heads`, `action_mask` (the legal `(4,)` mask). Action
space `Discrete(4)`.

### Tabular encoding (`rl/tabular/state_encoder.py`)
`encode_state(state, compress=False) -> StateKey` (a nested tuple, hashable for the Q-table);
`decode_state` reconstructs a `GameState`. Compression is lossy for larger boards (by design).

## Value-target convention
AlphaZero/curriculum nets output values in `[-1, 1]` (tanh); classical rollouts return per-player
`(path + claimed) / board_area` in `[0, 1]`. Terminal vs. n-step targets: ADR-004.

## Related docs
- [`DATA_MODEL.md`](DATA_MODEL.md) · [`PUBLIC_API.md`](PUBLIC_API.md)
- ADR-001 (int8 grid), ADR-002 (state split), ADR-004 (value target) in [`docs/adr/`](../adr/README.md)
