# ADR-001: Int8 grid encoding with dual PATH/CLAIMED codes

**Status:** Accepted
**Date:** 2026-04-22

## Context

The board has three cell states per player (empty / player's path / player's
claimed territory) and up to four players. A naive encoding would use one
boolean plane per (player × kind), yielding a shape `(2*P, H, W)` tensor
(16 planes at 4 players). But the engine runs inside MCTS tree-search
nodes, so the hot cost is *state cloning*, not inference: every MCTS
node clone pays the memory copy price.

Alternatives considered:

- **One int8 plane per kind** (2 planes: paths vs. claimed, player
  id stored as the integer value). Halves the planes of a boolean
  multi-plane layout but still doubles copy cost vs. a single plane.
- **Packed int8** (one plane, high nibble = kind, low nibble = player
  id). Minimum copy cost but branchy decoding everywhere.
- **Single int8 with flat enumerated codes** (the choice).

## Decision

A single `np.int8` array of shape `(board_size, board_size)`:

- `0` = EMPTY
- `1..4` = `PATH_CODES[player_id]` — that player's snake path
- `5..8` = `CLAIMED_CODES[player_id]` — that player's claimed territory

Eight distinct values fit trivially in int8 with headroom. Constants are
exposed as both scalar names (`PLAYER_1_PATH`) and indexed tuples
(`PATH_CODES`, `CLAIMED_CODES`), so code stays symmetric across players.

## Consequences

- **Cheap clone:** one `numpy` `memcpy` per clone instead of a
  multi-plane copy. Measured <50 µs at 40×40 (see
  `benchmarks/OPTIMIZATION_REPORT.md`).
- **Cache locality:** enclosure BFS and win-detection walks traverse
  the grid linearly, which is cache-friendly.
- **Symmetric code:** `PATH_CODES[pid]` / `CLAIMED_CODES[pid]` means
  per-player logic is a lookup, not a branch.
- **Ambiguity avoided:** a cell is *either* empty, path, or claimed —
  never overlapping states — so one scalar suffices.
- **Trade-off accepted:** enclosure detection cannot read
  "is this cell on player X's path?" from the grid alone in O(1); it
  needs `PlayerState.path_set`. See ADR-002 for why that cache exists.

## References

- `src/territory_takeover/constants.py` — `PATH_CODES`, `CLAIMED_CODES`,
  `EMPTY`.
- `src/territory_takeover/state.py` — `GameState.grid`.
- `CLAUDE.md` — "Tile encoding" section.
