"""Turn resolution, move application, and win detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import DIRECTIONS, EMPTY, OWNED_CODES
from .state import GameState, PlayerState


class IllegalMoveError(Exception):
    """Raised by `step` under `strict=True` when the action is not legal."""


@dataclass(slots=True)
class StepResult:
    """Return value of `step`.

    `state` is the same `GameState` instance passed in (mutated in place);
    it's echoed here for ergonomic chaining. `info` always contains keys
    `claimed_this_turn` (int), `player_who_moved` (int), `illegal_move` (bool).
    """

    state: GameState
    reward: float
    done: bool
    info: dict[str, Any]


def _default_spawns(board_size: int, num_players: int) -> list[tuple[int, int]]:
    """Canonical spawns: the exact corners of the board.

    4 players: all four corners. 2 players: diagonal corners
    (top-left, bottom-right).
    """
    far = board_size - 1
    if num_players == 4:
        return [(0, 0), (0, far), (far, 0), (far, far)]
    if num_players == 2:
        return [(0, 0), (far, far)]
    raise ValueError(
        f"Default spawns are defined only for num_players in (2, 4); got {num_players}"
    )


def _seed_player_state(state: GameState, spawns: list[tuple[int, int]]) -> None:
    """Write spawn tiles into `state.grid` and (re)seed each `PlayerState` in place.

    The spawn cell counts as the player's first claimed cell. Also seeds
    `state.alive_count` to the total seat count, since every seated player
    starts alive.
    """
    for i, spawn in enumerate(spawns):
        r, c = spawn
        state.grid[r, c] = OWNED_CODES[i]
        p = state.players[i]
        p.head = spawn
        p.territory_count = 1
        p.alive = True
    state.alive_count = len(spawns)
    h, w = state.grid.shape
    state.empty_count = h * w - len(spawns)


def new_game(
    board_size: int = 40,
    num_players: int = 4,
    spawn_positions: list[tuple[int, int]] | None = None,
    seed: int | None = None,
) -> GameState:
    """Build a freshly-initialized `GameState`.

    Args:
        board_size: Side length of the square grid.
        num_players: Number of players (1-4; tile encoding only supports up to 4).
        spawn_positions: Optional per-player spawn cells. If omitted, the exact
            board corners are used (supported for `num_players` in (2, 4) only).
        seed: Optional RNG seed. When `spawn_positions is None` and `seed is not None`,
            the default spawn list is permuted so player-to-corner assignment is
            randomized but reproducible. Ignored when explicit `spawn_positions`
            are supplied.

    Raises:
        ValueError: if `board_size` < 1, `num_players` not in [1, 4], spawn count
            mismatches `num_players`, any spawn is out of bounds, or spawn cells
            are not unique.
    """
    if board_size < 1:
        raise ValueError(f"board_size must be >= 1; got {board_size}")
    if not 1 <= num_players <= 4:
        raise ValueError(f"num_players must be in [1, 4]; got {num_players}")
    if board_size < 2 and num_players > 1:
        raise ValueError(
            f"board_size {board_size} cannot host {num_players} distinct corner spawns"
        )

    if spawn_positions is None:
        spawns = _default_spawns(board_size, num_players)
        if seed is not None:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(spawns))
            spawns = [spawns[int(i)] for i in order]
    else:
        if len(spawn_positions) != num_players:
            raise ValueError(
                f"spawn_positions length {len(spawn_positions)} != num_players {num_players}"
            )
        for r, c in spawn_positions:
            if not (0 <= r < board_size and 0 <= c < board_size):
                raise ValueError(
                    f"spawn ({r}, {c}) is out of bounds for board_size {board_size}"
                )
        if len(set(spawn_positions)) != num_players:
            raise ValueError(f"spawn_positions contain duplicates: {spawn_positions}")
        spawns = list(spawn_positions)

    grid = np.zeros((board_size, board_size), dtype=np.int8)
    players = [
        PlayerState(
            player_id=i,
            head=(-1, -1),
            territory_count=0,
            alive=True,
        )
        for i in range(num_players)
    ]
    state = GameState(
        grid=grid,
        players=players,
        current_player=0,
        turn_number=0,
        winner=None,
        done=False,
    )
    _seed_player_state(state, spawns)
    return state


def reset(state: GameState) -> None:
    """Restore `state` in place to a fresh default-spawn initial configuration.

    Reuses the existing grid buffer (no reallocation) so this is cheap inside
    RL training loops. Infers `board_size` from `state.grid` and `num_players`
    from `len(state.players)`. Requires `num_players in (2, 4)`; callers that
    need custom spawns should call `new_game` again instead.
    """
    height, width = state.grid.shape
    if height != width:
        raise ValueError(f"reset requires a square grid; got shape {state.grid.shape}")
    board_size = height
    num_players = len(state.players)

    spawns = _default_spawns(board_size, num_players)

    state.grid.fill(EMPTY)
    _seed_player_state(state, spawns)
    state.current_player = 0
    state.turn_number = 0
    state.winner = None
    state.done = False


def has_reachable_empty(state: GameState, player_id: int) -> bool:
    """Return True iff an EMPTY cell is reachable from the player's head.

    Reachability is through the player's own territory: a step is allowed
    onto own cells or EMPTY cells, so an EMPTY cell is reachable iff a path
    of own cells connects the head to some cell adjacent to it. This is the
    liveness rule — a player with no reachable EMPTY cell can never claim
    new territory and is dead.

    Algorithm: because a player's territory is connected and only ever
    grows, liveness is equivalent to "some EMPTY cell is adjacent to own
    territory". A successful check caches that empty cell as a witness on
    the `PlayerState`; the witness stays valid until that exact cell is
    claimed, so repeat checks are O(1) amortized. On a stale/absent witness
    we fall back to a BFS over the player's own cells that early-exits on
    the first EMPTY neighbor discovered, marking visited cells in a
    reusable scratch buffer via a monotonically-increasing stamp (no
    per-call allocation).
    """
    grid = state.grid
    h, w = grid.shape
    player = state.players[player_id]
    r, c = player.head

    # Witness fast path: the empty cell found by the last successful check
    # proves liveness as long as it is still EMPTY.
    witness = player.alive_witness
    if witness is not None:
        if grid.item(witness[0], witness[1]) == EMPTY:
            return True
        player.alive_witness = None

    # Fast path: an adjacent EMPTY cell means a claim is available right now.
    if r > 0 and grid.item(r - 1, c) == EMPTY:
        player.alive_witness = (r - 1, c)
        return True
    if r < h - 1 and grid.item(r + 1, c) == EMPTY:
        player.alive_witness = (r + 1, c)
        return True
    if c > 0 and grid.item(r, c - 1) == EMPTY:
        player.alive_witness = (r, c - 1)
        return True
    if c < w - 1 and grid.item(r, c + 1) == EMPTY:
        player.alive_witness = (r, c + 1)
        return True

    own = OWNED_CODES[player_id]

    scratch = state._scratch_reachable
    if scratch is None or scratch.shape != grid.shape:
        scratch = np.zeros((h, w), dtype=np.int32)
        state._scratch_reachable = scratch
    stamp = state._reach_stamp + 1
    if stamp >= 0x7FFFFFFF:
        scratch.fill(0)
        stamp = 1
    state._reach_stamp = stamp

    scratch[r, c] = stamp
    q: deque[tuple[int, int]] = deque()
    q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if scratch.item(nr, nc) == stamp:
                continue
            code = grid.item(nr, nc)
            if code == EMPTY:
                player.alive_witness = (nr, nc)
                return True
            if code == own:
                scratch[nr, nc] = stamp
                q.append((nr, nc))
    return False


def _advance_turn(state: GameState) -> None:
    """Move `current_player` forward to the next alive player who can still claim.

    Any alive player reached whose head cannot reach an EMPTY cell (see
    :func:`has_reachable_empty`) is marked `alive = False` and skipped (and
    `state.alive_count` decremented) — this is the death rule. `turn_number`
    is always incremented once (even if nobody can move, so the counter still
    reflects elapsed game time); the caller is responsible for setting
    `state.done` based on post-advance alive counts.
    """
    n = len(state.players)
    start = state.current_player
    for i in range(1, n + 1):
        candidate = (start + i) % n
        p = state.players[candidate]
        if not p.alive:
            continue
        if has_reachable_empty(state, candidate):
            state.current_player = candidate
            state.turn_number += 1
            return
        p.alive = False
        state.alive_count -= 1
    # No alive player can claim anything; still bump turn_number.
    state.turn_number += 1


def _compute_winner(state: GameState) -> int | None:
    """Pick the winner by argmax of `territory_count`; ties → None."""
    scores = [p.territory_count for p in state.players]
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    return winners[0] if len(winners) == 1 else None


def step(state: GameState, action: int, strict: bool = False) -> StepResult:
    """Apply `action` for `state.current_player`, then advance the turn.

    Args:
        state: Mutated in place. Must have `state.done is False`.
        action: Direction index in `[0, 4)` per `DIRECTIONS` (N, S, W, E).
        strict: If True, raise `IllegalMoveError` on illegal actions. Default
            False — an illegal move is a wasted turn (head stays put, reward
            0.0) because RL agents can emit garbage and raising mid-episode
            destabilizes training loops. Death comes only from the liveness
            rule (no reachable EMPTY cell), never from a bad action.

    Returns:
        `StepResult(state, reward, done, info)`. `reward` is from the moving
        player's perspective: `1.0` when the move claims a new cell, `0.0`
        for a traversal move over the player's own territory or an illegal
        move. `info` keys: `claimed_this_turn`, `player_who_moved`,
        `illegal_move`.
    """
    if state.done:
        raise ValueError("step() called on a finished game (state.done is True)")

    # Lazy-seed the incremental alive_count for callers that built a
    # `GameState` directly (bypassing `new_game` / `reset`). -1 is the
    # sentinel set by `GameState.__init__`'s default.
    if state.alive_count < 0:
        state.alive_count = sum(1 for _p in state.players if _p.alive)

    player_who_moved = state.current_player
    reward = 0.0
    claimed = 0
    illegal = False

    # Inline legality check — bypasses the np.zeros(4) allocation that
    # `legal_action_mask` performs. The target cell is legal iff in-bounds
    # and EMPTY or the mover's own territory.
    p = state.players[player_who_moved]
    pr, pc = p.head
    grid = state.grid
    h, w = grid.shape
    own = OWNED_CODES[player_who_moved]
    target_code = -1
    tr, tc = pr, pc
    if 0 <= action < 4:
        dr, dc = DIRECTIONS[action]
        tr, tc = pr + dr, pc + dc
        if 0 <= tr < h and 0 <= tc < w:
            target_code = grid.item(tr, tc)
    legal = target_code in (EMPTY, own)

    if not legal:
        if strict:
            raise IllegalMoveError(
                f"player {player_who_moved} tried illegal action {action}"
            )
        illegal = True
    else:
        p.head = (tr, tc)
        if target_code == EMPTY:
            grid[tr, tc] = own
            p.territory_count += 1
            if state.empty_count > 0:
                state.empty_count -= 1
            claimed = 1
            reward = 1.0

    _advance_turn(state)

    if state.alive_count <= 0:
        state.done = True
        state.winner = _compute_winner(state)

    return StepResult(
        state=state,
        reward=reward,
        done=state.done,
        info={
            "claimed_this_turn": claimed,
            "player_who_moved": player_who_moved,
            "illegal_move": illegal,
        },
    )


def compute_terminal_reward(
    state: GameState, scheme: str = "sparse"
) -> list[float]:
    """Per-player terminal rewards for RL.

    Args:
        state: Typically a finished `GameState`, but not required — this is a
            pure read of `state.winner` / territory sizes.
        scheme: "sparse" (default) returns `{+1 win, -1 loss, 0 tie}` using
            `state.winner`. "relative" returns a zero-centered, max-magnitude-
            normalized territory score: `(score_i - mean) / max(|s - mean|, 1)`.

    Raises:
        ValueError: for an unknown `scheme`.
    """
    n = len(state.players)
    if scheme == "sparse":
        if state.winner is None:
            return [0.0] * n
        return [1.0 if i == state.winner else -1.0 for i in range(n)]
    if scheme == "relative":
        scores = [float(p.territory_count) for p in state.players]
        mean = sum(scores) / n
        centered = [s - mean for s in scores]
        denom = max((abs(c) for c in centered), default=0.0)
        if denom == 0.0:
            return [0.0] * n
        return [c / denom for c in centered]
    raise ValueError(f"Unknown scheme {scheme!r}; expected 'sparse' or 'relative'")
