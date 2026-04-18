"""Turn resolution, move application, and win detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .actions import action_to_coord, legal_action_mask, legal_actions
from .constants import CLAIMED_CODES, DIRECTIONS, EMPTY, PATH_CODES
from .state import GameState, PlayerState

_SPAWN_INSET: int = 4


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
    """Canonical symmetric spawns defined for 2- and 4-player games.

    4 players: all four corners inset by `_SPAWN_INSET`.
    2 players: diagonal corners (top-left, bottom-right).
    """
    inset = _SPAWN_INSET
    far = board_size - 1 - inset
    if num_players == 4:
        return [(inset, inset), (inset, far), (far, inset), (far, far)]
    if num_players == 2:
        return [(inset, inset), (far, far)]
    raise ValueError(
        f"Default spawns are defined only for num_players in (2, 4); got {num_players}"
    )


def _seed_player_state(state: GameState, spawns: list[tuple[int, int]]) -> None:
    """Write spawn tiles into `state.grid` and (re)seed each `PlayerState` in place."""
    for i, spawn in enumerate(spawns):
        r, c = spawn
        state.grid[r, c] = PATH_CODES[i]
        p = state.players[i]
        p.path = [spawn]
        p.path_set = {spawn}
        p.head = spawn
        p.claimed_count = 0
        p.alive = True


def new_game(
    board_size: int = 40,
    num_players: int = 4,
    spawn_positions: list[tuple[int, int]] | None = None,
    seed: int | None = None,
) -> GameState:
    """Build a freshly-initialized `GameState`.

    Args:
        board_size: Side length of the square grid.
        num_players: Number of players (1\u20134; tile encoding only supports up to 4).
        spawn_positions: Optional per-player spawn cells. If omitted, symmetric
            corner spawns are used (supported for `num_players` in (2, 4) only).
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
            path=[],
            path_set=set(),
            head=(-1, -1),
            claimed_count=0,
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


def detect_and_apply_enclosure(
    state: GameState,
    player_id: int,
    placed_cell: tuple[int, int],
) -> int:
    """Claim any region enclosed by `player_id`'s path after `placed_cell` was appended.

    Preconditions (caller contract):
      - `placed_cell` has been appended to `state.players[player_id].path` and
        `.path_set`, and is the current `.head`.
      - `state.grid[placed_cell] == PATH_CODES[player_id]`.

    Attribution rule: enclosed empty cells are claimed by `player_id` (the player
    whose placement triggered detection), even if opponent path tiles form part of
    the pocket's boundary. A majority-boundary variant is a possible future
    extension; not implemented here.

    Returns the number of newly-claimed cells (0 if no enclosure formed).
    """
    player = state.players[player_id]
    path = player.path
    path_set = player.path_set
    grid = state.grid

    # Trigger check: a loop can only have just closed if `placed_cell` is
    # orthogonally adjacent to a same-player path tile OTHER than its predecessor
    # (the predecessor is always adjacent and always in path_set).
    predecessor = path[-2] if len(path) >= 2 else None
    r, c = placed_cell
    triggered = False
    for dr, dc in DIRECTIONS:
        nbr = (r + dr, c + dc)
        if nbr == predecessor:
            continue
        if nbr in path_set:
            triggered = True
            break
    if not triggered:
        return 0

    h, w = grid.shape
    reachable = np.zeros((h, w), dtype=np.bool_)
    q: deque[tuple[int, int]] = deque()

    for cc in range(w):
        if grid[0, cc] == EMPTY:
            reachable[0, cc] = True
            q.append((0, cc))
        if h > 1 and grid[h - 1, cc] == EMPTY:
            reachable[h - 1, cc] = True
            q.append((h - 1, cc))
    for rr in range(1, h - 1):
        if grid[rr, 0] == EMPTY:
            reachable[rr, 0] = True
            q.append((rr, 0))
        if w > 1 and grid[rr, w - 1] == EMPTY:
            reachable[rr, w - 1] = True
            q.append((rr, w - 1))

    while q:
        rr, cc = q.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = rr + dr, cc + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not reachable[nr, nc]
                and grid[nr, nc] == EMPTY
            ):
                reachable[nr, nc] = True
                q.append((nr, nc))

    enclosed_mask = (grid == EMPTY) & ~reachable
    count = int(enclosed_mask.sum())
    if count:
        grid[enclosed_mask] = CLAIMED_CODES[player_id]
        player.claimed_count += count
    return count


def _advance_turn(state: GameState) -> None:
    """Move `current_player` forward to the next alive player with ≥1 legal action.

    Any alive player reached with 0 legal actions is marked `alive = False` and
    skipped. `turn_number` is always incremented once (even if nobody can move,
    so the counter still reflects elapsed game time); the caller is responsible
    for setting `state.done` based on post-advance alive counts.
    """
    n = len(state.players)
    start = state.current_player
    for i in range(1, n + 1):
        candidate = (start + i) % n
        p = state.players[candidate]
        if not p.alive:
            continue
        if legal_actions(state, candidate):
            state.current_player = candidate
            state.turn_number += 1
            return
        p.alive = False
    # No alive player has any moves; still bump turn_number.
    state.turn_number += 1


def _compute_winner(state: GameState) -> int | None:
    """Pick the winner by argmax of `len(path) + claimed_count`; ties → None."""
    scores = [len(p.path) + p.claimed_count for p in state.players]
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    return winners[0] if len(winners) == 1 else None


def step(state: GameState, action: int, strict: bool = False) -> StepResult:
    """Apply `action` for `state.current_player`, resolve enclosures, advance turn.

    Args:
        state: Mutated in place. Must have `state.done is False`.
        action: Direction index in `[0, 4)` per `DIRECTIONS` (N, S, W, E).
        strict: If True, raise `IllegalMoveError` on illegal actions. Default
            False — illegal moves mark the player `alive = False` and the turn
            advances, because RL agents can emit garbage and raising mid-episode
            destabilizes training loops.

    Returns:
        `StepResult(state, reward, done, info)`. `reward` is from the moving
        player's perspective: `1.0 + claimed_this_turn` on a legal move, `0.0`
        on an illegal move. `info` keys: `claimed_this_turn`, `player_who_moved`,
        `illegal_move`.
    """
    if state.done:
        raise ValueError("step() called on a finished game (state.done is True)")

    player_who_moved = state.current_player
    info: dict[str, Any] = {
        "claimed_this_turn": 0,
        "player_who_moved": player_who_moved,
        "illegal_move": False,
    }
    reward = 0.0

    mask = legal_action_mask(state, player_who_moved)
    action_in_range = 0 <= action < 4
    if not action_in_range or not bool(mask[action]):
        if strict:
            raise IllegalMoveError(
                f"player {player_who_moved} tried illegal action {action}"
            )
        info["illegal_move"] = True
        state.players[player_who_moved].alive = False
    else:
        target = action_to_coord(state, player_who_moved, action)
        p = state.players[player_who_moved]
        state.grid[target] = PATH_CODES[player_who_moved]
        p.path.append(target)
        p.path_set.add(target)
        p.head = target
        claimed = detect_and_apply_enclosure(state, player_who_moved, target)
        info["claimed_this_turn"] = claimed
        reward = 1.0 + float(claimed)

    _advance_turn(state)

    alive_count = sum(1 for p in state.players if p.alive)
    if alive_count <= 1:
        state.done = True
        state.winner = _compute_winner(state)

    return StepResult(state=state, reward=reward, done=state.done, info=info)


def compute_terminal_reward(
    state: GameState, scheme: str = "sparse"
) -> list[float]:
    """Per-player terminal rewards for RL.

    Args:
        state: Typically a finished `GameState`, but not required — this is a
            pure read of `state.winner` / path & claim sizes.
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
        scores = [float(len(p.path) + p.claimed_count) for p in state.players]
        mean = sum(scores) / n
        centered = [s - mean for s in scores]
        denom = max((abs(c) for c in centered), default=0.0)
        if denom == 0.0:
            return [0.0] * n
        return [c / denom for c in centered]
    raise ValueError(f"Unknown scheme {scheme!r}; expected 'sparse' or 'relative'")
