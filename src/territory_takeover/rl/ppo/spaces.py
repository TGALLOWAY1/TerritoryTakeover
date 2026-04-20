"""Observation encoder and logit-level action masking for Phase 3b PPO.

The encoder turns a :class:`GameState` into two float32 tensors that can be
consumed by the actor-critic network (see :mod:`.network`):

- ``grid_planes``: shape ``(2N + 2, H, W)``.
  - Channels ``0 .. N - 1``: one per player's PATH cells, with the active
    player rotated to channel ``0`` so every input looks symmetric to the
    network ("I'm always channel 0").
  - Channels ``N .. 2N - 1``: one per player's CLAIMED cells, same rotation.
  - Channel ``2N``: EMPTY mask.
  - Channel ``2N + 1``: one-hot head position of the active player (all zeros
    if the active player has been eliminated and has no head).

- ``scalar_features``: shape ``(3 + N,)``.
  - Index 0: normalized turn number (``turn / (H * W)``, clamped to ``[0, 1]``).
  - Indices 1..N: per-player normalized claimed counts (``claimed / (H * W)``),
    rotated so the active player is index 1.
  - Index N + 1: board fill ratio (``1 - empty / (H * W)``).
  - Index N + 2: fraction of the active player's cells on the board
    (path + claimed, normalized) — cheap "how long is my snake" feature that
    correlates with enclosure potential.

Rotation is a pure channel / index permutation — the underlying player IDs
are untouched, so this can be inverted at inference time if needed.

Action masking uses a finite negative constant (``LOGIT_MASK_VALUE = -1e9``)
rather than ``-inf``. That keeps softmax gradients well-defined in fp32 and
numerically safe if the training loop ever drops to fp16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from territory_takeover.actions import legal_action_mask
from territory_takeover.constants import CLAIMED_CODES, EMPTY, PATH_CODES

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


LOGIT_MASK_VALUE: Final[float] = -1e9
"""Value substituted into illegal-action logits before softmax."""


def _rotated_player_order(active_player: int, num_players: int) -> list[int]:
    """Return player IDs with ``active_player`` first.

    Example: ``_rotated_player_order(2, 4) == [2, 3, 0, 1]``. Rotating rather
    than only moving the active to slot 0 preserves relative ordering of
    opponents, which may matter for larger player counts.
    """
    return [(active_player + i) % num_players for i in range(num_players)]


def encode_observation(
    state: GameState, player_id: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Encode ``state`` from ``player_id``'s perspective.

    Returns a pair ``(grid_planes, scalar_features)`` of float32 numpy arrays.

    The encoder never mutates ``state``. It reads ``state.grid`` directly
    (vectorized numpy comparisons) rather than iterating ``path_set`` for each
    player, which scales better to 40x40 boards.
    """
    num_players = len(state.players)
    h, w = state.grid.shape
    grid = state.grid

    planes = np.zeros((2 * num_players + 2, h, w), dtype=np.float32)
    order = _rotated_player_order(player_id, num_players)

    for out_idx, pid in enumerate(order):
        planes[out_idx] = (grid == PATH_CODES[pid]).astype(np.float32)
        planes[num_players + out_idx] = (grid == CLAIMED_CODES[pid]).astype(
            np.float32
        )

    planes[2 * num_players] = (grid == EMPTY).astype(np.float32)

    head_plane = planes[2 * num_players + 1]
    head = state.players[player_id].head
    if state.players[player_id].alive and head != (-1, -1):
        head_plane[head[0], head[1]] = 1.0

    total_cells = float(h * w)
    empty_count = float(np.count_nonzero(grid == EMPTY))
    active_occupancy = float(
        np.count_nonzero(grid == PATH_CODES[player_id])
        + np.count_nonzero(grid == CLAIMED_CODES[player_id])
    )

    scalars = np.zeros(3 + num_players, dtype=np.float32)
    scalars[0] = min(state.turn_number / total_cells, 1.0)
    for out_idx, pid in enumerate(order):
        scalars[1 + out_idx] = state.players[pid].claimed_count / total_cells
    scalars[1 + num_players] = 1.0 - empty_count / total_cells
    scalars[2 + num_players] = active_occupancy / total_cells

    return planes, scalars


def legal_mask_array(state: GameState, player_id: int) -> NDArray[np.bool_]:
    """Numpy bool mask of legal actions (shape ``(4,)``).

    Thin wrapper over :func:`territory_takeover.actions.legal_action_mask`;
    exists so downstream code can depend on :mod:`.spaces` without reaching
    back into the engine module.
    """
    return legal_action_mask(state, player_id)


def legal_mask_tensor(
    state: GameState,
    player_id: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Torch bool tensor of legal actions (shape ``(4,)``).

    ``device`` defaults to CPU. Callers doing vectorized rollout should build
    the mask on CPU and move once per minibatch.
    """
    mask = legal_mask_array(state, player_id)
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Substitute :data:`LOGIT_MASK_VALUE` at illegal positions.

    ``logits`` is expected to be shape ``(..., 4)`` and ``mask`` is a bool
    tensor of the same shape (or broadcastable). Legal positions keep their
    logit; illegal positions are overwritten.

    This must be applied BEFORE softmax so gradient flow through the policy
    head is unaffected on legal actions — post-softmax masking would leave
    gradient leaking into the illegal logits.
    """
    if logits.shape[-1] != mask.shape[-1]:
        raise ValueError(
            f"logits last dim {logits.shape[-1]} must match mask last dim "
            f"{mask.shape[-1]}"
        )
    return torch.where(
        mask.to(dtype=torch.bool),
        logits,
        torch.full_like(logits, LOGIT_MASK_VALUE),
    )


__all__ = [
    "LOGIT_MASK_VALUE",
    "apply_action_mask",
    "encode_observation",
    "legal_mask_array",
    "legal_mask_tensor",
]
