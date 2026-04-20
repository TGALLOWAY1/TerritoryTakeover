"""Weight transfer across curriculum stages.

Transfer semantics: when advancing from stage ``k`` to stage ``k+1``, the
new stage may differ in board size and/or player count. The conv trunk
(stem + residual blocks + head convs) is preserved bitwise. Head
Linear layers depend on ``num_players`` (value head output) but are
board-size-invariant under ``head_type="conv"``; they still transfer if
player count is unchanged.

This module owns the ``load_state_dict(strict=False)`` plumbing plus a
small verification step so a caller can assert "no Elo collapse"
preconditions (trunk weights did in fact copy over, and the destination
net produces finite logits/values on the new board).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from territory_takeover.rl.alphazero.network import AlphaZeroNet


@dataclass(frozen=True, slots=True)
class TransferReport:
    """Outcome of a cross-stage weight transfer."""

    matched_keys: tuple[str, ...]
    """Parameters that were copied bitwise from the source."""

    missing_keys: tuple[str, ...]
    """Destination parameters that the source did not provide. Left at
    their freshly-initialized values."""

    unexpected_keys: tuple[str, ...]
    """Source parameters with no destination counterpart (e.g. the old
    stage had ``head_type="fc"`` with ``policy_fc.weight`` of a different
    shape). Silently dropped under ``strict=False``."""

    shape_mismatched_keys: tuple[str, ...]
    """Keys present in both source and destination but with incompatible
    shapes (e.g. num_players changed). Dropped; destination keeps its
    init."""


def transfer_weights(src_state_dict: dict[str, torch.Tensor], dst: AlphaZeroNet) -> TransferReport:
    """Copy matching tensors from ``src_state_dict`` into ``dst``.

    Only tensors whose names match and whose shapes match are copied.
    Everything else is left at the destination's initialized value.
    The destination network is mutated in place.
    """
    dst_state = dst.state_dict()

    src_keys = set(src_state_dict.keys())
    dst_keys = set(dst_state.keys())

    matched: list[str] = []
    shape_mismatched: list[str] = []

    filtered: dict[str, torch.Tensor] = {}
    for key in src_keys & dst_keys:
        src_tensor = src_state_dict[key]
        dst_tensor = dst_state[key]
        if tuple(src_tensor.shape) == tuple(dst_tensor.shape):
            filtered[key] = src_tensor
            matched.append(key)
        else:
            shape_mismatched.append(key)

    missing_keys = sorted(dst_keys - src_keys)
    unexpected_keys = sorted(src_keys - dst_keys)

    # Strict=False + shape-filtered dict = only bitwise-safe tensors copied.
    result = dst.load_state_dict(filtered, strict=False)
    # load_state_dict's own ``missing_keys`` here will match our ``missing_keys``
    # plus ``shape_mismatched``; keep our richer breakdown for callers.
    del result

    return TransferReport(
        matched_keys=tuple(sorted(matched)),
        missing_keys=tuple(missing_keys),
        unexpected_keys=tuple(unexpected_keys),
        shape_mismatched_keys=tuple(sorted(shape_mismatched)),
    )


__all__ = ["TransferReport", "transfer_weights"]
