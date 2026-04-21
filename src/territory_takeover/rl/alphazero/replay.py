"""Replay buffer for AlphaZero self-play samples.

Each sample carries everything the training step consumes:

- ``grid``: ``(C, H, W)`` float32, straight from
  :func:`encode_az_observation`.
- ``scalars``: ``(3 + N,)`` float32.
- ``mask``: ``(4,)`` bool — legal-action mask at the state the sample was
  collected from.
- ``visits``: ``(4,)`` float32 — MCTS visit counts (the policy target).
  Stored as counts (not a distribution) so the loss function can
  normalize once consistently.
- ``final_scores``: ``(N,)`` float32 in ``[-1, 1]`` — terminal value
  backed up from the end of the game this sample was collected in. Same
  normalization as :func:`_terminal_value_normalized`.

Storage is numpy arrays pre-allocated to ``capacity``; ``add`` is O(1),
``sample`` is O(batch_size) uniform random. ``save`` / ``load`` serialize
to a single ``.npz`` for reproducibility. The buffer is intentionally
dumb — deduplication, prioritization, or disk rotation live in the
training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Sample:
    """A single training sample emitted by self-play.

    ``final_scores`` holds the per-seat value-head target: the terminal
    normalized score vector in the default "terminal" mode and the
    per-sample n-step bootstrapped return in the "nstep" mode. The name
    is kept for serialization continuity with Phase 3c buffers — the
    shape and dtype are unchanged.

    ``per_step_reward`` and ``step_index`` are metadata populated by
    :mod:`.selfplay` for every sample. They are consumed by unit tests
    and by offline analysis of n-step training; ``train_step`` does
    not read them because the value target is precomputed into
    ``final_scores`` at self-play time. Default ``0.0`` / ``-1``
    preserves backward compatibility with call sites that still
    construct Samples with only the original five fields.
    """

    grid: NDArray[np.float32]
    scalars: NDArray[np.float32]
    mask: NDArray[np.bool_]
    visits: NDArray[np.float32]
    final_scores: NDArray[np.float32]
    per_step_reward: float = 0.0
    step_index: int = -1


class ReplayBuffer:
    """Fixed-capacity ring buffer over tuples emitted by self-play."""

    def __init__(
        self,
        capacity: int,
        grid_shape: tuple[int, int, int],
        scalar_dim: int,
        num_players: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0; got {capacity}")
        self.capacity = capacity
        self.grid_shape = grid_shape
        self.scalar_dim = scalar_dim
        self.num_players = num_players
        self._grids = np.zeros((capacity, *grid_shape), dtype=np.float32)
        self._scalars = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self._masks = np.zeros((capacity, 4), dtype=np.bool_)
        self._visits = np.zeros((capacity, 4), dtype=np.float32)
        self._final = np.zeros((capacity, num_players), dtype=np.float32)
        self._per_step_reward = np.zeros((capacity,), dtype=np.float32)
        self._step_index = np.full((capacity,), -1, dtype=np.int32)
        self._write_idx: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def add(self, sample: Sample) -> None:
        i = self._write_idx
        self._grids[i] = sample.grid
        self._scalars[i] = sample.scalars
        self._masks[i] = sample.mask
        self._visits[i] = sample.visits
        self._final[i] = sample.final_scores
        self._per_step_reward[i] = sample.per_step_reward
        self._step_index[i] = sample.step_index
        self._write_idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def extend(self, samples: list[Sample]) -> None:
        for s in samples:
            self.add(s)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        idx = rng.integers(0, self._size, size=batch_size)
        return (
            self._grids[idx],
            self._scalars[idx],
            self._masks[idx],
            self._visits[idx],
            self._final[idx],
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            grids=self._grids[: self._size],
            scalars=self._scalars[: self._size],
            masks=self._masks[: self._size],
            visits=self._visits[: self._size],
            final=self._final[: self._size],
            per_step_reward=self._per_step_reward[: self._size],
            step_index=self._step_index[: self._size],
            write_idx=np.array([self._write_idx], dtype=np.int64),
            size=np.array([self._size], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path, capacity: int) -> ReplayBuffer:
        with np.load(path) as data:
            grids = data["grids"]
            scalars = data["scalars"]
            masks = data["masks"]
            visits = data["visits"]
            final = data["final"]
            size = int(data["size"][0])
            # Per-step reward / step-index columns were added with the
            # n-step value target. Older saves predate them; load as
            # zeros / -1 so downstream code that only reads `_final`
            # is unaffected.
            per_step_reward = (
                data["per_step_reward"]
                if "per_step_reward" in data.files
                else np.zeros((size,), dtype=np.float32)
            )
            step_index = (
                data["step_index"]
                if "step_index" in data.files
                else np.full((size,), -1, dtype=np.int32)
            )
        grid_shape = tuple(grids.shape[1:])
        assert len(grid_shape) == 3, "grid shape must be (C, H, W)"
        buf = cls(
            capacity=capacity,
            grid_shape=(int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])),
            scalar_dim=int(scalars.shape[1]),
            num_players=int(final.shape[1]),
        )
        n = min(size, capacity)
        buf._grids[:n] = grids[:n]
        buf._scalars[:n] = scalars[:n]
        buf._masks[:n] = masks[:n]
        buf._visits[:n] = visits[:n]
        buf._final[:n] = final[:n]
        buf._per_step_reward[:n] = per_step_reward[:n]
        buf._step_index[:n] = step_index[:n]
        buf._size = n
        buf._write_idx = n % capacity
        return buf


__all__ = ["ReplayBuffer", "Sample"]
