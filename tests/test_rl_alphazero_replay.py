"""Tests for the Phase 3c ReplayBuffer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from territory_takeover.rl.alphazero.replay import ReplayBuffer, Sample


def _make_sample(seed: int) -> Sample:
    rng = np.random.default_rng(seed)
    return Sample(
        grid=rng.random((8, 6, 6), dtype=np.float32),
        scalars=rng.random(5, dtype=np.float32),
        mask=np.array([True, True, False, True], dtype=np.bool_),
        visits=rng.random(4, dtype=np.float32),
        final_scores=rng.random(2, dtype=np.float32),
    )


def test_add_and_len() -> None:
    buf = ReplayBuffer(capacity=10, grid_shape=(8, 6, 6), scalar_dim=5, num_players=2)
    assert len(buf) == 0
    buf.add(_make_sample(0))
    buf.add(_make_sample(1))
    assert len(buf) == 2


def test_ring_overwrites_after_capacity() -> None:
    buf = ReplayBuffer(capacity=3, grid_shape=(8, 6, 6), scalar_dim=5, num_players=2)
    for i in range(5):
        buf.add(_make_sample(i))
    assert len(buf) == 3


def test_sample_returns_right_shapes() -> None:
    buf = ReplayBuffer(capacity=10, grid_shape=(8, 6, 6), scalar_dim=5, num_players=2)
    for i in range(8):
        buf.add(_make_sample(i))

    rng = np.random.default_rng(0)
    g, s, m, v, fs = buf.sample(batch_size=4, rng=rng)

    assert g.shape == (4, 8, 6, 6)
    assert s.shape == (4, 5)
    assert m.shape == (4, 4)
    assert v.shape == (4, 4)
    assert fs.shape == (4, 2)


def test_sample_empty_buffer_raises() -> None:
    buf = ReplayBuffer(capacity=5, grid_shape=(8, 6, 6), scalar_dim=5, num_players=2)
    rng = np.random.default_rng(0)
    try:
        buf.sample(batch_size=2, rng=rng)
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty buffer")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    buf = ReplayBuffer(capacity=5, grid_shape=(8, 6, 6), scalar_dim=5, num_players=2)
    samples = [_make_sample(i) for i in range(3)]
    buf.extend(samples)

    save_path = tmp_path / "buf.npz"
    buf.save(save_path)

    restored = ReplayBuffer.load(save_path, capacity=10)
    assert len(restored) == 3
    rng = np.random.default_rng(0)
    g, _s, m, _v, _fs = restored.sample(batch_size=3, rng=rng)
    assert g.shape[0] == 3
    assert m.dtype == np.bool_
