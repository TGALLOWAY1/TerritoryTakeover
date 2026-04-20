"""NN evaluator: cached and batched inference for the PUCT search.

Two performance levers matter at search time:

1. **Caching.** The same game state recurs many times across a search (the
   engine is deterministic once an action is chosen), so memoizing
   ``(policy_prior, value)`` by state hash can cut inference-bound time
   drastically. We key on ``(state hash, active_player)`` because two
   players can sit on the same state at different points.

2. **Batching.** A single forward pass over a batch of 32 leaves is
   roughly 32x cheaper than 32 forward passes in a tight Python loop,
   especially on CPU where kernel launches dominate small-tensor work.

The evaluator here is **not** concurrency-safe; Phase 3c self-play runs
serial PUCT searches per move and batches across the leaf queue of a
single search. Concurrent workers would need a lock around the cache and
a futures-style API — deferred as a performance follow-up.

Virtual loss is exposed as a pair of methods (:meth:`apply_virtual_loss`,
:meth:`revert_virtual_loss`) so the MCTS driver can discourage concurrent
workers from re-selecting a path while its leaf eval is still pending.
Serial search does not need this but it costs nothing to implement and
the unit test documents the contract for later.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch

from territory_takeover.rl.alphazero.spaces import encode_az_observation
from territory_takeover.rl.ppo.spaces import LOGIT_MASK_VALUE

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.rl.alphazero.network import AlphaZeroNet
    from territory_takeover.state import GameState


def state_hash(state: GameState, active_player: int) -> int:
    """Hash a ``(state, active_player)`` pair for evaluator caching.

    The grid bytes + active_player + each player's head position uniquely
    identify what the encoder produces. claimed_count is implicit in the
    grid. Turn number isn't hashed because two states with the same grid
    and head positions are symmetric from the network's perspective — turn
    only varies a scalar feature that normalizes to the same bucket within
    a single game. (If we later add features that depend on turn_number,
    include it here.)
    """
    heads = tuple(p.head for p in state.players)
    return hash((state.grid.tobytes(), active_player, heads))


class NNEvaluator:
    """Batched / cached wrapper around an :class:`AlphaZeroNet`.

    Typical use from the MCTS driver::

        evaluator = NNEvaluator(net, device="cpu", batch_size=32)
        prior, value = evaluator.evaluate(state, player_id, legal_mask)

    ``evaluate`` always returns ``(prior (4,), value (num_players,))``
    numpy arrays even when batching under the hood.
    """

    def __init__(
        self,
        net: AlphaZeroNet,
        device: torch.device | str = "cpu",
        batch_size: int = 32,
        cache_size: int = 4096,
        virtual_loss: float = 1.0,
    ) -> None:
        self.net = net.to(device)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.virtual_loss = virtual_loss
        self._cache: OrderedDict[int, tuple[NDArray[np.float32], NDArray[np.float32]]] = (
            OrderedDict()
        )
        self._pending: dict[int, int] = {}
        self.net.eval()

    def reset(self) -> None:
        self._cache.clear()
        self._pending.clear()

    def apply_virtual_loss(self, key: int) -> None:
        """Mark ``key`` as being evaluated by a concurrent worker."""
        self._pending[key] = self._pending.get(key, 0) + 1

    def revert_virtual_loss(self, key: int) -> None:
        """Decrement the virtual-loss counter for ``key``."""
        remaining = self._pending.get(key, 0) - 1
        if remaining <= 0:
            self._pending.pop(key, None)
        else:
            self._pending[key] = remaining

    def has_virtual_loss(self, key: int) -> bool:
        return self._pending.get(key, 0) > 0

    def evaluate(
        self,
        state: GameState,
        active_player: int,
        legal_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return ``(policy_prior (4,), value (num_players,))`` for ``state``.

        ``policy_prior`` is a softmax over masked logits — zero on illegal
        actions, non-negative and summing to 1 over legal actions.
        ``value`` is directly the network's per-seat tanh output in
        ``[-1, 1]``. Scalar-value-head networks return ``value`` of shape
        ``(1,)``; the MCTS driver broadcasts if needed.
        """
        key = state_hash(state, active_player)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            prior, value = cached
            return prior.copy(), value.copy()

        return self.evaluate_batch([(state, active_player, legal_mask)])[0]

    def evaluate_batch(
        self,
        requests: list[tuple[GameState, int, NDArray[np.bool_]]],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Batched evaluation of ``(state, active_player, legal_mask)`` triples.

        Cached entries are returned from the cache; misses are encoded and
        forwarded through the network in chunks of at most
        ``self.batch_size``. Return order matches the input order.
        """
        results: list[tuple[NDArray[np.float32], NDArray[np.float32]] | None] = [
            None
        ] * len(requests)
        miss_idxs: list[int] = []
        miss_grids: list[NDArray[np.float32]] = []
        miss_scalars: list[NDArray[np.float32]] = []
        miss_masks: list[NDArray[np.bool_]] = []

        for i, (state, active_player, legal_mask) in enumerate(requests):
            key = state_hash(state, active_player)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                prior, value = cached
                results[i] = (prior.copy(), value.copy())
                continue
            grid, scalars = encode_az_observation(state, active_player)
            miss_idxs.append(i)
            miss_grids.append(grid)
            miss_scalars.append(scalars)
            miss_masks.append(legal_mask)

        if miss_idxs:
            self._forward_misses(
                requests, results, miss_idxs, miss_grids, miss_scalars, miss_masks
            )

        out: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
        for r in results:
            assert r is not None
            out.append(r)
        return out

    def _forward_misses(
        self,
        requests: list[tuple[GameState, int, NDArray[np.bool_]]],
        results: list[tuple[NDArray[np.float32], NDArray[np.float32]] | None],
        miss_idxs: list[int],
        miss_grids: list[NDArray[np.float32]],
        miss_scalars: list[NDArray[np.float32]],
        miss_masks: list[NDArray[np.bool_]],
    ) -> None:
        total = len(miss_idxs)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            grid_batch = np.stack(miss_grids[start:end], axis=0)
            scalar_batch = np.stack(miss_scalars[start:end], axis=0)
            mask_batch = np.stack(miss_masks[start:end], axis=0)

            grid_t = torch.from_numpy(grid_batch).to(self.device)
            scalar_t = torch.from_numpy(scalar_batch).to(self.device)
            mask_t = torch.from_numpy(mask_batch).to(self.device)

            with torch.no_grad():
                logits, values = self.net(grid_t, scalar_t, mask_t)

            priors = _softmax_safe(logits).cpu().numpy().astype(np.float32)
            values_np = values.cpu().numpy().astype(np.float32)

            for offset, idx in enumerate(miss_idxs[start:end]):
                prior = priors[offset]
                value = values_np[offset]
                state, active_player, _ = requests[idx]
                self._insert_cache(state_hash(state, active_player), prior, value)
                results[idx] = (prior.copy(), value.copy())

    def _insert_cache(
        self,
        key: int,
        prior: NDArray[np.float32],
        value: NDArray[np.float32],
    ) -> None:
        self._cache[key] = (prior, value)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)


def _softmax_safe(masked_logits: torch.Tensor) -> torch.Tensor:
    """Softmax that zeros out positions set to ``LOGIT_MASK_VALUE``.

    A plain ``torch.softmax`` on logits that contain ``LOGIT_MASK_VALUE``
    would already produce near-zero probabilities at those positions, but
    floating-point underflow can leave tiny (~1e-45) mass there. We zero
    explicitly and re-normalize so downstream MCTS code can rely on
    exactly-zero priors for illegal actions.
    """
    probs = torch.softmax(masked_logits, dim=-1)
    legal = masked_logits > LOGIT_MASK_VALUE / 2
    probs = torch.where(legal, probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return probs / denom


__all__ = ["NNEvaluator", "state_hash"]
