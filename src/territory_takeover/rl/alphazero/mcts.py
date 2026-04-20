"""PUCT search with neural-network priors and 4-dim value backup.

The search reuses the selection / expansion / backup scaffolding from
:mod:`territory_takeover.search.mcts.uct` in spirit, but diverges in three
places:

1. **Selection uses PUCT** — ``Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) /
   (1 + N(s,a))`` — instead of UCB1. ``P(s,a)`` is the network's prior
   over ``a`` at ``s``; the sqrt exploration term matches AlphaZero.
2. **Expansion calls the NN** instead of running a random rollout. On the
   first visit to a node, we request ``(prior, value)`` from
   :class:`NNEvaluator`, store the prior on the node, and back up the
   per-seat value vector directly.
3. **Root-only Dirichlet noise** mixes ``Dir(alpha)`` into the root
   priors, weighted by ``dirichlet_eps``. This is pure exploration and
   must only fire at the root — mixing noise into interior nodes would
   degrade play quality. Seeded via the passed ``rng`` for reproducibility.

Terminal leaves use :func:`_terminal_value_normalized`, which maps the
existing ``(path + claimed) / board_area`` vector (in ``[0, 1]``) to
``[-1, 1]`` via ``2 * x - 1`` so targets match the network's tanh range.

:class:`AlphaZeroAgent` wraps :func:`puct_search` into the package's
:class:`~territory_takeover.search.agent.Agent` protocol so it drops into
existing tournament/harness code without modification.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from territory_takeover.actions import legal_action_mask, legal_actions
from territory_takeover.engine import step
from territory_takeover.search.mcts.node import MCTSNode
from territory_takeover.search.mcts.rollout import _terminal_value

from .evaluator import NNEvaluator
from .network import AlphaZeroNet, AZNetConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.state import GameState


def _terminal_value_normalized(state: GameState) -> NDArray[np.float64]:
    """Map the engine's ``[0, 1]`` territory vector to ``[-1, 1]``.

    The network's value head emits tanh in ``[-1, 1]``, so the MCTS backup
    values must live in the same range — otherwise leaf values from
    terminal states and leaf values from NN evaluations would be on
    different scales and selection would be biased toward whichever
    happens to be larger.
    """
    raw = _terminal_value(state)
    return 2.0 * raw - 1.0


def _prior_with_dirichlet(
    prior: NDArray[np.float32],
    legal_mask: NDArray[np.bool_],
    alpha: float,
    eps: float,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Mix ``prior`` with symmetric Dirichlet noise over the legal actions.

    Illegal slots stay zero. The mixed prior is re-normalized to 1 over
    the legal set. AlphaGo Zero used ``alpha = 0.03`` for 19x19 Go with
    huge action spaces; for our 4-way branching ``alpha = 0.3`` is a
    reasonable default and is what the plan specifies.
    """
    if eps <= 0:
        return prior.copy()
    legal_idx = np.flatnonzero(legal_mask)
    if legal_idx.size == 0:
        return prior.copy()
    noise = rng.dirichlet([alpha] * legal_idx.size).astype(np.float32)
    mixed = prior.copy()
    for j, i in enumerate(legal_idx):
        mixed[i] = (1.0 - eps) * prior[i] + eps * float(noise[j])
    s = mixed.sum()
    if s > 0:
        mixed /= s
    return mixed


def _puct_score(
    parent_visits: int,
    child_visits: int,
    child_q: float,
    prior: float,
    c_puct: float,
) -> float:
    """Compute the PUCT score for a child from the parent's perspective."""
    exploration = c_puct * prior * math.sqrt(max(parent_visits, 1)) / (1 + child_visits)
    return child_q + exploration


def _select_puct(
    root: MCTSNode,
    child_priors: dict[int, dict[int, float]],
    c_puct: float,
) -> tuple[MCTSNode, int]:
    """Descend by PUCT until a node that needs expansion or is terminal.

    ``child_priors[id(node)]`` holds the network's prior distribution over
    actions at ``node``, set at the node's first visit during expansion.
    Unexpanded actions exist as entries in ``node.untried_actions`` and
    still need to be attached as children.
    """
    depth = 0
    node = root
    while True:
        if node.terminal:
            return node, depth
        if node.untried_actions:
            return node, depth
        if not node.children:
            return node, depth
        parent_pid = node.player_to_move
        priors = child_priors.get(id(node), {})
        best_score = -math.inf
        best_child: MCTSNode | None = None
        for action, child in node.children.items():
            q = child.q_value(parent_pid)
            p = priors.get(action, 0.0)
            score = _puct_score(node.visits, child.visits, q, p, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            return node, depth
        node = best_child
        depth += 1


def _expand_with_nn(
    node: MCTSNode,
    evaluator: NNEvaluator,
    child_priors: dict[int, dict[int, float]],
    *,
    add_dirichlet: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Attach children for every legal action and return the NN value.

    Differs from UCT's one-child-at-a-time expansion: AlphaZero expands
    all legal children in a single pass so their priors are available
    immediately. The prior for each action is stored in
    ``child_priors[id(node)]``; children themselves start unexpanded.

    Returns the per-seat value vector (length ``num_players``) for
    backup.
    """
    if node.terminal:
        if node.terminal_value is None:
            node.terminal_value = _terminal_value_normalized(node.state)
        return node.terminal_value

    mask = legal_action_mask(node.state, node.state.current_player)
    prior, value = evaluator.evaluate(node.state, node.state.current_player, mask)

    if add_dirichlet:
        prior = _prior_with_dirichlet(
            prior, mask, dirichlet_alpha, dirichlet_eps, rng
        )

    legal = legal_actions(node.state, node.state.current_player)
    per_action: dict[int, float] = {}
    for a in legal:
        per_action[a] = float(prior[a])
    child_priors[id(node)] = per_action
    node.untried_actions = list(legal)

    if value.shape[0] == 1:
        # Scalar value head: broadcast into every seat. This is only used
        # in the ablation. The current player's score is the one that was
        # trained, but with scalar head we don't have per-seat targets, so
        # we duplicate and let the MCTS select based on player_to_move.
        broadcast = np.full(len(node.state.players), float(value[0]), dtype=np.float64)
        return broadcast
    return value.astype(np.float64)


def _attach_all_children(node: MCTSNode) -> None:
    """Instantiate an :class:`MCTSNode` per untried action and clear the list.

    Called after a node is expanded with NN priors: we want the children
    to exist before selection revisits this node, so PUCT can score them.
    Each child is attached with visits=0, and :func:`_select_puct` will
    pick via the ``prior * sqrt(N) / (1 + 0)`` term on its first visit.
    """
    if not node.untried_actions:
        return
    for action in node.untried_actions:
        child_state = node.state.copy()
        step(child_state, action, strict=True)
        child = MCTSNode(child_state, parent=node, incoming_action=action)
        node.children[action] = child
    node.untried_actions = []


def _backpropagate(node: MCTSNode, value: NDArray[np.float64]) -> None:
    cur: MCTSNode | None = node
    while cur is not None:
        cur.visits += 1
        cur.total_value += value
        cur = cur.parent


def puct_search(
    root_state: GameState,
    root_player: int,
    evaluator: NNEvaluator,
    iterations: int,
    c_puct: float = 1.25,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    rng: np.random.Generator | None = None,
    temperature: float = 1.0,
) -> tuple[int, NDArray[np.float32]]:
    """Run PUCT for ``iterations`` and return ``(action, visit_counts (4,))``.

    Visit counts are the policy training target. ``temperature`` controls
    how the returned action is sampled from the visit distribution:
    ``temperature == 0`` picks the argmax (deterministic / eval),
    ``temperature == 1`` samples proportional to visit counts (self-play
    exploration). In-between values raise visits to ``1 / temperature``
    before normalizing.
    """
    if root_state.current_player != root_player:
        raise ValueError(
            f"puct_search: root_state.current_player={root_state.current_player} "
            f"!= root_player={root_player}"
        )
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1; got {iterations}")
    rng = rng if rng is not None else np.random.default_rng()

    root = MCTSNode(root_state.copy())
    child_priors: dict[int, dict[int, float]] = {}

    # Prime the root with an expansion before the main loop so selection
    # has children to score on iteration 0.
    root_value = _expand_with_nn(
        root,
        evaluator,
        child_priors,
        add_dirichlet=(dirichlet_eps > 0),
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
        rng=rng,
    )
    _attach_all_children(root)
    _backpropagate(root, root_value)

    for _ in range(iterations):
        leaf, _ = _select_puct(root, child_priors, c_puct)
        if leaf.terminal:
            if leaf.terminal_value is None:
                leaf.terminal_value = _terminal_value_normalized(leaf.state)
            value = leaf.terminal_value
        else:
            value = _expand_with_nn(
                leaf,
                evaluator,
                child_priors,
                add_dirichlet=False,
                rng=rng,
            )
            _attach_all_children(leaf)
        _backpropagate(leaf, value)

    visits = np.zeros(4, dtype=np.float32)
    for action, child in root.children.items():
        visits[action] = float(child.visits)

    action = _sample_action_from_visits(visits, temperature, rng)
    return action, visits


def _sample_action_from_visits(
    visits: NDArray[np.float32],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if temperature <= 0:
        return int(np.argmax(visits))
    if temperature == 1.0:
        total = float(visits.sum())
        if total <= 0.0:
            return int(rng.integers(4))
        probs = visits / total
        return int(rng.choice(4, p=probs))
    scaled = np.power(visits, 1.0 / temperature)
    total = float(scaled.sum())
    if total <= 0.0:
        return int(rng.integers(4))
    probs = scaled / total
    return int(rng.choice(4, p=probs))


class AlphaZeroAgent:
    """Drop-in :class:`~territory_takeover.search.agent.Agent` wrapper."""

    name: str

    def __init__(
        self,
        net: AlphaZeroNet,
        iterations: int = 100,
        c_puct: float = 1.25,
        device: str = "cpu",
        temperature_eval: float = 0.0,
        name: str = "alphazero",
        seed: int | None = None,
    ) -> None:
        self.evaluator = NNEvaluator(net, device=device)
        self.iterations = iterations
        self.c_puct = c_puct
        self.temperature_eval = temperature_eval
        self.name = name
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        cfg: AZNetConfig,
        iterations: int = 100,
        c_puct: float = 1.25,
        device: str = "cpu",
        name: str = "alphazero",
        seed: int | None = None,
    ) -> AlphaZeroAgent:
        net = AlphaZeroNet(cfg)
        state_dict = torch.load(path, map_location=device)
        net.load_state_dict(state_dict)
        net.eval()
        return cls(
            net,
            iterations=iterations,
            c_puct=c_puct,
            device=device,
            name=name,
            seed=seed,
        )

    def select_action(
        self,
        state: GameState,
        player_id: int,
        time_budget_s: float | None = None,
        max_iterations: int | None = None,
    ) -> int:
        iters = max_iterations if max_iterations is not None else self.iterations
        action, _ = puct_search(
            state,
            player_id,
            self.evaluator,
            iterations=iters,
            c_puct=self.c_puct,
            dirichlet_eps=0.0,  # no Dirichlet at eval time
            rng=self._rng,
            temperature=self.temperature_eval,
        )
        return action

    def reset(self) -> None:
        self.evaluator.reset()


__all__ = [
    "AlphaZeroAgent",
    "puct_search",
]
