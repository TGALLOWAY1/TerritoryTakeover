"""Pairwise Bradley–Terry Elo computation anchored on a fixed player.

Why BT-MLE instead of the usual iterative K-factor Elo: with a small
closed pool of agents and a round-robin tournament, the online update has
stale-order effects (ratings depend on the order games are consumed).
Maximum-likelihood under the Bradley–Terry model is order-free and
converges in a few iterations, so we use it for the headline Elo table.

Multi-player games are reduced to pairwise outcomes by comparing final
ranks: every game produces ``N*(N-1)/2`` (winner, loser) or (tie) pairs.
This is a simplification over TrueSkill but keeps the estimator's input
shape the same as a pure head-to-head tournament.

Elo units: ratings are reported in the standard "400-log10" convention
(``R_i - R_j = 400 * log10(P(i beats j) / P(j beats i))``). One of the
agents is pinned to zero ("anchor"), so the numbers are interpretable as
"Elo gain over the anchor".

The module is network-agnostic — it only consumes (winner, loser, ties)
game outcomes. A round-robin *runner* lives in ``scripts/compute_elo.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class GameOutcome:
    """One pairwise outcome: either ``winner`` beat ``loser``, or a tie
    between the two agents. Ties are counted as 0.5 wins for each side.
    """

    agent_a: str
    agent_b: str
    score_a: float  # 1.0 if a won, 0.0 if b won, 0.5 if tie.

    def __post_init__(self) -> None:
        if self.agent_a == self.agent_b:
            raise ValueError("agent_a and agent_b must differ")
        if self.score_a not in (0.0, 0.5, 1.0):
            raise ValueError(f"score_a must be 0.0, 0.5, or 1.0; got {self.score_a}")


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    """Aggregate pairwise result between two agents."""

    agent_a: str
    agent_b: str
    wins_a: float  # Includes 0.5 per tie.
    games: int


def outcomes_from_rank(agents: Sequence[str], ranks: Sequence[int]) -> list[GameOutcome]:
    """Convert per-agent final ranks into pairwise :class:`GameOutcome`s.

    ``ranks[i]`` is ``agents[i]``'s final placement (1 = best, ties
    allowed). Returns one outcome per unordered pair: lower rank wins;
    equal ranks produce a tie.
    """
    if len(agents) != len(ranks):
        raise ValueError("agents and ranks must have the same length")
    out: list[GameOutcome] = []
    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            ri = ranks[i]
            rj = ranks[j]
            if ri < rj:
                score = 1.0
            elif ri > rj:
                score = 0.0
            else:
                score = 0.5
            out.append(GameOutcome(agents[i], agents[j], score))
    return out


def _aggregate(outcomes: Iterable[GameOutcome]) -> dict[tuple[str, str], PairwiseResult]:
    """Aggregate GameOutcomes into PairwiseResults keyed by
    ``(lexicographically-smaller, larger)``. ``wins_a`` is stored from
    the smaller agent's perspective for a canonical table.
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for oc in outcomes:
        if oc.agent_a < oc.agent_b:
            key = (oc.agent_a, oc.agent_b)
            wins_for_smaller = oc.score_a
        else:
            key = (oc.agent_b, oc.agent_a)
            wins_for_smaller = 1.0 - oc.score_a
        buckets.setdefault(key, []).append(wins_for_smaller)

    results: dict[tuple[str, str], PairwiseResult] = {}
    for (a, b), scores in buckets.items():
        results[(a, b)] = PairwiseResult(
            agent_a=a, agent_b=b, wins_a=sum(scores), games=len(scores)
        )
    return results


def compute_elo(
    outcomes: Iterable[GameOutcome],
    anchor: str,
    max_iters: int = 200,
    tolerance: float = 1e-7,
) -> dict[str, float]:
    """Estimate Elo ratings under Bradley–Terry MLE.

    ``anchor`` is pinned to 0.0 Elo each iteration; everyone else is
    reported as (Elo gain over the anchor).

    Algorithm: standard BT fixed-point iteration on the strength vector
    ``p_i`` (with ``sum p_i = 1``). One update:

        p_i <- sum_j w_ij / sum_j (n_ij / (p_i + p_j))

    Then ``R_i = 400 * log10(p_i / p_anchor)``.

    Ties contribute 0.5 to both ``w_ij`` and ``w_ji`` (i.e. a tie is the
    same likelihood as half a win for each side — the standard BT tie
    treatment). The algorithm converges to within ``tolerance`` (on the
    max per-agent log-strength delta) in a dozen or so iterations for
    well-connected tournaments.
    """
    pair_results = _aggregate(outcomes)
    if not pair_results:
        raise ValueError("no game outcomes provided")

    agents: set[str] = set()
    for a, b in pair_results:
        agents.add(a)
        agents.add(b)
    if anchor not in agents:
        raise ValueError(f"anchor {anchor!r} did not appear in any outcome")

    agent_list = sorted(agents)
    idx = {a: i for i, a in enumerate(agent_list)}
    n = len(agent_list)

    # wins[i][j] = times i beat j (tie counts as 0.5 for each direction).
    # games[i][j] = total games between i and j.
    wins = [[0.0] * n for _ in range(n)]
    games = [[0.0] * n for _ in range(n)]
    for (a, b), res in pair_results.items():
        ia, ib = idx[a], idx[b]
        wins[ia][ib] += res.wins_a
        wins[ib][ia] += res.games - res.wins_a
        games[ia][ib] += res.games
        games[ib][ia] += res.games

    # Initialize strengths uniformly.
    strengths = [1.0 / n] * n

    for _ in range(max_iters):
        new_strengths = [0.0] * n
        for i in range(n):
            numer = sum(wins[i][j] for j in range(n) if j != i)
            denom = 0.0
            for j in range(n):
                if j == i:
                    continue
                if games[i][j] == 0:
                    continue
                denom += games[i][j] / (strengths[i] + strengths[j])
            if denom <= 0 or numer <= 0:
                # Degenerate: agent has no games or never won.
                new_strengths[i] = 1e-12
            else:
                new_strengths[i] = numer / denom

        # Normalize to sum 1 to avoid drift.
        total = sum(new_strengths)
        if total <= 0:
            raise RuntimeError("strengths collapsed to zero; check tournament connectivity")
        new_strengths = [s / total for s in new_strengths]

        # Convergence check on log-strength delta.
        delta = max(
            abs(math.log(max(new_strengths[i], 1e-18)) - math.log(max(strengths[i], 1e-18)))
            for i in range(n)
        )
        strengths = new_strengths
        if delta < tolerance:
            break

    anchor_strength = strengths[idx[anchor]]
    scale = 400.0 / math.log(10.0)
    ratings = {
        a: scale * (math.log(max(strengths[idx[a]], 1e-18)) - math.log(max(anchor_strength, 1e-18)))
        for a in agent_list
    }
    return ratings


__all__ = ["GameOutcome", "PairwiseResult", "compute_elo", "outcomes_from_rank"]
