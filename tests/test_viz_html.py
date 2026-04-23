"""Tests for :mod:`territory_takeover.viz_html`."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from territory_takeover.actions import legal_actions
from territory_takeover.engine import new_game, step
from territory_takeover.eval.heuristic import default_evaluator
from territory_takeover.search.agent import Agent
from territory_takeover.search.mcts.rave import RaveAgent
from territory_takeover.search.random_agent import (
    HeuristicGreedyAgent,
    UniformRandomAgent,
)
from territory_takeover.search.registry import REGISTRY, STRATEGY_LABELS
from territory_takeover.state import GameState
from territory_takeover.viz_html import (
    AgentCard,
    build_payload,
    heuristic_win_probs,
    load_elo_csv,
    render_html,
    save_game_html,
)


def _short_trajectory(steps: int = 4) -> list[GameState]:
    state = new_game(8, 2, seed=0)
    traj = [state.copy()]
    for _ in range(steps):
        acts = legal_actions(state, state.current_player) or [0]
        step(state, acts[0])
        traj.append(state.copy())
    return traj


def test_heuristic_win_probs_sums_to_one() -> None:
    state = new_game(8, 2, seed=0)
    probs = heuristic_win_probs(state, default_evaluator())
    assert probs.shape == (2,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)


def test_heuristic_win_probs_dead_player_is_zero() -> None:
    state = new_game(8, 2, seed=0)
    state.players[1].alive = False
    probs = heuristic_win_probs(state, default_evaluator())
    assert probs[1] == 0.0
    assert np.isclose(probs[0], 1.0)


def test_heuristic_win_probs_rejects_bad_temperature() -> None:
    state = new_game(8, 2, seed=0)
    with pytest.raises(ValueError):
        heuristic_win_probs(state, default_evaluator(), temperature=0.0)


def test_heuristic_win_probs_terminal_matches_winner() -> None:
    # At a terminal state, the probability vector should reflect the
    # engine's computed winner rather than the heuristic's alive-filtered
    # softmax (the winner can be a dead-by-territory player).
    state = new_game(8, 2, seed=0)
    state.done = True
    state.winner = 1
    probs = heuristic_win_probs(state, default_evaluator())
    assert probs[0] == 0.0
    assert probs[1] == 1.0


def test_heuristic_win_probs_terminal_tie_is_uniform() -> None:
    state = new_game(8, 4, seed=0)
    state.done = True
    state.winner = None
    probs = heuristic_win_probs(state, default_evaluator())
    assert np.allclose(probs, 0.25)


def test_strategy_labels_cover_registry() -> None:
    # Every registered agent class should have a strategy label so the HTML
    # card renders a readable string rather than a raw class name.
    for class_name in REGISTRY:
        assert class_name in STRATEGY_LABELS, (
            f"{class_name} missing from STRATEGY_LABELS"
        )


def test_strategy_label_lookup_by_instance() -> None:
    # Construct one of each registered agent class and confirm lookup by
    # ``type(agent).__name__`` resolves. MCTS agents require ``iterations``;
    # MaxN / Paranoid default depth is fine.
    extra_kwargs: dict[str, dict[str, object]] = {
        "HeuristicGreedyAgent": {"evaluator": default_evaluator()},
        "UCTAgent": {"iterations": 4},
        "RaveAgent": {"iterations": 4},
    }
    for class_name, ctor in REGISTRY.items():
        kwargs: dict[str, object] = {"name": class_name.lower()}
        kwargs.update(extra_kwargs.get(class_name, {}))
        agent = ctor(**kwargs)
        assert STRATEGY_LABELS[type(agent).__name__]


def test_build_payload_frame_shape() -> None:
    traj = _short_trajectory(steps=3)
    probs = [heuristic_win_probs(s, default_evaluator()) for s in traj]
    cards = [
        AgentCard(seat=0, name="rave", strategy="rave", elo=50.0),
        AgentCard(seat=1, name="greedy", strategy="heuristic-greedy", elo=None),
    ]
    payload = build_payload(
        trajectory=traj,
        agent_cards=cards,
        win_probs_per_frame=probs,
        title="unit-test",
        fps=4,
    )
    assert payload["num_players"] == 2
    assert payload["board_height"] == 8
    assert payload["board_width"] == 8
    frames = payload["frames"]
    assert isinstance(frames, list)
    assert len(frames) == len(traj)
    first = frames[0]
    assert isinstance(first, dict)
    assert len(first["grid"]) == 8 * 8
    assert len(first["win_probs"]) == 2


def test_build_payload_length_mismatch_raises() -> None:
    traj = _short_trajectory(steps=2)
    cards = [
        AgentCard(seat=0, name="a", strategy="random", elo=None),
        AgentCard(seat=1, name="b", strategy="random", elo=None),
    ]
    with pytest.raises(ValueError):
        build_payload(
            trajectory=traj,
            agent_cards=cards,
            win_probs_per_frame=[],
            title="t",
            fps=4,
        )


def test_build_payload_rejects_bad_fps() -> None:
    traj = _short_trajectory(steps=1)
    probs = [heuristic_win_probs(s, default_evaluator()) for s in traj]
    cards = [
        AgentCard(seat=0, name="a", strategy="random", elo=None),
        AgentCard(seat=1, name="b", strategy="random", elo=None),
    ]
    with pytest.raises(ValueError):
        build_payload(
            trajectory=traj,
            agent_cards=cards,
            win_probs_per_frame=probs,
            title="t",
            fps=0,
        )


def test_render_html_escapes_closing_script_tag() -> None:
    # A literal "</script>" inside a JSON string payload would end the
    # embedding script tag early; the renderer must escape it.
    traj = _short_trajectory(steps=1)
    probs = [heuristic_win_probs(s, default_evaluator()) for s in traj]
    cards = [
        AgentCard(seat=0, name="</script>attack", strategy="rave", elo=None),
        AgentCard(seat=1, name="b", strategy="random", elo=None),
    ]
    payload = build_payload(
        trajectory=traj,
        agent_cards=cards,
        win_probs_per_frame=probs,
        title="t",
        fps=4,
    )
    rendered = render_html(payload)
    # The malicious name must appear escaped, not as a real closing tag.
    assert "</script>attack" not in rendered
    assert "<\\/script>attack" in rendered


def test_save_game_html_smoke(tmp_path: Path) -> None:
    traj = _short_trajectory(steps=4)
    probs = [heuristic_win_probs(s, default_evaluator()) for s in traj]
    cards = [
        AgentCard(seat=0, name="rave", strategy="rave", elo=88.6),
        AgentCard(seat=1, name="greedy", strategy="heuristic-greedy", elo=None),
    ]
    out = tmp_path / "demo.html"
    save_game_html(
        trajectory=traj,
        agent_cards=cards,
        win_probs_per_frame=probs,
        path=out,
        title="TerritoryTakeover smoke",
        fps=4,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "TerritoryTakeover smoke" in text
    # The embedded JSON payload should be parseable.
    match = re.search(
        r'<script id="tt-data" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    assert payload["num_players"] == 2
    assert payload["agents"][1]["elo"] is None  # None → rendered as "—"
    assert payload["agents"][0]["elo"] == pytest.approx(88.6)
    assert len(payload["frames"]) == len(traj)


def test_load_elo_csv_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "elo.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["agent", "elo_vs_anchor", "anchor"])
        writer.writerow(["rave", "42.5", "random"])
        writer.writerow(["greedy", "not-a-number", "random"])
        writer.writerow(["random", "0", "random"])
    ratings = load_elo_csv(csv_path)
    assert ratings == {"rave": 42.5, "random": 0.0}


def test_load_elo_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_elo_csv(tmp_path / "does-not-exist.csv")


def test_win_probs_respond_to_claimed_territory() -> None:
    # When a player has a clear territorial lead, the heuristic win-prob
    # estimator should assign them the larger share.
    state = new_game(10, 2, seed=0)
    # Force a few no-op style moves so RAVE/greedy would differ, then
    # synthetically boost player 0's claimed count to simulate a lead.
    from territory_takeover.constants import CLAIMED_CODES

    lead_code = CLAIMED_CODES[0]
    for r in range(3, 7):
        for c in range(3, 7):
            if state.grid[r, c] == 0:
                state.grid[r, c] = lead_code
                state.players[0].claimed_count += 1

    probs = heuristic_win_probs(state, default_evaluator())
    assert probs[0] > probs[1]
    assert np.isclose(probs.sum(), 1.0)


def test_end_to_end_short_game_html(tmp_path: Path) -> None:
    # Run a 2-player game to termination with a tiny RAVE budget so the test
    # stays fast, then render HTML end-to-end.
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    agents: list[Agent] = [
        RaveAgent(iterations=8, name="rave-tiny", rng=rng0),
        HeuristicGreedyAgent(rng=rng1, name="greedy"),
    ]

    state = new_game(8, 2, seed=0)
    traj = [state.copy()]
    probs = [heuristic_win_probs(state, default_evaluator())]

    turns_cap = 120
    turns = 0
    while not state.done and turns < turns_cap:
        seat = state.current_player
        action = agents[seat].select_action(state, seat)
        step(state, action, strict=True)
        traj.append(state.copy())
        probs.append(heuristic_win_probs(state, default_evaluator()))
        turns += 1

    cards = [
        AgentCard(seat=0, name=agents[0].name, strategy="rave", elo=50.0),
        AgentCard(seat=1, name=agents[1].name, strategy="heuristic-greedy", elo=33.0),
    ]
    out = tmp_path / "game.html"
    save_game_html(
        trajectory=traj,
        agent_cards=cards,
        win_probs_per_frame=probs,
        path=out,
        fps=6,
    )
    assert out.exists()
    assert out.stat().st_size > 1024


def test_uniform_random_agent_has_strategy_label() -> None:
    agent = UniformRandomAgent(rng=np.random.default_rng(0))
    assert STRATEGY_LABELS[type(agent).__name__] == "random"
