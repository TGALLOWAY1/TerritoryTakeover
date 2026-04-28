"""Tests for :mod:`territory_takeover.viz_live`."""

from __future__ import annotations

import json
import urllib.request
from typing import cast

import numpy as np

from territory_takeover.engine import new_game
from territory_takeover.eval.heuristic import default_evaluator
from territory_takeover.viz_html import AgentCard, heuristic_win_probs
from territory_takeover.viz_live import (
    LiveConfig,
    LiveServer,
    play_and_serve,
    render_live_html,
)


def _card(seat: int, name: str = "p") -> AgentCard:
    return AgentCard(seat=seat, name=f"{name}{seat}", strategy="random", elo=None)


def test_render_live_html_escapes_title() -> None:
    page = render_live_html("Foo & <bar>")
    assert "Foo &amp; &lt;bar&gt;" in page
    assert "<canvas" in page
    assert "fetch(" in page  # JS polling present


def test_set_episode_bumps_id_and_clears_frames() -> None:
    s = LiveServer(port=0)
    s.set_episode(
        agent_cards=[_card(0), _card(1)],
        board_height=8,
        board_width=8,
        num_players=2,
        fps=4,
    )
    state = new_game(8, 2, seed=0)
    s.push_frame(state, heuristic_win_probs(state, default_evaluator()))
    snap1 = s.snapshot(client_episode=-1, client_frame=0)
    assert snap1["episode"] == 1
    init1 = cast(dict[str, object], snap1["init"])
    assert init1["num_players"] == 2
    assert init1["board_height"] == 8
    assert len(cast(list[object], snap1["frames"])) == 1

    # New episode wipes frames and bumps id.
    s.set_episode(
        agent_cards=[_card(0), _card(1)],
        board_height=8,
        board_width=8,
        num_players=2,
        fps=4,
    )
    snap2 = s.snapshot(client_episode=1, client_frame=1)
    assert snap2["episode"] == 2
    assert snap2["from_frame"] == 0
    assert snap2["frames"] == []


def test_snapshot_only_returns_new_frames_when_episode_matches() -> None:
    s = LiveServer(port=0)
    s.set_episode(
        agent_cards=[_card(0), _card(1)],
        board_height=8,
        board_width=8,
        num_players=2,
        fps=4,
    )
    state = new_game(8, 2, seed=0)
    probs = heuristic_win_probs(state, default_evaluator())
    for _ in range(3):
        s.push_frame(state, probs)

    # Cursor at 1 ⇒ get frames 1 and 2.
    snap = s.snapshot(client_episode=1, client_frame=1)
    assert snap["episode"] == 1
    assert snap["init"] is None
    frames = cast(list[object], snap["frames"])
    assert len(frames) == 2
    assert snap["from_frame"] == 1


def test_request_and_consume_reset_is_one_shot() -> None:
    s = LiveServer(port=0)
    assert s.consume_reset() is False
    s.request_reset()
    assert s.consume_reset() is True
    # Cleared after consume.
    assert s.consume_reset() is False


def test_play_and_serve_runs_one_game() -> None:
    """Smoke: ``loop=False`` plays a single game to completion."""
    from territory_takeover.search.random_agent import UniformRandomAgent

    server = LiveServer(port=0)
    cfg = LiveConfig(board_size=6, num_players=2, fps=1000, seed=0)

    def agent_factory(episode_index: int) -> list[object]:
        return [
            UniformRandomAgent(rng=np.random.default_rng(episode_index * 2), name="a"),
            UniformRandomAgent(rng=np.random.default_rng(episode_index * 2 + 1), name="b"),
        ]

    def card_factory(agents: list[object]) -> list[AgentCard]:
        return [_card(i) for i in range(len(agents))]

    play_and_serve(
        server=server,
        agent_factory=agent_factory,  # type: ignore[arg-type]
        agent_card_factory=card_factory,  # type: ignore[arg-type]
        config=cfg,
        loop=False,
    )
    snap = server.snapshot(client_episode=-1, client_frame=0)
    init = cast(dict[str, object], snap["init"])
    assert init["board_height"] == 6
    frames = cast(list[dict[str, object]], snap["frames"])
    assert len(frames) >= 2
    assert bool(frames[-1]["done"]) is True


def test_http_endpoints_serve_index_and_state() -> None:
    """Start the server, hit GET / and GET /state, verify shapes."""
    server = LiveServer(port=0, title="Smoke")
    server.set_episode(
        agent_cards=[_card(0), _card(1)],
        board_height=8,
        board_width=8,
        num_players=2,
        fps=4,
    )
    state = new_game(8, 2, seed=0)
    server.push_frame(state, heuristic_win_probs(state, default_evaluator()))
    server.start()
    try:
        with urllib.request.urlopen(server.url, timeout=2.0) as r:
            body = r.read().decode("utf-8")
            assert r.status == 200
            assert "<canvas" in body
            assert "Smoke" in body
        with urllib.request.urlopen(
            server.url + "state?episode=-1&since=0",
            timeout=2.0,
        ) as r:
            j = json.loads(r.read().decode("utf-8"))
            assert j["episode"] == 1
            assert isinstance(j["init"], dict)
            assert len(j["frames"]) == 1
        # POST /reset
        req = urllib.request.Request(server.url + "reset", method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as r:
            j = json.loads(r.read().decode("utf-8"))
            assert j["ok"] is True
        assert server.consume_reset() is True
    finally:
        server.stop()
