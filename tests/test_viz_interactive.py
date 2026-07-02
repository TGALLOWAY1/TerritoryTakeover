"""Tests for the interactive Arena server (``viz_interactive``).

These cover the front-end HTML structure, the agent-preset endpoint, live
play/pause/step/speed controls, agent substitution, and that real agents
actually advance a match to completion through the server.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from territory_takeover.viz_interactive import (
    AGENT_PRESETS,
    InteractiveServer,
    render_interactive_html,
)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return resp.status, resp.read().decode("utf-8")


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


@contextmanager
def _running_server() -> Iterator[InteractiveServer]:
    server = InteractiveServer(host="127.0.0.1", port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _full_state(base: str) -> dict:
    """Fetch the whole frame buffer (episode mismatch -> full snapshot)."""
    return _get_json(f"{base}/state?episode=-1&since=0")


def _wait_until(pred: Callable[[], bool], timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


# ---------------------------------------------------------------------------
# Static HTML / template
# ---------------------------------------------------------------------------

def test_render_html_has_arena_structure() -> None:
    html = render_interactive_html("Territory Takeover")
    # Header + branding.
    assert "Territory Takeover" in html
    assert "Claim territory. Cut off opponents. Win." in html
    # Core arena widgets.
    assert 'id="selbar"' in html          # swappable agent selector bar
    assert "<canvas" in html
    assert 'id="play-btn"' in html
    assert 'id="step-btn"' in html
    assert 'id="reset-btn"' in html
    assert 'id="speed-sel"' in html
    assert 'id="stats"' in html
    assert 'id="debug-drawer"' in html    # verification debug panel
    # Speed options from the spec.
    for label in ("0.5x", "1.0x", "2.0x", "5.0x", "20x"):
        assert label in html
    # Bottom navigation.
    for tab in ("Arena", "Agents", "History"):
        assert tab in html
    # Talks to the server.
    assert "/state" in html
    assert "/control" in html
    assert "/agents" in html
    assert "fetch(" in html


def test_render_html_escapes_title() -> None:
    html = render_interactive_html("<b>x</b>")
    assert "<b>x</b>" not in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def test_index_served() -> None:
    with _running_server() as server:
        status, body = _get_text(server.url)
        assert status == 200
        assert "<canvas" in body


def test_agents_endpoint_lists_presets() -> None:
    with _running_server() as server:
        data = _get_json(f"{server.url}agents")
        presets = data["presets"]
        keys = {p["key"] for p in presets}
        # The user explicitly required random + greedy to be available.
        assert "random" in keys
        assert "greedy" in keys
        assert keys == set(AGENT_PRESETS)
        for p in presets:
            assert p["label"]
            assert p["description"]


# ---------------------------------------------------------------------------
# Simulation: agents actually play
# ---------------------------------------------------------------------------

def test_match_advances_and_finishes() -> None:
    with _running_server() as server:
        base = server.url.rstrip("/")
        _post_json(
            f"{base}/start",
            {
                "board_size": 8,
                "num_players": 2,
                "fps": 8,
                "human_seat": None,
                "agents": {"0": "random", "1": "random"},
                "autoplay": True,
                "speed": 20,
            },
        )

        def has_many_frames() -> bool:
            return len(_full_state(base)["frames"]) > 3

        assert _wait_until(has_many_frames), "simulation did not advance"

        # Turn numbers must be monotonically non-decreasing across frames.
        frames = _full_state(base)["frames"]
        turns = [f["turn"] for f in frames]
        assert turns == sorted(turns)

        # Total territory (spawn + claimed cells) must grow as agents move.
        first, last = frames[0], frames[-1]
        owned0 = sum(first["territory"])
        owned1 = sum(last["territory"])
        assert owned1 > owned0

        def is_done() -> bool:
            fr = _full_state(base)["frames"]
            return bool(fr) and bool(fr[-1]["done"])

        assert _wait_until(is_done), "game never reached a terminal state"
        final = _full_state(base)["frames"][-1]
        assert "winner" in final
        # The game ends only when every player is dead (no reachable EMPTY).
        assert sum(1 for a in final["alive"] if a) == 0
        # Death keeps territory: every seat retains at least its spawn cell.
        assert all(t >= 1 for t in final["territory"])


# ---------------------------------------------------------------------------
# Controls: pause / step / play / speed
# ---------------------------------------------------------------------------

def test_step_and_pause_play_controls() -> None:
    with _running_server() as server:
        base = server.url.rstrip("/")
        _post_json(
            f"{base}/start",
            {
                "board_size": 14,
                "num_players": 2,
                "fps": 4,
                "human_seat": None,
                "agents": {"0": "random", "1": "random"},
                "autoplay": False,  # start paused
                "speed": 1,
            },
        )

        # Paused: only the initial frame, and no growth.
        assert _wait_until(lambda: len(_full_state(base)["frames"]) >= 1)
        state = _full_state(base)
        assert state["paused"] is True
        n0 = len(state["frames"])
        time.sleep(0.4)
        assert len(_full_state(base)["frames"]) == n0, "paused sim advanced on its own"

        # Step advances exactly one move.
        _post_json(f"{base}/control", {"cmd": "step"})
        assert _wait_until(lambda: len(_full_state(base)["frames"]) == n0 + 1)
        time.sleep(0.3)
        assert len(_full_state(base)["frames"]) == n0 + 1, "step advanced more than once"

        _post_json(f"{base}/control", {"cmd": "step"})
        assert _wait_until(lambda: len(_full_state(base)["frames"]) == n0 + 2)

        # Play resumes automatic stepping.
        _post_json(f"{base}/control", {"cmd": "play"})
        assert _wait_until(lambda: len(_full_state(base)["frames"]) > n0 + 4)
        assert _full_state(base)["paused"] is False

        # Pause halts growth again (only when the game is still running).
        _post_json(f"{base}/control", {"cmd": "pause"})
        time.sleep(0.3)
        snap = _full_state(base)
        assert snap["paused"] is True
        if not snap["frames"][-1]["done"]:
            n_pause = len(snap["frames"])
            time.sleep(0.4)
            assert len(_full_state(base)["frames"]) == n_pause


def test_speed_control_updates_state() -> None:
    with _running_server() as server:
        base = server.url.rstrip("/")
        _post_json(
            f"{base}/start",
            {
                "board_size": 12,
                "num_players": 2,
                "agents": {"0": "random", "1": "random"},
                "autoplay": False,
            },
        )
        assert _wait_until(lambda: len(_full_state(base)["frames"]) >= 1)
        _post_json(f"{base}/control", {"cmd": "speed", "speed": 5})
        assert _wait_until(lambda: abs(_full_state(base)["speed"] - 5.0) < 1e-6)


# ---------------------------------------------------------------------------
# Agent substitution
# ---------------------------------------------------------------------------

def test_substitution_changes_lineup_and_episode() -> None:
    with _running_server() as server:
        base = server.url.rstrip("/")
        _post_json(
            f"{base}/start",
            {
                "board_size": 10,
                "num_players": 2,
                "agents": {"0": "random", "1": "random"},
                "autoplay": False,
            },
        )
        assert _wait_until(lambda: _full_state(base)["init"] is not None)
        s0 = _full_state(base)
        ep0 = s0["episode"]
        assert s0["init"]["agents"][0]["strategy"] == "random"

        # Swap seat 0 to greedy -> new episode, updated label/strategy.
        _post_json(
            f"{base}/start",
            {
                "board_size": 10,
                "num_players": 2,
                "agents": {"0": "greedy", "1": "random"},
                "autoplay": False,
            },
        )
        assert _wait_until(lambda: _full_state(base)["episode"] != ep0)
        s1 = _full_state(base)
        assert s1["init"]["agents"][0]["strategy"] == "greedy"
        assert s1["init"]["agents"][0]["name"] == "Greedy"


def test_reset_control_restarts_match() -> None:
    with _running_server() as server:
        base = server.url.rstrip("/")
        cfg = {
            "board_size": 10,
            "num_players": 2,
            "agents": {"0": "random", "1": "random"},
            "autoplay": False,
        }
        _post_json(f"{base}/start", cfg)
        assert _wait_until(lambda: _full_state(base)["init"] is not None)
        ep0 = _full_state(base)["episode"]
        _post_json(f"{base}/control", {"cmd": "reset"})
        assert _wait_until(lambda: _full_state(base)["episode"] != ep0)
