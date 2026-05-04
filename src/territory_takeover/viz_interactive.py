"""Interactive HTTP server for human-vs-agent and spectator play.

Serves a browser UI that lets visitors either watch agents battle each other or
take control of seat 0 with arrow keys (Tron-style). The server exposes four
endpoints on top of the :mod:`territory_takeover.viz_live` frame-streaming
pattern:

- ``GET  /``        — setup + game page (single HTML file, no external assets)
- ``GET  /state``   — incremental frame poll (same shape as viz_live, plus
                      ``waiting_for_human`` and ``human_seat`` fields)
- ``POST /start``   — start a new game with a JSON config body
- ``POST /action``  — submit a human move (``{"action": 0|1|2|3}``)

Typical usage::

    from territory_takeover.viz_interactive import InteractiveServer
    server = InteractiveServer(port=8000)
    server.start()
    print(f"Open {server.url}")
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
"""

from __future__ import annotations

import contextlib
import html as _html
import json
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import parse_qs, urlsplit

import numpy as np

from territory_takeover.engine import new_game, step
from territory_takeover.eval.heuristic import default_evaluator
from territory_takeover.search.mcts.uct import UCTAgent
from territory_takeover.search.random_agent import HeuristicGreedyAgent, UniformRandomAgent
from territory_takeover.viz import HEAD_EDGE_COLORS, TILE_COLORS
from territory_takeover.viz_html import AgentCard, _frame_payload, heuristic_win_probs

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.search.agent import Agent
    from territory_takeover.state import GameState


# ---------------------------------------------------------------------------
# Agent presets
# ---------------------------------------------------------------------------

AGENT_PRESETS: Final[dict[str, dict[str, object]]] = {
    "random": {
        "label": "Random",
        "class": "UniformRandomAgent",
        "kwargs": {},
    },
    "greedy": {
        "label": "Greedy",
        "class": "HeuristicGreedyAgent",
        "kwargs": {},
    },
    "mcts-easy": {
        "label": "MCTS Easy  (~50 sims)",
        "class": "UCTAgent",
        "kwargs": {"iterations": 50},
    },
    "mcts-medium": {
        "label": "MCTS Medium (~200 sims)",
        "class": "UCTAgent",
        "kwargs": {"iterations": 200},
    },
    "mcts-hard": {
        "label": "MCTS Hard  (~800 sims)",
        "class": "UCTAgent",
        "kwargs": {"iterations": 800},
    },
}


def _build_agent(preset_key: str, seat: int, rng: np.random.Generator) -> Agent:
    """Construct an agent from a preset key."""
    if preset_key not in AGENT_PRESETS:
        preset_key = "greedy"
    preset = AGENT_PRESETS[preset_key]
    cls_name = str(preset["class"])
    kwargs: dict[str, object] = dict(preset["kwargs"])  # type: ignore[arg-type]
    name = f"p{seat + 1}-{preset_key}"
    if cls_name == "UniformRandomAgent":
        return UniformRandomAgent(rng=rng, name=name)  # type: ignore[return-value]
    if cls_name == "HeuristicGreedyAgent":
        return HeuristicGreedyAgent(rng=rng, name=name)  # type: ignore[return-value]
    if cls_name == "UCTAgent":
        return UCTAgent(rng=rng, name=name, **kwargs)  # type: ignore[return-value]
    raise ValueError(f"Unknown agent class {cls_name!r}")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class InteractiveServer:
    """HTTP server that streams live game frames and accepts human arrow-key input.

    Call :meth:`start` to launch the background HTTP thread, then POST ``/start``
    from the browser (or call :meth:`start_game` directly) to begin a game.
    The server is self-contained: no external assets, no additional threads
    beyond the HTTP server and the game-loop daemon.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        title: str = "TerritoryTakeover",
    ) -> None:
        self._host = host
        self._port = port
        self._title = title

        # Frame buffer shared with the HTTP handler (guarded by _lock).
        self._lock = threading.Lock()
        self._episode: int = 0
        self._frames: list[dict[str, object]] = []
        self._init: dict[str, object] = {}

        # Signals the game loop to stop (set by /reset or start_game).
        self._reset_event = threading.Event()

        # Human-turn synchronisation.
        # maxsize=1 so stale inputs cannot accumulate; the latest keypress wins.
        self._action_queue: queue.Queue[int] = queue.Queue(maxsize=1)
        self._waiting_for_human: bool = False
        self._human_seat: int | None = None

        # HTTP + game threads.
        self._httpd: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._game_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    # ------------------------------------------------------------------
    # Frame buffer helpers (called from the game-loop thread)
    # ------------------------------------------------------------------

    def _set_episode(
        self,
        agent_cards: list[AgentCard],
        board_height: int,
        board_width: int,
        num_players: int,
        fps: int,
    ) -> None:
        agents_payload: list[dict[str, object]] = [
            {"seat": c.seat, "name": c.name, "strategy": c.strategy, "elo": c.elo}
            for c in agent_cards
        ]
        with self._lock:
            self._episode += 1
            self._frames = []
            self._init = {
                "title": self._title,
                "board_height": board_height,
                "board_width": board_width,
                "num_players": num_players,
                "fps": fps,
                "tile_colors": list(TILE_COLORS),
                "head_edge_colors": list(HEAD_EDGE_COLORS),
                "agents": agents_payload,
            }

    def _push_frame(self, state: GameState, win_probs: NDArray[np.float64]) -> None:
        frame = _frame_payload(state, win_probs)
        with self._lock:
            self._frames.append(frame)

    # ------------------------------------------------------------------
    # Human action API (called from the HTTP handler thread)
    # ------------------------------------------------------------------

    def queue_action(self, action: int) -> bool:
        """Enqueue a human move. Returns True only if it is the human's turn."""
        if not self._waiting_for_human:
            return False
        # Replace any pending action with the newer one (latest keypress wins).
        with contextlib.suppress(queue.Empty):
            self._action_queue.get_nowait()
        with contextlib.suppress(queue.Full):
            self._action_queue.put_nowait(action)
        return True

    # ------------------------------------------------------------------
    # Snapshot (called from the HTTP handler thread)
    # ------------------------------------------------------------------

    def snapshot(self, client_episode: int, client_frame: int) -> dict[str, object]:
        """Return new frames and metadata for the polling browser client."""
        with self._lock:
            if client_episode != self._episode:
                return {
                    "episode": self._episode,
                    "init": dict(self._init),
                    "frames": list(self._frames),
                    "from_frame": 0,
                    "waiting_for_human": self._waiting_for_human,
                    "human_seat": self._human_seat,
                }
            tail = (
                self._frames[client_frame:]
                if client_frame >= 0
                else list(self._frames)
            )
            return {
                "episode": self._episode,
                "init": None,
                "frames": list(tail),
                "from_frame": max(0, client_frame),
                "waiting_for_human": self._waiting_for_human,
                "human_seat": self._human_seat,
            }

    # ------------------------------------------------------------------
    # Game management
    # ------------------------------------------------------------------

    def start_game(self, config: dict[str, object]) -> None:
        """Stop any running game and launch a new one from *config*."""
        self._reset_event.set()
        if self._game_thread is not None and self._game_thread.is_alive():
            self._game_thread.join(timeout=3.0)
        self._reset_event.clear()

        # Drain stale queued actions.
        while True:
            try:
                self._action_queue.get_nowait()
            except queue.Empty:
                break

        self._waiting_for_human = False
        raw_hs = config.get("human_seat")
        self._human_seat = None if raw_hs is None else int(raw_hs)  # type: ignore[arg-type]

        self._game_thread = threading.Thread(
            target=self._run_game,
            args=(config,),
            daemon=True,
            name="tt-game-loop",
        )
        self._game_thread.start()

    def _run_game(self, config: dict[str, object]) -> None:
        board_size = int(config.get("board_size", 15))  # type: ignore[arg-type]
        num_players = int(config.get("num_players", 2))  # type: ignore[arg-type]
        fps = int(config.get("fps", 4))  # type: ignore[arg-type]
        raw_hs = config.get("human_seat")
        human_seat: int | None = None if raw_hs is None else int(raw_hs)  # type: ignore[arg-type]
        agents_cfg: dict[str, str] = {
            str(k): str(v)
            for k, v in (config.get("agents") or {}).items()  # type: ignore[union-attr]
        }

        evaluator = default_evaluator()

        def win_probs(s: GameState) -> NDArray[np.float64]:
            return heuristic_win_probs(s, evaluator)

        rng = np.random.default_rng()
        agents: list[Agent | None] = []
        agent_cards: list[AgentCard] = []

        for seat in range(num_players):
            if seat == human_seat:
                agents.append(None)
                agent_cards.append(
                    AgentCard(seat=seat, name="You", strategy="human", elo=None)
                )
            else:
                preset_key = agents_cfg.get(str(seat), "greedy")
                agent = _build_agent(
                    preset_key,
                    seat,
                    np.random.default_rng(int(rng.integers(2**32))),
                )
                agents.append(agent)
                preset = AGENT_PRESETS.get(preset_key, AGENT_PRESETS["greedy"])
                agent_cards.append(
                    AgentCard(
                        seat=seat,
                        name=getattr(agent, "name", f"p{seat + 1}"),
                        strategy=str(preset["label"]),
                        elo=None,
                    )
                )

        state = new_game(board_size=board_size, num_players=num_players)
        self._set_episode(
            agent_cards=agent_cards,
            board_height=int(state.grid.shape[0]),
            board_width=int(state.grid.shape[1]),
            num_players=num_players,
            fps=fps,
        )
        self._push_frame(state, win_probs(state))

        for a in agents:
            if a is not None:
                a.reset()

        # In watch mode delay frames so the browser can keep up.
        # In play mode the human's pace drives the tempo; AI responds immediately.
        ai_delay = 0.0 if human_seat is not None else 1.0 / max(1, fps)

        while not state.done:
            if self._reset_event.is_set():
                return

            seat = state.current_player

            if seat == human_seat:
                self._waiting_for_human = True
                action: int | None = None
                while action is None:
                    if self._reset_event.is_set():
                        self._waiting_for_human = False
                        return
                    with contextlib.suppress(queue.Empty):
                        action = self._action_queue.get(timeout=0.05)
                self._waiting_for_human = False
            else:
                ai = agents[seat]
                assert ai is not None
                action = ai.select_action(state, seat)
                if ai_delay > 0.0:
                    time.sleep(ai_delay)

            step(state, action, strict=False)
            self._push_frame(state, win_probs(state))

        # Brief pause on the final frame so the winner banner is visible.
        for _ in range(max(1, fps) * 2):
            if self._reset_event.is_set():
                return
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # HTTP lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the HTTP server in a background daemon thread."""
        handler_cls = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self._host, self._port), handler_cls)
        self._port = self._httpd.server_address[1]
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="tt-interactive-http",
        )
        self._http_thread.start()

    def stop(self) -> None:
        """Signal the game loop to exit and shut down the HTTP server."""
        self._reset_event.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
            self._http_thread = None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _make_handler(server: InteractiveServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: ANN401
            del format, args

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._serve_index()
            elif path == "/state":
                self._serve_state()
            elif path == "/agents":
                presets = [
                    {"key": k, "label": v["label"]}
                    for k, v in AGENT_PRESETS.items()
                ]
                self._send_json(HTTPStatus.OK, {"presets": presets})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path == "/start":
                body = self._read_json_body()
                if body is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
                    return
                server.start_game(body)
                self._send_json(HTTPStatus.OK, {"ok": True})
            elif path == "/action":
                body = self._read_json_body()
                if body is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
                    return
                raw = body.get("action")
                if not isinstance(raw, int) or raw not in (0, 1, 2, 3):
                    self._send_json(
                        HTTPStatus.BAD_REQUEST, {"error": "action must be 0-3"}
                    )
                    return
                queued = server.queue_action(raw)
                self._send_json(HTTPStatus.OK, {"ok": True, "queued": queued})
            elif path == "/reset":
                server._reset_event.set()
                self._send_json(HTTPStatus.OK, {"ok": True})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _read_json_body(self) -> dict[str, object] | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data: object = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(data, dict):
                return None
            return data  # type: ignore[return-value]

        def _serve_index(self) -> None:
            body = render_interactive_html(server._title).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_state(self) -> None:
            qs = parse_qs(urlsplit(self.path).query)
            try:
                ep = int(qs.get("episode", ["-1"])[0])
                fr = int(qs.get("since", ["0"])[0])
            except (ValueError, IndexError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad params"})
                return
            self._send_json(HTTPStatus.OK, server.snapshot(ep, fr))

        def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_TITLE_MARKER: Final[str] = "__TT_ITITLE__"


def render_interactive_html(title: str) -> str:
    """Render the interactive page with *title* injected."""
    return _TEMPLATE.replace(_TITLE_MARKER, _html.escape(title))


_TEMPLATE: Final[str] = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>__TT_ITITLE__</title>
<style>
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{
  margin:0;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0f1115;color:#e6e6e6;
}

/* ── setup overlay ─────────────────────────────────────────────────────── */
#setup-overlay{
  position:fixed;inset:0;
  background:rgba(10,11,15,0.95);
  display:flex;align-items:center;justify-content:center;
  z-index:100;
  transition:opacity .18s;
}
#setup-overlay.hidden{opacity:0;pointer-events:none}
.setup-box{
  background:#1a1d23;
  border:1px solid #2a2f38;border-radius:8px;
  padding:28px 32px;width:380px;max-width:96vw;
}
.setup-box h2{font-size:18px;margin:0 0 20px;letter-spacing:.3px}
.sf{margin-bottom:16px}
.sf-label{
  font-size:11px;color:#9aa0a6;
  text-transform:uppercase;letter-spacing:.7px;
  margin-bottom:6px;
}
.seg-row{display:flex}
.seg{
  flex:1;background:#22252c;color:#9aa0a6;
  border:1px solid #3a4049;
  padding:6px 0;font-size:13px;cursor:pointer;
  transition:background .12s,color .12s;
}
.seg:first-child{border-radius:4px 0 0 4px}
.seg:last-child{border-radius:0 4px 4px 0}
.seg:not(:first-child){border-left-width:0}
.seg.on{background:#2e3340;color:#e6e6e6;border-color:#5a6275}
.seat-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.seat-label{font-size:12px;color:#9aa0a6;width:110px;flex-shrink:0}
.seat-val{font-size:13px;color:#c4c7cc}
select.agent-sel{
  flex:1;background:#22252c;color:#e6e6e6;
  border:1px solid #3a4049;border-radius:4px;
  padding:5px 8px;font-size:13px;cursor:pointer;
}
#start-btn{
  width:100%;padding:10px;font-size:14px;
  background:#2563eb;border:1px solid #1d4ed8;color:#fff;
  border-radius:4px;cursor:pointer;margin-top:8px;
  transition:background .12s;
}
#start-btn:hover{background:#1d4ed8}

/* ── game area ──────────────────────────────────────────────────────────── */
.wrap{display:flex;flex-wrap:wrap;gap:24px;padding:24px}
.board-col{flex:1 1 420px;min-width:320px}
.side-col{flex:1 1 300px;min-width:280px;max-width:460px}
h1{font-size:20px;margin:0 0 10px;letter-spacing:.3px}
.meta{font-size:12px;color:#9aa0a6;margin-bottom:10px}
canvas{
  width:100%;height:auto;
  background:#1a1d23;border:1px solid #2a2f38;
  border-radius:6px;image-rendering:pixelated;display:block;
}
.controls{display:flex;align-items:center;gap:10px;margin-top:12px;flex-wrap:wrap}
button{
  background:#2a2f38;color:#e6e6e6;
  border:1px solid #3a4049;border-radius:4px;
  padding:6px 12px;cursor:pointer;font-size:13px;
  transition:background .12s;
}
button:hover{background:#353b45}
.turn-ro{font-size:12px;color:#c4c7cc;min-width:100px}
.live-pill{
  font-size:11px;padding:2px 8px;border-radius:999px;
  background:#2a2f38;color:#9aa0a6;
  text-transform:uppercase;letter-spacing:.6px;
}
.live-pill.on{background:#1f3a1f;color:#b8e6b8}
.live-pill.off{background:#3a1f1f;color:#e6b8b8}

/* ── your-turn banner ───────────────────────────────────────────────────── */
#your-turn{
  display:none;margin-top:10px;
  padding:9px 14px;border-radius:6px;
  background:#0e2419;border:1px solid #2d6a42;color:#a8e6be;
  font-size:13px;
  animation:pulse-border 1.1s ease-in-out infinite;
}
#your-turn.show{display:block}
@keyframes pulse-border{
  0%,100%{border-color:#2d6a42}
  50%{border-color:#5aad7a;box-shadow:0 0 8px rgba(90,173,122,.25)}
}

/* ── key hint strip ─────────────────────────────────────────────────────── */
#key-hint{
  display:none;margin-top:6px;
  font-size:11px;color:#6a7280;letter-spacing:.3px;
}
#key-hint.show{display:block}

/* ── winner banner ──────────────────────────────────────────────────────── */
.w-banner{
  margin-top:12px;padding:10px 12px;
  border-radius:6px;background:#1a1d23;
  border:1px solid #3a4049;font-size:14px;
}
.w-banner.active{border-color:#f5c542;color:#fff5cf}

/* ── agent cards ─────────────────────────────────────────────────────────── */
.agent-card{
  background:#1a1d23;border:1px solid #2a2f38;
  border-left-width:6px;border-radius:6px;
  padding:10px 12px;margin-bottom:10px;
}
.agent-card.current{box-shadow:0 0 0 1px #f5c542 inset}
.agent-card.human-card{box-shadow:0 0 0 1px #5aad7a inset}
.a-head{display:flex;align-items:baseline;justify-content:space-between;gap:8px}
.a-name{font-weight:600;font-size:14px}
.a-strat{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:#9aa0a6}
.a-stats{
  display:grid;grid-template-columns:repeat(3,1fr);
  gap:4px 8px;font-size:12px;margin-top:6px;color:#c4c7cc;
}
.a-stats .k{color:#9aa0a6;font-size:11px}
.prob-bar{
  margin-top:8px;height:14px;background:#2a2f38;
  border-radius:3px;overflow:hidden;position:relative;
}
.prob-fill{height:100%;width:0%;transition:width 220ms ease-out}
.prob-label{
  position:absolute;top:-1px;right:6px;
  font-size:11px;line-height:16px;color:#e6e6e6;
  text-shadow:0 1px 2px rgba(0,0,0,.4);
}
.dead{opacity:.42}
</style>
</head>
<body>

<!-- ═══════════ SETUP OVERLAY ═══════════ -->
<div id="setup-overlay">
  <div class="setup-box">
    <h2>__TT_ITITLE__</h2>

    <div class="sf">
      <div class="sf-label">Mode</div>
      <div class="seg-row" id="mode-row">
        <button class="seg on" data-val="play">Play (arrow keys)</button>
        <button class="seg" data-val="watch">Watch agents</button>
      </div>
    </div>

    <div class="sf">
      <div class="sf-label">Board size</div>
      <div class="seg-row" id="board-row">
        <button class="seg" data-val="10">10</button>
        <button class="seg on" data-val="15">15</button>
        <button class="seg" data-val="20">20</button>
        <button class="seg" data-val="30">30</button>
      </div>
    </div>

    <div class="sf">
      <div class="sf-label">Players</div>
      <div class="seg-row" id="players-row">
        <button class="seg on" data-val="2">2</button>
        <button class="seg" data-val="3">3</button>
        <button class="seg" data-val="4">4</button>
      </div>
    </div>

    <div class="sf" id="seat-box"></div>

    <button id="start-btn">Start Game</button>
  </div>
</div>

<!-- ═══════════ GAME AREA ═══════════ -->
<div class="wrap">
  <div class="board-col">
    <h1>__TT_ITITLE__</h1>
    <div class="meta" id="meta">Configure a game above to begin.</div>
    <canvas id="board" width="360" height="360"></canvas>
    <div class="controls">
      <button id="new-game-btn">New Game</button>
      <button id="restart-btn" disabled>Restart</button>
      <span class="turn-ro" id="turn-ro"></span>
      <span class="live-pill off" id="pill">offline</span>
    </div>
    <div id="your-turn">&#x2190;&#x2191;&#x2192;&#x2193; Your turn &mdash; press an arrow key!</div>
    <div id="key-hint">&#x2191; N &nbsp;&#x2193; S &nbsp;&#x2190; W &nbsp;&#x2192; E</div>
    <div class="w-banner" id="w-banner">Waiting for game&hellip;</div>
  </div>
  <div class="side-col" id="side-col"></div>
</div>

<script>
(function () {
"use strict";

/* ══════════════════════════════════════════════════════════════════
   SETUP FORM
   ══════════════════════════════════════════════════════════════════ */

var F = { mode: "play", boardSize: 15, numPlayers: 2 };
var currentConfig = null;  // last successfully started config

var PRESET_KEYS  = ["random","greedy","mcts-easy","mcts-medium","mcts-hard"];
var PRESET_LABELS= ["Random","Greedy",
  "MCTS Easy (~50 sims)","MCTS Medium (~200 sims)","MCTS Hard (~800 sims)"];

function makeSeg(rowId, onChange) {
  var row = document.getElementById(rowId);
  var btns = row ? Array.prototype.slice.call(row.querySelectorAll(".seg")) : [];
  btns.forEach(function (b) {
    b.addEventListener("click", function () {
      btns.forEach(function (x) { x.classList.remove("on"); });
      b.classList.add("on");
      if (onChange) onChange(b.dataset.val);
    });
  });
}

function renderSeatBox() {
  var box = document.getElementById("seat-box");
  box.innerHTML = "";
  var start = F.mode === "play" ? 1 : 0;
  if (F.mode === "play") {
    var youRow = document.createElement("div");
    youRow.className = "seat-row";
    youRow.innerHTML = '<span class="seat-label">Seat 1 — You</span>'
      + '<span class="seat-val" style="color:#5aad7a">Human · arrow keys</span>';
    box.appendChild(youRow);
  }
  for (var s = start; s < F.numPlayers; s++) {
    var row = document.createElement("div");
    row.className = "seat-row";
    var label = F.mode === "play"
      ? "Seat " + (s + 1) + " — Opp"
      : "Seat " + (s + 1);
    var sel = document.createElement("select");
    sel.className = "agent-sel";
    sel.dataset.seat = String(s);
    for (var i = 0; i < PRESET_KEYS.length; i++) {
      var opt = document.createElement("option");
      opt.value = PRESET_KEYS[i];
      opt.textContent = PRESET_LABELS[i];
      if (PRESET_KEYS[i] === "greedy") opt.selected = true;
      sel.appendChild(opt);
    }
    row.innerHTML = '<span class="seat-label">' + esc(label) + '</span>';
    row.appendChild(sel);
    box.appendChild(row);
  }
}

function buildConfig() {
  var agents = {};
  var start = F.mode === "play" ? 1 : 0;
  for (var s = start; s < F.numPlayers; s++) {
    var sel = document.querySelector('.agent-sel[data-seat="' + s + '"]');
    agents[String(s)] = sel ? sel.value : "greedy";
  }
  return {
    mode: F.mode,
    board_size: F.boardSize,
    num_players: F.numPlayers,
    fps: F.mode === "play" ? 8 : 4,
    human_seat: F.mode === "play" ? 0 : null,
    agents: agents
  };
}

makeSeg("mode-row", function (v) {
  F.mode = v;
  renderSeatBox();
  updateKeyHint();
});
makeSeg("board-row", function (v) { F.boardSize = parseInt(v, 10); });
makeSeg("players-row", function (v) {
  F.numPlayers = parseInt(v, 10);
  renderSeatBox();
});
renderSeatBox();

document.getElementById("start-btn").addEventListener("click", function () {
  var cfg = buildConfig();
  fetch("/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
    cache: "no-store"
  })
  .then(function (r) { return r.json(); })
  .then(function (j) {
    if (j.ok) {
      currentConfig = cfg;
      document.getElementById("setup-overlay").classList.add("hidden");
      document.getElementById("restart-btn").disabled = false;
      updateKeyHint();
    }
  })
  .catch(function (e) { console.error("start failed", e); });
});

document.getElementById("new-game-btn").addEventListener("click", function () {
  document.getElementById("setup-overlay").classList.remove("hidden");
});

document.getElementById("restart-btn").addEventListener("click", function () {
  if (!currentConfig) return;
  fetch("/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentConfig),
    cache: "no-store"
  }).catch(function (e) { console.error("restart failed", e); });
});

function updateKeyHint() {
  var hint = document.getElementById("key-hint");
  if (currentConfig && currentConfig.human_seat !== null) {
    hint.classList.add("show");
  } else {
    hint.classList.remove("show");
  }
}

/* ══════════════════════════════════════════════════════════════════
   GAME CLIENT
   ══════════════════════════════════════════════════════════════════ */

var EPISODE = -1;
var FRAMES  = [];
var INIT    = null;
var CARDS   = [];
var CELL    = 24;

var waitingForHuman  = false;
var humanSeat        = null;
var actionSubmitted  = false;

var canvas    = document.getElementById("board");
var ctx       = canvas.getContext("2d");
var meta      = document.getElementById("meta");
var wBanner   = document.getElementById("w-banner");
var turnRo    = document.getElementById("turn-ro");
var pill      = document.getElementById("pill");
var sideCol   = document.getElementById("side-col");
var yourTurn  = document.getElementById("your-turn");

function esc(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];
  });
}

function setPill(state) {
  pill.classList.remove("on","off");
  if (state === "live") { pill.classList.add("on"); pill.textContent = "live"; }
  else { pill.classList.add("off"); pill.textContent = "offline"; }
}

/* ── canvas drawing ─────────────────────────────────────────────────── */
function drawFrame(f) {
  if (!INIT) return;
  var W = INIT.board_width, H = INIT.board_height;
  var TILE = INIT.tile_colors, EDGE = INIT.head_edge_colors;
  ctx.fillStyle = TILE[0];
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (var r = 0; r < H; r++) {
    for (var c = 0; c < W; c++) {
      var v = f.grid[r * W + c];
      if (v === 0) continue;
      ctx.fillStyle = TILE[v];
      ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
    }
  }
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (var i = 1; i < W; i++) {
    ctx.beginPath();
    ctx.moveTo(i * CELL + .5, 0);
    ctx.lineTo(i * CELL + .5, canvas.height);
    ctx.stroke();
  }
  for (var j = 1; j < H; j++) {
    ctx.beginPath();
    ctx.moveTo(0, j * CELL + .5);
    ctx.lineTo(canvas.width, j * CELL + .5);
    ctx.stroke();
  }
  for (var p = 0; p < INIT.num_players; p++) {
    var head = f.heads[p];
    if (head[0] < 0 || !f.alive[p]) continue;
    ctx.strokeStyle = EDGE[p];
    ctx.lineWidth = 3;
    ctx.strokeRect(head[1]*CELL+1.5, head[0]*CELL+1.5, CELL-3, CELL-3);
  }
}

/* ── side panel ─────────────────────────────────────────────────────── */
function buildSidePanel(init) {
  sideCol.innerHTML = "";
  CARDS = [];
  for (var s = 0; s < init.num_players; s++) {
    var agent = init.agents[s];
    var pc = init.tile_colors[s + 1];
    var elo = (agent.elo === null || agent.elo === undefined)
      ? "—" : (agent.elo >= 0 ? "+" : "") + Math.round(agent.elo);
    var isHuman = (agent.strategy === "human");
    var card = document.createElement("div");
    card.className = "agent-card" + (isHuman ? " human-card" : "");
    card.style.borderLeftColor = pc;
    card.innerHTML =
      '<div class="a-head">'
        + '<span class="a-name">' + esc(agent.name) + '</span>'
        + '<span class="a-strat">' + esc(agent.strategy) + '</span>'
      + '</div>'
      + '<div class="a-stats">'
        + '<div><div class="k">Seat</div>' + (s+1) + '</div>'
        + '<div><div class="k">Elo</div>' + elo + '</div>'
        + '<div><div class="k">Status</div><span class="st">alive</span></div>'
        + '<div><div class="k">Claimed</div><span class="cl">0</span></div>'
        + '<div><div class="k">Path</div><span class="pl">0</span></div>'
        + '<div><div class="k">Win prob</div><span class="pp">0%</span></div>'
      + '</div>'
      + '<div class="prob-bar">'
        + '<div class="prob-fill" style="background:' + pc + '"></div>'
      + '</div>';
    sideCol.appendChild(card);
    CARDS.push({
      root: card,
      st:  card.querySelector(".st"),
      cl:  card.querySelector(".cl"),
      pl:  card.querySelector(".pl"),
      pp:  card.querySelector(".pp"),
      fill:card.querySelector(".prob-fill")
    });
  }
}

function updateSide(f) {
  if (!INIT) return;
  for (var s = 0; s < INIT.num_players; s++) {
    var c = CARDS[s];
    if (!c) continue;
    c.cl.textContent = f.claimed[s];
    c.pl.textContent = f.path_len[s];
    var pct = Math.round(f.win_probs[s] * 100);
    c.pp.textContent = pct + "%";
    c.fill.style.width = pct + "%";
    if (f.alive[s]) {
      c.st.textContent = "alive";
      c.root.classList.remove("dead");
    } else {
      c.st.textContent = "dead";
      c.root.classList.add("dead");
    }
    if (s === f.current_player && !f.done) {
      c.root.classList.add("current");
    } else {
      c.root.classList.remove("current");
    }
  }
}

/* ── applyInit / applyFrame ─────────────────────────────────────────── */
function applyInit(init) {
  INIT = init;
  canvas.width  = init.board_width  * CELL;
  canvas.height = init.board_height * CELL;
  meta.textContent = init.board_height + "x" + init.board_width
      + " board · " + init.num_players + " players";
  wBanner.classList.remove("active");
  wBanner.textContent = "Game in progress…";
  buildSidePanel(init);
}

function applyFrame(f) {
  drawFrame(f);
  updateSide(f);
  turnRo.textContent = "Turn " + f.turn + " · Frame " + FRAMES.length;
  if (f.done) {
    wBanner.classList.add("active");
    if (f.winner === null || f.winner === undefined) {
      wBanner.textContent = "Game over — tie.";
    } else {
      wBanner.textContent = "Winner: " + esc(INIT.agents[f.winner].name)
          + " (seat " + (f.winner + 1) + ")";
    }
  } else {
    wBanner.classList.remove("active");
    wBanner.textContent = "Game in progress…";
  }
}

/* ── polling ────────────────────────────────────────────────────────── */
function poll() {
  var url = "/state?episode=" + EPISODE + "&since=" + FRAMES.length;
  fetch(url, { cache: "no-store" })
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (j) {
      setPill("live");
      if (j.episode !== EPISODE) {
        EPISODE = j.episode;
        FRAMES  = [];
        humanSeat = (j.human_seat === null || j.human_seat === undefined)
            ? null : j.human_seat;
        if (j.init) applyInit(j.init);
      }
      for (var i = 0; i < j.frames.length; i++) {
        FRAMES.push(j.frames[i]);
        applyFrame(j.frames[i]);
      }
      var prevWaiting = waitingForHuman;
      waitingForHuman = !!j.waiting_for_human;
      if (!waitingForHuman) actionSubmitted = false;
      if (waitingForHuman && !prevWaiting) {
        yourTurn.classList.add("show");
      } else if (!waitingForHuman) {
        yourTurn.classList.remove("show");
      }
    })
    .catch(function () { setPill("offline"); })
    .then(function () { setTimeout(poll, 150); });
}

/* ── arrow-key handler ──────────────────────────────────────────────── */
// DIRECTIONS = (N=0, S=1, W=2, E=3) matching engine constants.py
var KEY_MAP = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 };

document.addEventListener("keydown", function (e) {
  if (!waitingForHuman || actionSubmitted) return;
  var action = KEY_MAP[e.key];
  if (action === undefined) return;
  e.preventDefault();
  actionSubmitted = true;
  yourTurn.classList.remove("show");
  fetch("/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: action }),
    cache: "no-store"
  }).catch(function () {
    // Allow retry if the POST failed.
    actionSubmitted = false;
  });
});

setPill("offline");
poll();
})();
</script>
</body>
</html>
"""


__all__ = [
    "AGENT_PRESETS",
    "InteractiveServer",
    "render_interactive_html",
]
