"""Live HTTP viewer for an in-progress game.

Where :mod:`territory_takeover.viz_html` writes a self-contained HTML file
that replays a recorded trajectory, this module hosts a tiny stdlib HTTP
server that streams game frames to a browser as they are produced. Open the
served URL, watch agents play in real time, click "Reset" to start a fresh
match.

No external dependencies: :mod:`http.server` + :mod:`threading` only. The
browser polls ``/state`` for incremental frames; each frame uses the same
JSON shape as :mod:`territory_takeover.viz_html` so the rendering code is
identical in spirit.

Typical usage::

    from territory_takeover.viz_live import LiveServer, play_and_serve
    server = LiveServer(host="127.0.0.1", port=8000)
    server.start()
    play_and_serve(server, agent_factory=..., agent_card_factory=..., ...)
"""

from __future__ import annotations

import html
import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import parse_qs, urlsplit

import numpy as np

from territory_takeover.engine import new_game, step
from territory_takeover.eval.heuristic import default_evaluator
from territory_takeover.viz import HEAD_EDGE_COLORS, TILE_COLORS
from territory_takeover.viz_html import (
    AgentCard,
    _frame_payload,
    heuristic_win_probs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from territory_takeover.search.agent import Agent
    from territory_takeover.state import GameState


@dataclass(frozen=True, slots=True)
class LiveConfig:
    """Per-episode knobs for :func:`play_and_serve`."""

    board_size: int = 20
    num_players: int = 2
    fps: int = 4
    seed: int = 0


class LiveServer:
    """HTTP server that streams game frames to connected browsers.

    The server holds a per-episode buffer of frames. Browsers poll
    ``GET /state?episode=<int>&since=<int>`` and receive any frames produced
    after the supplied cursor; an episode-id mismatch signals a reset. A
    ``POST /reset`` request flips an internal flag that the game loop
    consumes via :meth:`consume_reset` between moves.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        title: str = "TerritoryTakeover Live",
    ) -> None:
        self._host = host
        self._port = port
        self._title = title
        self._lock = threading.Lock()
        self._episode: int = 0
        self._frames: list[dict[str, object]] = []
        self._init: dict[str, object] = {}
        self._reset_event = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    @property
    def title(self) -> str:
        return self._title

    def set_episode(
        self,
        agent_cards: list[AgentCard],
        board_height: int,
        board_width: int,
        num_players: int,
        fps: int,
    ) -> None:
        """Begin a new episode: bump the episode id and reset the frame buffer."""
        agents_payload: list[dict[str, object]] = []
        for card in agent_cards:
            elo: float | None = None if card.elo is None else float(card.elo)
            agents_payload.append(
                {
                    "seat": int(card.seat),
                    "name": card.name,
                    "strategy": card.strategy,
                    "elo": elo,
                }
            )
        with self._lock:
            self._episode += 1
            self._frames = []
            self._init = {
                "title": self._title,
                "board_height": int(board_height),
                "board_width": int(board_width),
                "num_players": int(num_players),
                "fps": int(fps),
                "tile_colors": list(TILE_COLORS),
                "head_edge_colors": list(HEAD_EDGE_COLORS),
                "agents": agents_payload,
            }

    def push_frame(self, state: GameState, win_probs: NDArray[np.float64]) -> None:
        """Append one frame to the current episode's buffer."""
        frame = _frame_payload(state, win_probs)
        with self._lock:
            self._frames.append(frame)

    def request_reset(self) -> None:
        """Mark a reset as pending; the game loop will see it on the next check."""
        self._reset_event.set()

    def consume_reset(self) -> bool:
        """Return whether a reset is pending and clear the flag."""
        if self._reset_event.is_set():
            self._reset_event.clear()
            return True
        return False

    def snapshot(self, client_episode: int, client_frame: int) -> dict[str, object]:
        """Return frames newer than the client's cursor.

        If ``client_episode`` does not match the server's current episode, the
        full ``init`` payload and every frame so far are returned with
        ``from_frame=0``. Otherwise only frames with index ``>= client_frame``
        are returned and ``init`` is ``None`` to save bytes on the wire.
        """
        with self._lock:
            if client_episode != self._episode:
                return {
                    "episode": self._episode,
                    "init": dict(self._init),
                    "frames": list(self._frames),
                    "from_frame": 0,
                }
            tail = self._frames[client_frame:] if client_frame >= 0 else list(self._frames)
            return {
                "episode": self._episode,
                "init": None,
                "frames": list(tail),
                "from_frame": max(0, client_frame),
            }

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        handler_cls = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self._host, self._port), handler_cls)
        # Bound port is useful when the caller passed port=0.
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="tt-live-http",
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down the server and join the background thread."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _make_handler(server: LiveServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # Silence the default per-request stderr log; the script prints its
        # own one-line "serving at..." message and noisy access logs would
        # bury the gameplay output. The ``format`` parameter name and
        # ``*args: Any`` signature mirror :class:`BaseHTTPRequestHandler`.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: ANN401
            del format, args
            return

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._serve_index()
                return
            if path == "/state":
                self._serve_state()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path == "/reset":
                server.request_reset()
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _serve_index(self) -> None:
            body = render_live_html(server.title).encode("utf-8")
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
            except ValueError:
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


def render_live_html(title: str) -> str:
    """Render the live-mode HTML page with ``title`` interpolated into the head."""
    safe = html.escape(title)
    return _LIVE_TEMPLATE.replace("__TT_LIVE_TITLE__", safe)


_DEFAULT_LIVE_CONFIG: Final[LiveConfig] = LiveConfig()


def play_and_serve(
    server: LiveServer,
    agent_factory: Callable[[int], list[Agent]],
    agent_card_factory: Callable[[list[Agent]], list[AgentCard]],
    config: LiveConfig = _DEFAULT_LIVE_CONFIG,
    win_prob_fn: Callable[[GameState], NDArray[np.float64]] | None = None,
    loop: bool = True,
) -> None:
    """Drive the game loop and feed frames into ``server``.

    ``agent_factory(episode_index) -> list[Agent]`` builds a fresh agent
    roster per episode (so RNG state restarts cleanly on reset). The default
    ``win_prob_fn`` is a softmax over the heuristic evaluator. If ``loop`` is
    ``True``, the function plays games forever — pause with ``POST /reset``
    or interrupt the process. ``loop=False`` plays a single game then returns.
    """
    if win_prob_fn is None:
        evaluator = default_evaluator()

        def _default_win_probs(s: GameState) -> NDArray[np.float64]:
            return heuristic_win_probs(s, evaluator)

        win_prob_fn = _default_win_probs

    episode_index = 0
    while True:
        agents = agent_factory(episode_index)
        cards = agent_card_factory(agents)
        if len(cards) != config.num_players:
            raise ValueError(
                f"agent_card_factory returned {len(cards)} cards; "
                f"expected {config.num_players}"
            )

        ss = np.random.SeedSequence(config.seed + episode_index)
        game_seed = int(ss.generate_state(1, dtype=np.uint32)[0])
        state = new_game(
            board_size=config.board_size,
            num_players=config.num_players,
            seed=game_seed,
        )
        server.set_episode(
            agent_cards=cards,
            board_height=state.grid.shape[0],
            board_width=state.grid.shape[1],
            num_players=config.num_players,
            fps=config.fps,
        )
        server.push_frame(state, win_prob_fn(state))

        delay = 1.0 / max(1, config.fps)
        for a in agents:
            a.reset()
        while not state.done:
            if server.consume_reset():
                break
            seat = state.current_player
            action = agents[seat].select_action(state, seat)
            step(state, action, strict=False)
            server.push_frame(state, win_prob_fn(state))
            time.sleep(delay)

        if not loop:
            return
        # Brief idle so the final frame is visible before resetting; a reset
        # request during this window is honoured immediately.
        for _ in range(int(max(1, config.fps) * 2)):
            if server.consume_reset():
                break
            time.sleep(0.5)
        episode_index += 1


_LIVE_TEMPLATE: Final[str] = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__TT_LIVE_TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1115;
    color: #e6e6e6;
  }
  .wrap { display: flex; flex-wrap: wrap; gap: 24px; padding: 24px; }
  .board-col { flex: 1 1 420px; min-width: 360px; }
  .side-col { flex: 1 1 320px; min-width: 300px; max-width: 480px; }
  h1 { font-size: 20px; margin: 0 0 12px; letter-spacing: 0.3px; }
  .meta { font-size: 12px; color: #9aa0a6; margin-bottom: 12px; }
  canvas {
    width: 100%;
    height: auto;
    background: #1a1d23;
    border: 1px solid #2a2f38;
    border-radius: 6px;
    image-rendering: pixelated;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 12px;
    flex-wrap: wrap;
  }
  button {
    background: #2a2f38;
    color: #e6e6e6;
    border: 1px solid #3a4049;
    border-radius: 4px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 13px;
  }
  button:hover { background: #353b45; }
  button:disabled { opacity: 0.4; cursor: default; }
  .turn-readout { font-size: 12px; color: #c4c7cc; min-width: 110px; }
  .live-pill {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 999px;
    background: #2a2f38;
    color: #9aa0a6;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  .live-pill.on { background: #1f3a1f; color: #b8e6b8; }
  .live-pill.off { background: #3a1f1f; color: #e6b8b8; }
  .agent-card {
    background: #1a1d23;
    border: 1px solid #2a2f38;
    border-left-width: 6px;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 10px;
  }
  .agent-card.current { box-shadow: 0 0 0 1px #f5c542 inset; }
  .agent-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
  }
  .agent-name { font-weight: 600; font-size: 14px; }
  .agent-strategy {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #9aa0a6;
  }
  .agent-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 4px 8px;
    font-size: 12px;
    margin-top: 6px;
    color: #c4c7cc;
  }
  .agent-stats .k { color: #9aa0a6; font-size: 11px; }
  .prob-bar {
    margin-top: 8px;
    height: 14px;
    background: #2a2f38;
    border-radius: 3px;
    overflow: hidden;
    position: relative;
  }
  .prob-fill {
    height: 100%;
    width: 0%;
    transition: width 240ms ease-out;
  }
  .prob-label {
    position: absolute;
    top: -1px;
    right: 6px;
    font-size: 11px;
    line-height: 16px;
    color: #e6e6e6;
    text-shadow: 0 1px 2px rgba(0,0,0,0.4);
  }
  .winner-banner {
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 6px;
    background: #1a1d23;
    border: 1px solid #3a4049;
    font-size: 14px;
  }
  .winner-banner.active { border-color: #f5c542; color: #fff5cf; }
  .dead { opacity: 0.45; }
</style>
</head>
<body>
<div class="wrap">
  <div class="board-col">
    <h1>__TT_LIVE_TITLE__</h1>
    <div class="meta" id="meta">Connecting&hellip;</div>
    <canvas id="board" width="480" height="480"></canvas>
    <div class="controls">
      <button id="reset" title="Start a fresh game">Reset</button>
      <span class="turn-readout" id="turn-readout"></span>
      <span class="live-pill" id="live-pill">offline</span>
    </div>
    <div class="winner-banner" id="winner-banner">Waiting for game&hellip;</div>
  </div>
  <div class="side-col" id="side-col"></div>
</div>
<script>
(function () {
  "use strict";

  var canvas = document.getElementById("board");
  var ctx = canvas.getContext("2d");
  var meta = document.getElementById("meta");
  var banner = document.getElementById("winner-banner");
  var turnReadout = document.getElementById("turn-readout");
  var pill = document.getElementById("live-pill");
  var sideCol = document.getElementById("side-col");

  var INIT = null;       // last init payload {board_height, board_width, ...}
  var EPISODE = -1;      // last seen episode id
  var FRAMES = [];       // accumulated frames for this episode
  var CARDS = [];        // per-seat DOM card refs
  var CELL = 24;

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {
        "&": "&amp;", "<": "&lt;", ">": "&gt;",
        '"': "&quot;", "'": "&#39;"
      }[c];
    });
  }

  function setPill(state) {
    pill.classList.remove("on", "off");
    if (state === "live") {
      pill.classList.add("on");
      pill.textContent = "live";
    } else if (state === "offline") {
      pill.classList.add("off");
      pill.textContent = "offline";
    } else {
      pill.textContent = state;
    }
  }

  function rebuildSidePanel(init) {
    sideCol.innerHTML = "";
    CARDS = [];
    for (var s = 0; s < init.num_players; s++) {
      var agent = init.agents[s];
      var pathColor = init.tile_colors[s + 1];
      var card = document.createElement("div");
      card.className = "agent-card";
      card.style.borderLeftColor = pathColor;
      var elo = agent.elo === null || agent.elo === undefined
          ? "—"
          : (agent.elo >= 0 ? "+" : "") + Math.round(agent.elo);
      card.innerHTML =
        '<div class="agent-head">' +
          '<span class="agent-name">' + escapeHtml(agent.name) + '</span>' +
          '<span class="agent-strategy">' + escapeHtml(agent.strategy) + '</span>' +
        '</div>' +
        '<div class="agent-stats">' +
          '<div><div class="k">Seat</div>' + (s + 1) + '</div>' +
          '<div><div class="k">Elo</div>' + elo + '</div>' +
          '<div><div class="k">Status</div><span class="status">alive</span></div>' +
          '<div><div class="k">Claimed</div><span class="claimed">0</span></div>' +
          '<div><div class="k">Path</div><span class="path-len">0</span></div>' +
          '<div><div class="k">Win&nbsp;prob</div><span class="prob-pct">0%</span></div>' +
        '</div>' +
        '<div class="prob-bar">' +
          '<div class="prob-fill" style="background-color:' + pathColor + ';"></div>' +
        '</div>';
      sideCol.appendChild(card);
      CARDS.push({
        root: card,
        status: card.querySelector(".status"),
        claimed: card.querySelector(".claimed"),
        pathLen: card.querySelector(".path-len"),
        pct: card.querySelector(".prob-pct"),
        fill: card.querySelector(".prob-fill")
      });
    }
  }

  function applyInit(init) {
    INIT = init;
    canvas.width = init.board_width * CELL;
    canvas.height = init.board_height * CELL;
    meta.textContent = init.board_height + "x" + init.board_width
        + " board, " + init.num_players + " players";
    banner.classList.remove("active");
    banner.textContent = "Game in progress…";
    rebuildSidePanel(init);
  }

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
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (var i = 1; i < W; i++) {
      ctx.beginPath();
      ctx.moveTo(i * CELL + 0.5, 0);
      ctx.lineTo(i * CELL + 0.5, canvas.height);
      ctx.stroke();
    }
    for (var j = 1; j < H; j++) {
      ctx.beginPath();
      ctx.moveTo(0, j * CELL + 0.5);
      ctx.lineTo(canvas.width, j * CELL + 0.5);
      ctx.stroke();
    }
    for (var p = 0; p < INIT.num_players; p++) {
      var head = f.heads[p];
      if (head[0] < 0 || head[1] < 0) continue;
      if (!f.alive[p]) continue;
      ctx.strokeStyle = EDGE[p];
      ctx.lineWidth = 3;
      ctx.strokeRect(
        head[1] * CELL + 1.5,
        head[0] * CELL + 1.5,
        CELL - 3,
        CELL - 3
      );
    }
  }

  function updateSide(f) {
    if (!INIT) return;
    for (var s = 0; s < INIT.num_players; s++) {
      var c = CARDS[s];
      if (!c) continue;
      c.claimed.textContent = f.claimed[s];
      c.pathLen.textContent = f.path_len[s];
      var pct = Math.round(f.win_probs[s] * 100);
      c.pct.textContent = pct + "%";
      c.fill.style.width = pct + "%";
      if (f.alive[s]) {
        c.status.textContent = "alive";
        c.root.classList.remove("dead");
      } else {
        c.status.textContent = "dead";
        c.root.classList.add("dead");
      }
      if (s === f.current_player && !f.done) {
        c.root.classList.add("current");
      } else {
        c.root.classList.remove("current");
      }
    }
  }

  function applyFrame(f) {
    drawFrame(f);
    updateSide(f);
    turnReadout.textContent = "Turn " + f.turn
        + " · Frame " + FRAMES.length;
    if (f.done) {
      banner.classList.add("active");
      if (f.winner === null || f.winner === undefined) {
        banner.textContent = "Game over — tie.";
      } else {
        var w = INIT.agents[f.winner];
        banner.textContent = "Winner: " + w.name + " (seat " + (f.winner + 1) + ")";
      }
    } else {
      banner.classList.remove("active");
      banner.textContent = "Game in progress…";
    }
  }

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
          FRAMES = [];
          if (j.init) applyInit(j.init);
        }
        for (var i = 0; i < j.frames.length; i++) {
          FRAMES.push(j.frames[i]);
          applyFrame(j.frames[i]);
        }
      })
      .catch(function () {
        setPill("offline");
      })
      .then(function () {
        setTimeout(poll, 200);
      });
  }

  document.getElementById("reset").onclick = function () {
    fetch("/reset", { method: "POST", cache: "no-store" })
      .catch(function () { setPill("offline"); });
  };

  setPill("offline");
  poll();
})();
</script>
</body>
</html>
"""


__all__ = [
    "LiveConfig",
    "LiveServer",
    "play_and_serve",
    "render_live_html",
]
