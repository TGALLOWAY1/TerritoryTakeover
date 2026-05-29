"""Interactive HTTP server for the Territory Takeover *Arena* front end.

Serves a polished, mobile-first single-page UI (no external assets) that drives
the real :mod:`territory_takeover.engine` enclosure game. Visitors watch up to
four agents battle for territory, swap any agent's strategy from a dropdown, and
drive the simulation with play / pause / step / speed / reset controls. Seat 0
can optionally be handed to a human (arrow keys / WASD) for Tron-style play.

The territory model is the engine's own: a cell is *owned* by a player the moment
that player visits it (its path) and stays owned for the rest of the game, plus
any cells the player encloses. A player's territory is therefore
``len(path) + claimed_count`` — exactly the engine's scoring.

Endpoints (all served from one process, no build tooling):

- ``GET  /``        — the arena page (single self-contained HTML file)
- ``GET  /state``   — incremental frame poll (adds ``paused``/``speed`` plus
                      ``waiting_for_human``/``human_seat`` to the viz_live shape)
- ``GET  /agents``  — available strategy presets (key/label/description)
- ``GET  /healthz`` — liveness probe (always open, for cloud health checks)
- ``POST /start``   — start a match from a JSON config body
- ``POST /control`` — ``{"cmd": play|pause|step|reset|speed, "speed"?: float}``
- ``POST /action``  — submit a human move (``{"action": 0|1|2|3}``)

An optional shared access token (``InteractiveServer(token=...)``) gates every
endpoint except ``/healthz``; clients present it via the ``token`` query param,
an ``X-Arena-Token`` header, or a ``tt_token`` cookie. When the token is ``None``
(the default) all endpoints are open, which suits localhost use.

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
import hmac
import html as _html
import json
import queue
import threading
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
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
        "description": "Random walk — picks uniformly among legal moves.",
        "class": "UniformRandomAgent",
        "kwargs": {},
    },
    "greedy": {
        "label": "Greedy",
        "description": "1-ply heuristic — maximizes board score after the move.",
        "class": "HeuristicGreedyAgent",
        "kwargs": {},
    },
    "mcts-easy": {
        "label": "MCTS Easy",
        "description": "Monte-Carlo tree search — 50 simulations per move.",
        "class": "UCTAgent",
        "kwargs": {"iterations": 50},
    },
    "mcts-medium": {
        "label": "MCTS Medium",
        "description": "Monte-Carlo tree search — 200 simulations per move.",
        "class": "UCTAgent",
        "kwargs": {"iterations": 200},
    },
    "mcts-hard": {
        "label": "MCTS Hard",
        "description": "Monte-Carlo tree search — 800 simulations per move.",
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
    name = f"p{seat + 1}-{preset_key}"
    if cls_name == "UniformRandomAgent":
        return UniformRandomAgent(rng=rng, name=name)
    if cls_name == "HeuristicGreedyAgent":
        return HeuristicGreedyAgent(rng=rng, name=name)
    if cls_name == "UCTAgent":
        raw_kwargs = preset.get("kwargs", {})
        raw_iters = raw_kwargs.get("iterations", 200) if isinstance(raw_kwargs, dict) else 200
        iterations = int(raw_iters) if isinstance(raw_iters, (int, float)) else 200
        return UCTAgent(rng=rng, name=name, iterations=iterations)
    raise ValueError(f"Unknown agent class {cls_name!r}")


def _cfg_int(config: dict[str, object], key: str, default: int) -> int:
    """Coerce a config value to ``int``, falling back to *default*."""
    value = config.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _cfg_float(config: dict[str, object], key: str, default: float) -> float:
    """Coerce a config value to ``float``, falling back to *default*."""
    value = config.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _cfg_human_seat(config: dict[str, object]) -> int | None:
    """Read an optional ``human_seat`` index from *config*."""
    value = config.get("human_seat")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _cfg_agents(config: dict[str, object]) -> dict[str, str]:
    """Read the per-seat ``{seat: preset_key}`` mapping from *config*."""
    raw = config.get("agents")
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class InteractiveServer:
    """HTTP server that streams live game frames and accepts control commands.

    Call :meth:`start` to launch the background HTTP thread, then POST ``/start``
    from the browser (or call :meth:`start_game` directly) to begin a match.
    Watch-mode pacing is driven by play / pause / step / speed controls; play
    mode hands seat 0 to a human whose keypresses set the tempo.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        title: str = "Territory Takeover",
        token: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._title = title
        # Optional shared access token. When None, all endpoints are open
        # (fine for localhost); when set, every request must present it via
        # the ``token`` query param, ``X-Arena-Token`` header, or ``tt_token``
        # cookie. Empty strings are treated as "no token" so a blank env var
        # does not silently lock everyone out.
        self._token = token or None

        # Frame buffer shared with the HTTP handler (guarded by _lock).
        self._lock = threading.Lock()
        self._episode: int = 0
        self._frames: list[dict[str, object]] = []
        self._init: dict[str, object] = {}

        # Signals the game loop to stop (set by /control reset or start_game).
        self._reset_event = threading.Event()

        # Watch-mode pacing controls (guarded by _ctrl_cond).
        self._ctrl_cond = threading.Condition()
        self._paused: bool = False
        self._speed: float = 1.0
        self._step_requests: int = 0
        self._last_config: dict[str, object] | None = None

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
    # Watch-mode controls (called from the HTTP handler thread)
    # ------------------------------------------------------------------

    def set_paused(self, paused: bool) -> None:
        """Pause or resume automatic stepping in watch mode."""
        with self._ctrl_cond:
            self._paused = paused
            self._ctrl_cond.notify_all()

    def set_speed(self, speed: float) -> None:
        """Set the watch-mode speed multiplier (clamped to a sane range)."""
        with self._ctrl_cond:
            self._speed = max(0.1, min(50.0, speed))
            self._ctrl_cond.notify_all()

    def request_step(self) -> None:
        """Advance exactly one move, then remain paused."""
        with self._ctrl_cond:
            self._paused = True
            self._step_requests += 1
            self._ctrl_cond.notify_all()

    def _wait_to_step(self) -> tuple[bool, bool, float]:
        """Block until a watch-mode move may run.

        Returns ``(proceed, manual, speed)``. ``proceed`` is ``False`` only when
        a reset was requested; ``manual`` is ``True`` for a single-step request
        (no pacing sleep should follow).
        """
        with self._ctrl_cond:
            while True:
                if self._reset_event.is_set():
                    return (False, False, 1.0)
                if self._step_requests > 0:
                    self._step_requests -= 1
                    return (True, True, self._speed)
                if not self._paused:
                    return (True, False, self._speed)
                self._ctrl_cond.wait(timeout=0.1)

    def _paced_sleep(self, seconds: float) -> None:
        """Sleep up to *seconds*, waking early if a reset is requested."""
        deadline = time.monotonic() + seconds
        while not self._reset_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.05))

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
                    "paused": self._paused,
                    "speed": self._speed,
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
                "paused": self._paused,
                "speed": self._speed,
                "waiting_for_human": self._waiting_for_human,
                "human_seat": self._human_seat,
            }

    # ------------------------------------------------------------------
    # Game management
    # ------------------------------------------------------------------

    def start_game(self, config: dict[str, object]) -> None:
        """Stop any running game and launch a new one from *config*."""
        self._reset_event.set()
        with self._ctrl_cond:
            self._ctrl_cond.notify_all()
        if self._game_thread is not None and self._game_thread.is_alive():
            self._game_thread.join(timeout=3.0)
        self._reset_event.clear()

        # Drain stale queued actions.
        while True:
            try:
                self._action_queue.get_nowait()
            except queue.Empty:
                break

        # Reset watch-mode controls for the new match.
        with self._ctrl_cond:
            self._paused = not bool(config.get("autoplay", True))
            self._speed = _cfg_float(config, "speed", 1.0)
            self._step_requests = 0
        self._last_config = dict(config)

        self._waiting_for_human = False
        self._human_seat = _cfg_human_seat(config)

        self._game_thread = threading.Thread(
            target=self._run_game,
            args=(config,),
            daemon=True,
            name="tt-game-loop",
        )
        self._game_thread.start()

    def _run_game(self, config: dict[str, object]) -> None:
        board_size = _cfg_int(config, "board_size", 32)
        num_players = _cfg_int(config, "num_players", 4)
        base_fps = max(1, _cfg_int(config, "fps", 6))
        human_seat = _cfg_human_seat(config)
        agents_cfg = _cfg_agents(config)

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
                        name=str(preset["label"]),
                        strategy=preset_key,
                        elo=None,
                    )
                )

        state = new_game(board_size=board_size, num_players=num_players)
        self._set_episode(
            agent_cards=agent_cards,
            board_height=int(state.grid.shape[0]),
            board_width=int(state.grid.shape[1]),
            num_players=num_players,
            fps=base_fps,
        )
        self._push_frame(state, win_probs(state))

        for a in agents:
            if a is not None:
                a.reset()

        while not state.done:
            if self._reset_event.is_set():
                return

            seat = state.current_player

            if human_seat is None:
                # Watch mode: pacing is driven by the play/pause/step controls.
                proceed, manual, speed = self._wait_to_step()
                if not proceed:
                    return
                ai = agents[seat]
                assert ai is not None
                action = ai.select_action(state, seat)
                step(state, action, strict=False)
                self._push_frame(state, win_probs(state))
                if not manual and not self._paused:
                    self._paced_sleep(1.0 / (base_fps * max(0.1, speed)))
                continue

            # Play mode: the human drives tempo; AI opponents respond at once.
            if seat == human_seat:
                self._waiting_for_human = True
                action_h: int | None = None
                while action_h is None:
                    if self._reset_event.is_set():
                        self._waiting_for_human = False
                        return
                    with contextlib.suppress(queue.Empty):
                        action_h = self._action_queue.get(timeout=0.05)
                self._waiting_for_human = False
                step(state, action_h, strict=False)
            else:
                ai = agents[seat]
                assert ai is not None
                step(state, ai.select_action(state, seat), strict=False)
            self._push_frame(state, win_probs(state))

        # Brief pause on the final frame so the winner banner is visible.
        for _ in range(base_fps * 2):
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
        with self._ctrl_cond:
            self._ctrl_cond.notify_all()
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

        def _provided_token(self) -> str | None:
            """Extract a candidate token from query, header, or cookie."""
            qs = parse_qs(urlsplit(self.path).query)
            tokens = qs.get("token")
            if tokens:
                return tokens[0]
            header = self.headers.get("X-Arena-Token")
            if header:
                return header
            raw_cookie = self.headers.get("Cookie")
            if raw_cookie:
                jar: SimpleCookie = SimpleCookie()
                jar.load(raw_cookie)
                if "tt_token" in jar:
                    return jar["tt_token"].value
            return None

        def _check_auth(self) -> bool:
            """True if no token is configured or the request presents it."""
            expected = server._token
            if expected is None:
                return True
            provided = self._provided_token()
            if provided is None:
                return False
            return hmac.compare_digest(provided, expected)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            # Liveness probe for cloud health checks — always open (no token),
            # so deploys stay healthy even when an access token is configured.
            if path == "/healthz":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            if not self._check_auth():
                if path in ("/", "/index.html"):
                    self._serve_login()
                else:
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if path in ("/", "/index.html"):
                self._serve_index()
            elif path == "/state":
                self._serve_state()
            elif path == "/agents":
                presets = [
                    {
                        "key": k,
                        "label": v["label"],
                        "description": v.get("description", ""),
                    }
                    for k, v in AGENT_PRESETS.items()
                ]
                self._send_json(HTTPStatus.OK, {"presets": presets})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if not self._check_auth():
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if path == "/start":
                body = self._read_json_body()
                if body is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
                    return
                server.start_game(body)
                self._send_json(HTTPStatus.OK, {"ok": True})
            elif path == "/control":
                self._handle_control()
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
                if server._last_config is not None:
                    server.start_game(server._last_config)
                self._send_json(HTTPStatus.OK, {"ok": True})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _handle_control(self) -> None:
            body = self._read_json_body()
            if body is None:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
                return
            cmd = body.get("cmd")
            if cmd == "play":
                server.set_paused(False)
            elif cmd == "pause":
                server.set_paused(True)
            elif cmd == "step":
                server.request_step()
            elif cmd == "speed":
                sp = body.get("speed")
                if not isinstance(sp, (int, float)):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad speed"})
                    return
                server.set_speed(float(sp))
            elif cmd == "reset":
                if server._last_config is not None:
                    server.start_game(server._last_config)
            else:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "unknown cmd"})
                return
            self._send_json(HTTPStatus.OK, {"ok": True})

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
            return {str(k): v for k, v in data.items()}

        def _serve_index(self) -> None:
            body = render_interactive_html(server._title).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # If the token arrived via the URL, persist it as a cookie so the
            # page's polling/control fetches authenticate automatically and the
            # token drops out of the address bar on the next navigation.
            if server._token is not None:
                qs = parse_qs(urlsplit(self.path).query)
                if qs.get("token", [None])[0] == server._token:
                    cookie = (
                        f"tt_token={server._token}; Path=/; SameSite=Strict; "
                        "Max-Age=604800"
                    )
                    self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def _serve_login(self) -> None:
            body = _LOGIN_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.UNAUTHORIZED)
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
    """Render the arena page with *title* injected."""
    return _TEMPLATE.replace(_TITLE_MARKER, _html.escape(title))


# Shown (HTTP 401) when an access token is required but missing/invalid. The
# form simply reloads ``/?token=...``; on success the server sets the cookie.
_LOGIN_PAGE: Final[str] = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="theme-color" content="#070a12"/>
<title>Access required</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#070a12;color:#e8edf7;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif}
.wrap{min-height:100%;display:flex;align-items:center;justify-content:center;padding:24px}
.card{width:100%;max-width:340px;background:linear-gradient(180deg,#0f1626,#131c30);
  border:1px solid #1d2944;border-radius:16px;padding:22px;
  box-shadow:0 10px 30px rgba(0,0,0,.35)}
h1{font-size:18px;margin:0 0 4px;letter-spacing:1px}
p{font-size:13px;color:#8a96b0;margin:0 0 16px}
input{width:100%;padding:12px;font-size:15px;border-radius:10px;
  background:#0b1220;border:1px solid #1d2944;color:#e8edf7;margin-bottom:12px}
button{width:100%;padding:12px;font-size:15px;font-weight:700;border:none;
  border-radius:10px;background:#3b82f6;color:#fff;cursor:pointer}
</style>
</head>
<body>
<div class="wrap"><div class="card">
<h1>Territory Takeover</h1>
<p>Enter the access token to view the arena.</p>
<form onsubmit="go(event)">
<input id="t" type="password" autocomplete="off" placeholder="Access token" autofocus/>
<button type="submit">Enter</button>
</form>
</div></div>
<script>
function go(e){e.preventDefault();
  var v=document.getElementById("t").value;
  location.href="/?token="+encodeURIComponent(v);}
</script>
</body>
</html>
"""


_TEMPLATE: Final[str] = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta name="theme-color" content="#070a12"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<meta name="mobile-web-app-capable" content="yes"/>
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
<title>__TT_ITITLE__</title>
<style>
:root{
  --bg:#070a12; --panel:#0f1626; --panel2:#131c30; --line:#1d2944;
  --ink:#e8edf7; --mut:#8a96b0; --mut2:#5d6886; --accent:#3b82f6;
  --s0:#ff6a3d; --s1:#a855f7; --s2:#22c55e; --s3:#3b82f6;
}
*{box-sizing:border-box}
/* Kill the 300ms tap delay / double-tap zoom on interactive controls. */
.icon-btn,.chip,.ctl,.seg,.mi,.speed-sel{touch-action:manipulation}
html,body{margin:0;background:var(--bg);color:var(--ink)}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.app{max-width:480px;margin:0 auto;padding:14px 14px 84px;position:relative}
.glass{
  background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);
  border:1px solid var(--line);border-radius:16px;
  box-shadow:0 10px 30px rgba(0,0,0,.35);
}

/* header */
.hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.icon-btn{
  width:42px;height:42px;border-radius:12px;cursor:pointer;
  background:var(--panel);border:1px solid var(--line);color:var(--ink);
  display:flex;align-items:center;justify-content:center;
  transition:background .15s,border-color .15s;
}
.icon-btn:hover{background:var(--panel2);border-color:#2a3a5e}
.title-wrap{text-align:center;flex:1;padding:0 8px}
.title{
  font-size:21px;font-weight:800;letter-spacing:3px;margin:0;
  background:linear-gradient(90deg,#fff,#9fc2ff);
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.subtitle{font-size:11px;color:var(--mut);margin-top:3px;letter-spacing:.4px}

/* section label */
.sec-label{
  font-size:11px;color:var(--mut2);letter-spacing:1.4px;
  text-transform:uppercase;margin:0 4px 8px;
}

/* agent selector bar */
.selbar{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:16px}
.chip{
  position:relative;display:flex;align-items:center;gap:8px;
  padding:9px 10px;border-radius:12px;cursor:pointer;
  background:var(--panel);border:1px solid var(--line);
  border-left:3px solid var(--c);
  transition:border-color .15s,background .15s;
}
.chip:hover{background:var(--panel2);border-color:#2a3a5e;border-left-color:var(--c)}
.chip.open{border-color:var(--c);box-shadow:0 0 0 1px var(--c) inset}
.chip-ic{color:var(--c);display:flex;flex-shrink:0;filter:drop-shadow(0 0 4px var(--c))}
.chip-name{font-size:13px;font-weight:600;flex:1;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.chev{color:var(--mut);font-size:11px;flex-shrink:0;transition:transform .15s}
.chip.open .chev{transform:rotate(180deg)}
.chip-menu{
  display:none;position:absolute;top:calc(100% + 6px);left:0;right:0;z-index:40;
  background:var(--panel2);border:1px solid var(--line);border-radius:12px;
  padding:5px;box-shadow:0 16px 40px rgba(0,0,0,.5);
}
.chip.open .chip-menu{display:block}
.mi{padding:8px 9px;border-radius:8px;cursor:pointer}
.mi:hover{background:var(--panel)}
.mi.on{background:rgba(59,130,246,.16)}
.mi-name{font-size:13px;font-weight:600}
.mi-desc{font-size:11px;color:var(--mut);margin-top:2px;line-height:1.3}

/* board */
.board-wrap{padding:10px;margin-bottom:14px;position:relative}
canvas{width:100%;height:auto;display:block;border-radius:10px;background:#05080f}
.board-glow{position:absolute;inset:0;border-radius:16px;pointer-events:none;
  box-shadow:0 0 60px rgba(59,130,246,.06) inset}
.winner{
  position:absolute;inset:10px;border-radius:10px;display:none;
  align-items:center;justify-content:center;flex-direction:column;gap:6px;
  background:rgba(5,8,15,.78);backdrop-filter:blur(2px);z-index:20;text-align:center;
}
.winner.show{display:flex}
.winner .wt{font-size:12px;color:var(--mut);letter-spacing:2px;text-transform:uppercase}
.winner .wn{font-size:24px;font-weight:800}
.winner .wh{font-size:12px;color:var(--mut)}

/* controls */
.controls{display:flex;align-items:center;gap:8px;padding:10px;margin-bottom:16px}
.ctl{
  width:44px;height:44px;border-radius:12px;cursor:pointer;flex-shrink:0;
  background:var(--panel);border:1px solid var(--line);color:var(--ink);
  display:flex;align-items:center;justify-content:center;font-size:15px;
  transition:background .15s,border-color .15s;
}
.ctl:hover{background:var(--panel2);border-color:#2a3a5e}
.ctl.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.ctl.primary:hover{background:#2f6fe0}
.ctl-spacer{flex:1}
.readout{text-align:center;line-height:1.1}
.readout .turn{font-size:15px;font-weight:700}
.readout .time{font-size:11px;color:var(--mut);font-variant-numeric:tabular-nums}
.speed-sel{
  background:var(--panel);border:1px solid var(--line);color:var(--ink);
  border-radius:10px;padding:0 8px;height:40px;font-size:13px;cursor:pointer;
}

/* stats */
.stats{padding:6px;margin-bottom:14px}
.stat-row{display:flex;align-items:center;gap:11px;padding:11px 8px;
  border-bottom:1px solid var(--line)}
.stat-row:last-child{border-bottom:none}
.stat-ic{width:36px;height:36px;border-radius:10px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;color:var(--c);
  background:rgba(255,255,255,.03);border:1px solid var(--line)}
.stat-main{flex:1;min-width:0}
.stat-name{font-size:14px;font-weight:700}
.stat-desc{font-size:11px;color:var(--mut);margin:1px 0 5px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.terr{display:flex;align-items:center;gap:8px}
.terr-pct{font-size:13px;font-weight:700;min-width:50px;font-variant-numeric:tabular-nums}
.terr-bar{flex:1;height:6px;border-radius:4px;background:var(--line);overflow:hidden}
.terr-fill{height:100%;width:0%;border-radius:4px;background:var(--c);
  transition:width .25s ease-out}
.stat-right{text-align:right;flex-shrink:0}
.status{font-size:12px;font-weight:600;display:flex;align-items:center;
  gap:5px;justify-content:flex-end}
.dot{width:7px;height:7px;border-radius:50%;background:#22c55e;
  box-shadow:0 0 6px #22c55e}
.status.dead{color:var(--mut2)}
.status.dead .dot{background:#64748b;box-shadow:none}
.mini{font-size:10px;color:var(--mut2);margin-top:4px;font-variant-numeric:tabular-nums}
.stat-row.dead-row{opacity:.5}

/* placeholder views */
.ph{padding:40px 20px;text-align:center;margin-bottom:14px;display:none}
.ph.show{display:block}
.ph h3{margin:0 0 6px;font-size:16px}
.ph p{margin:0;color:var(--mut);font-size:13px}

/* drawers (debug + settings) */
.drawer{display:none;padding:14px;margin-bottom:14px}
.drawer.show{display:block}
.drawer h3{margin:0 0 10px;font-size:13px;letter-spacing:1px;
  text-transform:uppercase;color:var(--mut)}
.dbg-grid{font-size:12px;font-variant-numeric:tabular-nums}
.dbg-line{display:flex;justify-content:space-between;padding:4px 0;
  border-bottom:1px solid var(--line)}
.dbg-line:last-child{border-bottom:none}
.dbg-k{color:var(--mut)}
.seg-row{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 12px}
.seg{flex:1;min-width:56px;background:var(--panel);color:var(--mut);
  border:1px solid var(--line);border-radius:9px;padding:8px 0;font-size:13px;
  cursor:pointer;transition:background .12s,color .12s}
.seg.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.chk{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--mut);
  margin-bottom:12px;cursor:pointer}
.apply{width:100%;padding:11px;border-radius:10px;cursor:pointer;
  background:var(--accent);border:1px solid var(--accent);color:#fff;
  font-size:14px;font-weight:600}

/* bottom nav */
.nav{position:fixed;left:0;right:0;bottom:0;z-index:50;
  background:rgba(8,11,18,.92);backdrop-filter:blur(10px);
  border-top:1px solid var(--line);display:flex}
.nav-inner{max-width:480px;margin:0 auto;width:100%;display:flex}
.nav-item{flex:1;padding:11px 0 13px;text-align:center;cursor:pointer;
  color:var(--mut2);font-size:11px;letter-spacing:.4px}
.nav-item.on{color:var(--accent)}
.nav-item .ni{display:block;margin:0 auto 3px;width:22px;height:22px}
.kbd-hint{display:none;text-align:center;font-size:11px;color:var(--mut2);
  margin-bottom:12px}
.kbd-hint.show{display:block}

/* very small phones — tighten gaps so the control row / agent grid fit */
@media (max-width:360px){
  .app{padding:10px 10px 80px}
  .selbar{gap:6px}
  .chip{padding:8px;gap:6px}
  .chip-name{font-size:12px}
  .controls{gap:6px;padding:8px}
  .ctl{width:40px;height:40px}
  .speed-sel{height:38px;padding:0 6px}
  .title{font-size:18px;letter-spacing:2px}
}
</style>
</head>
<body>
<div class="app">

  <!-- header -->
  <div class="hdr">
    <div class="icon-btn" id="menu-btn" title="Debug panel">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round"><path d="M3 6h18M3 12h18M3 18h18"/></svg>
    </div>
    <div class="title-wrap">
      <h1 class="title">__TT_ITITLE__</h1>
      <div class="subtitle">Claim territory. Cut off opponents. Win.</div>
    </div>
    <div class="icon-btn" id="settings-btn" title="Settings">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65
          1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51
          1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82
          1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82
          l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2
          2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83
          2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65
          1.65 0 0 0-1.51 1z"/></svg>
    </div>
  </div>

  <!-- settings drawer -->
  <div class="glass drawer" id="settings-drawer">
    <h3>Settings</h3>
    <div class="sec-label" style="margin-left:0">Board size</div>
    <div class="seg-row" id="size-row">
      <button class="seg" data-val="16">16</button>
      <button class="seg" data-val="24">24</button>
      <button class="seg on" data-val="32">32</button>
      <button class="seg" data-val="40">40</button>
    </div>
    <label class="chk"><input type="checkbox" id="human-chk"/>
      Play seat 1 myself (arrow keys / WASD)</label>
    <button class="apply" id="apply-settings">Apply &amp; new match</button>
  </div>

  <!-- debug drawer -->
  <div class="glass drawer" id="debug-drawer">
    <h3>Debug panel</h3>
    <div class="dbg-grid" id="dbg-grid"></div>
  </div>

  <!-- ARENA view -->
  <div id="arena-view">
    <div class="sec-label">Select agents &mdash; tap to swap strategy</div>
    <div class="selbar" id="selbar"></div>

    <div class="glass board-wrap">
      <div class="board-glow"></div>
      <canvas id="board" width="640" height="640"></canvas>
      <div class="winner" id="winner">
        <div class="wt">Game over</div>
        <div class="wn" id="winner-name">&mdash;</div>
        <div class="wh">Tap reset for a new match</div>
      </div>
    </div>

    <div class="glass controls">
      <div class="ctl primary" id="play-btn" title="Play / pause"></div>
      <div class="ctl" id="step-btn" title="Step one move">&#x21E5;</div>
      <div class="ctl" id="reset-btn" title="Reset match">&#x21BB;</div>
      <div class="ctl-spacer"></div>
      <div class="readout">
        <div class="turn" id="turn-ro">Turn 0</div>
        <div class="time" id="time-ro">00:00</div>
      </div>
      <select class="speed-sel" id="speed-sel">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1.0x</option>
        <option value="2">2.0x</option>
        <option value="5">5.0x</option>
        <option value="20">20x</option>
      </select>
    </div>

    <div class="kbd-hint" id="kbd-hint">Your turn &mdash; &#x2190;&#x2191;&#x2192;&#x2193;
      or WASD</div>

    <div class="sec-label">Agents</div>
    <div class="glass stats" id="stats"></div>
  </div>

  <!-- placeholder views -->
  <div class="glass ph" id="agents-view">
    <h3>Agents</h3>
    <p>Strategy library &amp; tuning coming soon. Swap agents from the Arena for now.</p>
  </div>
  <div class="glass ph" id="history-view">
    <h3>History</h3>
    <p>Match history &amp; replays coming soon.</p>
  </div>
</div>

<!-- bottom nav -->
<div class="nav"><div class="nav-inner">
  <div class="nav-item on" data-view="arena">
    <svg class="ni" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="3" y="3" width="7" height="7" rx="1"/>
      <rect x="14" y="3" width="7" height="7" rx="1"/>
      <rect x="3" y="14" width="7" height="7" rx="1"/>
      <rect x="14" y="14" width="7" height="7" rx="1"/>
    </svg>Arena</div>
  <div class="nav-item" data-view="agents">
    <svg class="ni" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round"><rect x="5" y="8" width="14" height="11" rx="3"/>
      <path d="M12 4v4"/><circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/>
    </svg>Agents</div>
  <div class="nav-item" data-view="history">
    <svg class="ni" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      stroke-linecap="round"><path d="M4 19V5M10 19v-8M16 19v-5M20 19V9"/></svg>History</div>
</div></div>

<script>
(function () {
"use strict";

/* ── config / state ───────────────────────────────────────────────── */
var DEFAULT_LINEUP = ["greedy", "mcts-easy", "mcts-medium", "random"];
var LINEUP   = DEFAULT_LINEUP.slice();
var PRESETS  = [];
var PRESET_MAP = {};
var BOARD_SIZE = 32, NUM_PLAYERS = 4, BASE_FPS = 6, SPEED = 1, HUMAN = false;
var SEAT = [
  {base:"#ff6a3d"}, {base:"#a855f7"}, {base:"#22c55e"}, {base:"#3b82f6"}
];

var EPISODE = -1, FRAMES = [], INIT = null, CELL = 20;
var PREV_HEADS = [], LAST_MOVE = ["-","-","-","-"];
var matchStart = 0, doneAt = 0, gameDone = false;
var serverPaused = false, waitingForHuman = false, actionSubmitted = false;
var STAT_ROWS = [];

var canvas = document.getElementById("board");
var ctx = canvas.getContext("2d");

function esc(s){
  return String(s).replace(/[&<>"']/g, function(c){
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];
  });
}
function robotSVG(){
  return '<svg viewBox="0 0 24 24" width="20" height="20" fill="none"'
    + ' stroke="currentColor" stroke-width="1.7" stroke-linecap="round"'
    + ' stroke-linejoin="round"><rect x="4" y="8" width="16" height="11" rx="3"/>'
    + '<path d="M12 4v4"/><circle cx="12" cy="3" r="1.2" fill="currentColor"/>'
    + '<circle cx="9" cy="13.5" r="1.3" fill="currentColor" stroke="none"/>'
    + '<circle cx="15" cy="13.5" r="1.3" fill="currentColor" stroke="none"/>'
    + '<path d="M9.5 16.5h5"/></svg>';
}
function fmtTime(ms){
  var s = Math.max(0, Math.floor(ms/1000));
  var m = Math.floor(s/60); s = s % 60;
  return (m<10?"0":"")+m+":"+(s<10?"0":"")+s;
}
function dirName(dr, dc){
  if (dr===-1) return "N"; if (dr===1) return "S";
  if (dc===-1) return "W"; if (dc===1) return "E"; return "-";
}

/* ── agent selector bar ───────────────────────────────────────────── */
function buildSelbar(){
  var bar = document.getElementById("selbar");
  bar.innerHTML = "";
  for (var s = 0; s < NUM_PLAYERS; s++){
    (function(seat){
      var chip = document.createElement("div");
      chip.className = "chip";
      chip.style.setProperty("--c", SEAT[seat].base);
      var menu = '<div class="chip-menu">';
      for (var i = 0; i < PRESETS.length; i++){
        var p = PRESETS[i];
        var on = (p.key === LINEUP[seat]) ? " on" : "";
        menu += '<div class="mi'+on+'" data-key="'+esc(p.key)+'">'
          + '<div class="mi-name">'+esc(p.label)+'</div>'
          + '<div class="mi-desc">'+esc(p.description||"")+'</div></div>';
      }
      menu += '</div>';
      var label = PRESET_MAP[LINEUP[seat]] ? PRESET_MAP[LINEUP[seat]].label : LINEUP[seat];
      chip.innerHTML = '<span class="chip-ic">'+robotSVG()+'</span>'
        + '<span class="chip-name">'+esc(label)+'</span>'
        + '<span class="chev">&#x25BE;</span>' + menu;
      chip.addEventListener("click", function(e){
        var mi = e.target.closest ? e.target.closest(".mi") : null;
        if (mi){
          LINEUP[seat] = mi.dataset.key;
          chip.classList.remove("open");
          buildSelbar();
          startGame();
          return;
        }
        var wasOpen = chip.classList.contains("open");
        closeMenus();
        if (!wasOpen) chip.classList.add("open");
        e.stopPropagation();
      });
      bar.appendChild(chip);
    })(s);
  }
}
function closeMenus(){
  var open = document.querySelectorAll(".chip.open");
  for (var i = 0; i < open.length; i++) open[i].classList.remove("open");
}
document.addEventListener("click", closeMenus);

/* ── stats panel ──────────────────────────────────────────────────── */
function buildStats(){
  var box = document.getElementById("stats");
  box.innerHTML = "";
  STAT_ROWS = [];
  for (var s = 0; s < NUM_PLAYERS; s++){
    var lp = PRESET_MAP[LINEUP[s]] || {label:LINEUP[s], description:""};
    var name = HUMAN && s === 0 ? "You" : lp.label;
    var desc = HUMAN && s === 0 ? "Human · arrow keys" : (lp.description||"");
    var row = document.createElement("div");
    row.className = "stat-row";
    row.innerHTML =
      '<div class="stat-ic" style="color:'+SEAT[s].base+'">'+robotSVG()+'</div>'
      + '<div class="stat-main"><div class="stat-name">'+esc(name)+'</div>'
      + '<div class="stat-desc">'+esc(desc)+'</div>'
      + '<div class="terr"><span class="terr-pct">0.0%</span>'
      + '<div class="terr-bar"><div class="terr-fill" '
      + 'style="background:'+SEAT[s].base+'"></div></div></div></div>'
      + '<div class="stat-right"><div class="status"><span class="dot"></span>'
      + '<span class="stxt">Alive</span></div>'
      + '<div class="mini">owned 0 · legal 0</div></div>';
    box.appendChild(row);
    STAT_ROWS.push({
      root: row,
      pct: row.querySelector(".terr-pct"),
      fill: row.querySelector(".terr-fill"),
      status: row.querySelector(".status"),
      stxt: row.querySelector(".stxt"),
      mini: row.querySelector(".mini")
    });
  }
}
function legalCount(f, seat){
  if (!f.alive[seat]) return 0;
  var h = f.heads[seat];
  if (h[0] < 0) return 0;
  var W = INIT.board_width, H = INIT.board_height, n = 0;
  var D = [[-1,0],[1,0],[0,-1],[0,1]];
  for (var d = 0; d < 4; d++){
    var rr = h[0]+D[d][0], cc = h[1]+D[d][1];
    if (rr<0||rr>=H||cc<0||cc>=W) continue;
    if (f.grid[rr*W+cc] === 0) n++;
  }
  return n;
}
function updateStats(f){
  var owned = [], total = 0;
  for (var s = 0; s < NUM_PLAYERS; s++){
    owned[s] = f.path_len[s] + f.claimed[s];
    total += owned[s];
  }
  for (s = 0; s < NUM_PLAYERS; s++){
    var r = STAT_ROWS[s]; if (!r) continue;
    var pct = total > 0 ? (owned[s]/total*100) : 0;
    r.pct.textContent = pct.toFixed(1)+"%";
    r.fill.style.width = pct.toFixed(1)+"%";
    var lc = legalCount(f, s);
    r.mini.textContent = "owned "+owned[s]+" · legal "+lc;
    if (f.alive[s]){
      r.status.classList.remove("dead"); r.stxt.textContent = "Alive";
      r.root.classList.remove("dead-row");
    } else {
      r.status.classList.add("dead"); r.stxt.textContent = "Dead";
      r.root.classList.add("dead-row");
    }
  }
}

/* ── canvas ───────────────────────────────────────────────────────── */
function hexA(hex, a){
  var n = parseInt(hex.slice(1), 16);
  return "rgba("+((n>>16)&255)+","+((n>>8)&255)+","+(n&255)+","+a+")";
}
function drawFrame(f){
  if (!INIT) return;
  var W = INIT.board_width, H = INIT.board_height;
  ctx.fillStyle = "#05080f";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (var r = 0; r < H; r++){
    for (var c = 0; c < W; c++){
      var v = f.grid[r*W+c];
      if (v === 0) continue;
      var seat = v <= 4 ? v-1 : v-5;
      var isPath = v <= 4;
      var alive = f.alive[seat];
      var a = isPath ? 0.96 : 0.62;
      if (!alive) a *= 0.38;
      ctx.fillStyle = hexA(SEAT[seat].base, a);
      ctx.fillRect(c*CELL+0.5, r*CELL+0.5, CELL-1, CELL-1);
    }
  }
  ctx.strokeStyle = "rgba(120,150,210,0.05)";
  ctx.lineWidth = 1;
  for (var i = 1; i < W; i++){
    ctx.beginPath(); ctx.moveTo(i*CELL+0.5, 0);
    ctx.lineTo(i*CELL+0.5, canvas.height); ctx.stroke();
  }
  for (var j = 1; j < H; j++){
    ctx.beginPath(); ctx.moveTo(0, j*CELL+0.5);
    ctx.lineTo(canvas.width, j*CELL+0.5); ctx.stroke();
  }
  for (var p = 0; p < NUM_PLAYERS; p++){
    var head = f.heads[p];
    if (head[0] < 0 || !f.alive[p]) continue;
    var x = head[1]*CELL, y = head[0]*CELL;
    ctx.save();
    ctx.shadowColor = SEAT[p].base; ctx.shadowBlur = 12;
    ctx.fillStyle = SEAT[p].base;
    ctx.fillRect(x+CELL*0.18, y+CELL*0.18, CELL*0.64, CELL*0.64);
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.strokeRect(x+1.5, y+1.5, CELL-3, CELL-3);
    ctx.restore();
  }
}

/* ── frame application ────────────────────────────────────────────── */
function applyInit(init){
  INIT = init;
  NUM_PLAYERS = init.num_players;
  CELL = Math.max(6, Math.floor(640 / init.board_width));
  canvas.width = init.board_width * CELL;
  canvas.height = init.board_height * CELL;
  PREV_HEADS = [];
  LAST_MOVE = ["-","-","-","-"];
  gameDone = false; doneAt = 0;
  document.getElementById("winner").classList.remove("show");
  buildStats();
}
function trackMoves(f){
  for (var s = 0; s < NUM_PLAYERS; s++){
    var h = f.heads[s];
    var ph = PREV_HEADS[s];
    if (ph && h[0] >= 0 && (ph[0] !== h[0] || ph[1] !== h[1])){
      LAST_MOVE[s] = dirName(h[0]-ph[0], h[1]-ph[1]);
    }
    PREV_HEADS[s] = [h[0], h[1]];
  }
}
function renderLast(f){
  drawFrame(f);
  updateStats(f);
  document.getElementById("turn-ro").textContent = "Turn " + f.turn;
  updateDebug(f);
  if (f.done && !gameDone){
    gameDone = true; doneAt = Date.now();
    var w = document.getElementById("winner");
    var nm = document.getElementById("winner-name");
    if (f.winner === null || f.winner === undefined){
      nm.textContent = "Tie"; nm.style.color = "#fff";
    } else {
      var lab = PRESET_MAP[LINEUP[f.winner]]
        ? PRESET_MAP[LINEUP[f.winner]].label : ("Seat "+(f.winner+1));
      if (HUMAN && f.winner === 0) lab = "You";
      nm.textContent = lab + " wins"; nm.style.color = SEAT[f.winner].base;
    }
    w.classList.add("show");
  }
}

/* ── debug panel ──────────────────────────────────────────────────── */
function updateDebug(f){
  var d = document.getElementById("debug-drawer");
  if (!d.classList.contains("show")) return;
  var aliveN = 0;
  for (var s = 0; s < NUM_PLAYERS; s++) if (f.alive[s]) aliveN++;
  var status = f.done
    ? (f.winner == null ? "done (tie)" : "done (seat "+(f.winner+1)+")")
    : "running";
  var rows = ''
    + line("Turn", f.turn)
    + line("Status", status)
    + line("Alive", aliveN + " / " + NUM_PLAYERS)
    + line("Current seat", f.current_player + 1);
  for (s = 0; s < NUM_PLAYERS; s++){
    var lp = PRESET_MAP[LINEUP[s]] ? PRESET_MAP[LINEUP[s]].label : LINEUP[s];
    var v = esc(lp) + " · " + (f.alive[s] ? "alive" : "dead")
      + " · move " + LAST_MOVE[s] + " · legal " + legalCount(f, s);
    rows += line("Seat " + (s+1), v);
  }
  document.getElementById("dbg-grid").innerHTML = rows;
}
function line(k, v){
  return '<div class="dbg-line"><span class="dbg-k">'+esc(k)+'</span>'
    + '<span>'+esc(v)+'</span></div>';
}

/* ── controls ─────────────────────────────────────────────────────── */
var PLAY_ICON = "&#x25B6;", PAUSE_ICON = "&#x2759;&#x2759;";
function setPlayIcon(){
  document.getElementById("play-btn").innerHTML = serverPaused ? PLAY_ICON : PAUSE_ICON;
}
function control(cmd, extra){
  var body = {cmd: cmd};
  if (extra) for (var k in extra) body[k] = extra[k];
  fetch("/control", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body), cache:"no-store"})
    .catch(function(){});
}
function startGame(){
  var agents = {};
  for (var s = 0; s < NUM_PLAYERS; s++) agents[String(s)] = LINEUP[s];
  matchStart = Date.now(); gameDone = false; doneAt = 0;
  var cfg = {
    mode: HUMAN ? "play" : "watch",
    board_size: BOARD_SIZE, num_players: NUM_PLAYERS, fps: BASE_FPS,
    human_seat: HUMAN ? 0 : null, agents: agents,
    autoplay: true, speed: SPEED
  };
  fetch("/start", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(cfg), cache:"no-store"}).catch(function(){});
  document.getElementById("kbd-hint").classList.toggle("show", HUMAN);
}
document.getElementById("play-btn").addEventListener("click", function(){
  control(serverPaused ? "play" : "pause");
});
document.getElementById("step-btn").addEventListener("click", function(){
  control("step");
});
document.getElementById("reset-btn").addEventListener("click", function(){
  startGame();
});
document.getElementById("speed-sel").addEventListener("change", function(e){
  SPEED = parseFloat(e.target.value);
  control("speed", {speed: SPEED});
});

/* ── header drawers ───────────────────────────────────────────────── */
document.getElementById("menu-btn").addEventListener("click", function(){
  document.getElementById("settings-drawer").classList.remove("show");
  document.getElementById("debug-drawer").classList.toggle("show");
});
document.getElementById("settings-btn").addEventListener("click", function(){
  document.getElementById("debug-drawer").classList.remove("show");
  document.getElementById("settings-drawer").classList.toggle("show");
});
(function(){
  var row = document.getElementById("size-row");
  var btns = row.querySelectorAll(".seg");
  for (var i = 0; i < btns.length; i++){
    btns[i].addEventListener("click", function(e){
      for (var j = 0; j < btns.length; j++) btns[j].classList.remove("on");
      e.target.classList.add("on");
    });
  }
})();
document.getElementById("apply-settings").addEventListener("click", function(){
  var on = document.querySelector("#size-row .seg.on");
  BOARD_SIZE = on ? parseInt(on.dataset.val, 10) : 32;
  HUMAN = document.getElementById("human-chk").checked;
  document.getElementById("settings-drawer").classList.remove("show");
  startGame();
});

/* ── bottom nav ───────────────────────────────────────────────────── */
(function(){
  var items = document.querySelectorAll(".nav-item");
  for (var i = 0; i < items.length; i++){
    items[i].addEventListener("click", function(e){
      var view = e.currentTarget.dataset.view;
      for (var j = 0; j < items.length; j++) items[j].classList.remove("on");
      e.currentTarget.classList.add("on");
      document.getElementById("arena-view").style.display =
        view === "arena" ? "block" : "none";
      document.getElementById("agents-view").classList.toggle("show", view === "agents");
      document.getElementById("history-view").classList.toggle("show", view === "history");
    });
  }
})();

/* ── arrow-key / WASD handler ─────────────────────────────────────── */
var KEY_MAP = {
  ArrowUp:0, ArrowDown:1, ArrowLeft:2, ArrowRight:3,
  w:0, s:1, a:2, d:3, W:0, S:1, A:2, D:3
};
document.addEventListener("keydown", function(e){
  if (!waitingForHuman || actionSubmitted) return;
  var action = KEY_MAP[e.key];
  if (action === undefined) return;
  e.preventDefault();
  actionSubmitted = true;
  fetch("/action", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({action: action}), cache:"no-store"})
    .catch(function(){ actionSubmitted = false; });
});

/* ── polling ──────────────────────────────────────────────────────── */
function poll(){
  fetch("/state?episode=" + EPISODE + "&since=" + FRAMES.length, {cache:"no-store"})
    .then(function(r){ if (!r.ok) throw new Error("http"); return r.json(); })
    .then(function(j){
      if (j.episode !== EPISODE){
        EPISODE = j.episode; FRAMES = [];
        if (j.init) applyInit(j.init);
      }
      var last = null;
      for (var i = 0; i < j.frames.length; i++){
        var f = j.frames[i];
        FRAMES.push(f); trackMoves(f); last = f;
      }
      if (last) renderLast(last);
      serverPaused = !!j.paused;
      setPlayIcon();
      waitingForHuman = !!j.waiting_for_human;
      if (!waitingForHuman) actionSubmitted = false;
    })
    .catch(function(){})
    .then(function(){ setTimeout(poll, 150); });
}
function tickTime(){
  var end = gameDone && doneAt ? doneAt : Date.now();
  if (matchStart) document.getElementById("time-ro").textContent = fmtTime(end - matchStart);
}
setInterval(tickTime, 250);

/* ── boot ─────────────────────────────────────────────────────────── */
fetch("/agents", {cache:"no-store"})
  .then(function(r){ return r.json(); })
  .then(function(j){ PRESETS = j.presets || []; })
  .catch(function(){
    PRESETS = [
      {key:"random", label:"Random", description:"Random walk."},
      {key:"greedy", label:"Greedy", description:"1-ply heuristic."}
    ];
  })
  .then(function(){
    for (var i = 0; i < PRESETS.length; i++) PRESET_MAP[PRESETS[i].key] = PRESETS[i];
    var valid = {};
    for (i = 0; i < PRESETS.length; i++) valid[PRESETS[i].key] = true;
    for (var s = 0; s < LINEUP.length; s++){
      if (!valid[LINEUP[s]]) LINEUP[s] = PRESETS.length ? PRESETS[0].key : "greedy";
    }
    buildSelbar();
    setPlayIcon();
    startGame();
    poll();
  });
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
