"""Interactive HTML viewer for recorded games.

Where :mod:`territory_takeover.viz` produces static assets (ASCII, matplotlib
figures, animated GIF), this module emits a single self-contained HTML page
that replays a full game trajectory in the browser. Each frame carries its
own per-player stats and (optionally) a live win-probability vector, so the
page can show:

- the grid rendered with the same palette as the GIF renderer
  (see :data:`territory_takeover.viz.TILE_COLORS`),
- an agent card per seat (name, strategy label, Elo rating),
- a win-probability bar per player that updates each turn,
- playback controls (play/pause, step, scrub, FPS).

The output is a single HTML file with no external assets and no runtime
dependencies beyond the Python standard library. Win probabilities are
precomputed on the Python side and embedded as a JSON blob; the browser only
reads the data and paints it.

Two win-probability helpers are bundled for the common cases:

- :func:`heuristic_win_probs` — softmax over
  :class:`territory_takeover.eval.heuristic.LinearEvaluator` scores; works
  out of the box for any game.
- :func:`alphazero_win_probs` — value-head output from a loaded
  :class:`territory_takeover.rl.alphazero.evaluator.NNEvaluator`; opt-in.

Callers that want a different estimator can compute their own probability
vectors and pass them directly to :func:`save_game_html`.
"""

from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from territory_takeover.eval.heuristic import DEAD_SENTINEL, LinearEvaluator
from territory_takeover.viz import HEAD_EDGE_COLORS, TILE_COLORS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from territory_takeover.rl.alphazero.evaluator import NNEvaluator
    from territory_takeover.state import GameState


@dataclass(frozen=True, slots=True)
class AgentCard:
    """Display-only metadata for one seat in the rendered game.

    ``elo`` may be ``None`` (the UI renders it as ``—``) when the agent
    does not appear in the Elo CSV or no CSV was supplied.
    """

    seat: int
    name: str
    strategy: str
    elo: float | None


def heuristic_win_probs(
    state: GameState,
    evaluator: LinearEvaluator,
    temperature: float = 5.0,
) -> NDArray[np.float64]:
    """Softmax over :class:`LinearEvaluator` scores → probability vector.

    Terminal states short-circuit to the true outcome (``state.winner`` gets
    1.0, or uniform if the game ended in a tie). For live states, dead players
    get probability zero and the remaining mass is renormalised so the result
    always sums to ``1.0``. Temperature ``T`` controls sharpness — higher ``T``
    flattens, lower ``T`` sharpens. The default ``T=5.0`` is a calibration-
    free starting point that matches the relative magnitudes of the default
    weight set in :func:`territory_takeover.eval.heuristic.default_evaluator`.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}")

    num_players = len(state.players)
    if state.done:
        if state.winner is None:
            return np.full(num_players, 1.0 / num_players, dtype=np.float64)
        probs = np.zeros(num_players, dtype=np.float64)
        probs[state.winner] = 1.0
        return probs

    scores = evaluator.evaluate(state)
    alive_mask = scores > DEAD_SENTINEL / 2
    if not alive_mask.any():
        return np.full(num_players, 1.0 / num_players, dtype=np.float64)

    logits = scores.astype(np.float64) / float(temperature)
    logits[~alive_mask] = -np.inf
    logits -= float(np.max(logits[alive_mask]))
    probs = np.exp(logits)
    probs[~alive_mask] = 0.0
    total = float(probs.sum())
    if total <= 0.0:
        uniform = np.zeros(num_players, dtype=np.float64)
        uniform[alive_mask] = 1.0 / int(alive_mask.sum())
        return uniform
    return probs / total


def alphazero_win_probs(
    state: GameState,
    evaluator: NNEvaluator,
    active_player: int,
) -> NDArray[np.float64]:
    """Map an AlphaZero value head's per-seat tanh output to a probability vector.

    Terminal states short-circuit to the true outcome (same convention as
    :func:`heuristic_win_probs`). Otherwise, the network emits values in
    ``[-1, 1]`` (see :class:`territory_takeover.rl.alphazero.evaluator.NNEvaluator`);
    we map them into ``[0, 1]`` via ``(v + 1) / 2`` and renormalise so the
    result sums to one. Scalar-head networks that emit a single value are
    broadcast to every seat before normalisation.
    """
    from territory_takeover.actions import legal_action_mask

    num_players = len(state.players)
    if state.done:
        if state.winner is None:
            return np.full(num_players, 1.0 / num_players, dtype=np.float64)
        probs_terminal = np.zeros(num_players, dtype=np.float64)
        probs_terminal[state.winner] = 1.0
        return probs_terminal

    mask = legal_action_mask(state, active_player)
    _, value = evaluator.evaluate(state, active_player, mask)
    v = np.asarray(value, dtype=np.float64)
    if v.shape == (1,):
        v = np.full(num_players, float(v[0]), dtype=np.float64)
    probs = np.clip((v + 1.0) / 2.0, 0.0, None)
    for p in state.players:
        if not p.alive:
            probs[p.player_id] = 0.0
    total = float(probs.sum())
    if total <= 0.0:
        return np.full(num_players, 1.0 / num_players, dtype=np.float64)
    return probs / total


def load_elo_csv(path: str | Path) -> dict[str, float]:
    """Load an Elo ratings CSV of the form ``agent,elo_vs_anchor,anchor``.

    Returns a mapping ``agent_name -> elo``. Unknown or malformed rows are
    skipped silently — callers that need strict parsing should use a dedicated
    loader. Missing files raise :class:`FileNotFoundError` so the CLI can
    surface the problem clearly.
    """
    ratings: dict[str, float] = {}
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("agent")
            elo_str = row.get("elo_vs_anchor")
            if name is None or elo_str is None:
                continue
            try:
                ratings[name] = float(elo_str)
            except ValueError:
                continue
    return ratings


def _frame_payload(
    state: GameState,
    win_probs: NDArray[np.float64],
) -> dict[str, object]:
    """Serialise one frame to a JSON-safe dict."""
    num_players = len(state.players)
    if win_probs.shape != (num_players,):
        raise ValueError(
            f"win_probs shape {win_probs.shape} does not match num_players={num_players}"
        )
    grid_flat: list[int] = state.grid.flatten().tolist()
    claimed: list[int] = [p.claimed_count for p in state.players]
    path_len: list[int] = [len(p.path) for p in state.players]
    alive: list[bool] = [bool(p.alive) for p in state.players]
    heads: list[list[int]] = [[p.head[0], p.head[1]] for p in state.players]
    probs: list[float] = [float(x) for x in win_probs]
    return {
        "grid": grid_flat,
        "claimed": claimed,
        "path_len": path_len,
        "alive": alive,
        "heads": heads,
        "win_probs": probs,
        "turn": int(state.turn_number),
        "current_player": int(state.current_player),
        "done": bool(state.done),
        "winner": state.winner,
    }


def build_payload(
    trajectory: list[GameState],
    agent_cards: list[AgentCard],
    win_probs_per_frame: list[NDArray[np.float64]],
    title: str,
    fps: int,
) -> dict[str, object]:
    """Assemble the full JSON payload embedded in the HTML.

    Separated from :func:`save_game_html` so tests (and future alternative
    transports like a Flask/FastAPI endpoint) can reuse the same serialiser.
    """
    if not trajectory:
        raise ValueError("trajectory is empty; nothing to render")
    if len(win_probs_per_frame) != len(trajectory):
        raise ValueError(
            f"win_probs_per_frame length {len(win_probs_per_frame)} "
            f"!= trajectory length {len(trajectory)}"
        )
    if fps < 1:
        raise ValueError(f"fps must be >= 1; got {fps}")

    first = trajectory[0]
    board_h, board_w = first.grid.shape
    num_players = len(first.players)
    if len(agent_cards) != num_players:
        raise ValueError(
            f"agent_cards length {len(agent_cards)} "
            f"!= num_players {num_players}"
        )

    agents_payload: list[dict[str, object]] = []
    for card in agent_cards:
        elo_field: float | None = (
            None
            if card.elo is None or math.isnan(card.elo)
            else float(card.elo)
        )
        agents_payload.append(
            {
                "seat": int(card.seat),
                "name": card.name,
                "strategy": card.strategy,
                "elo": elo_field,
            }
        )

    frames_payload: list[dict[str, object]] = [
        _frame_payload(state, probs)
        for state, probs in zip(trajectory, win_probs_per_frame, strict=True)
    ]

    return {
        "title": title,
        "board_height": int(board_h),
        "board_width": int(board_w),
        "num_players": num_players,
        "fps": int(fps),
        "tile_colors": list(TILE_COLORS),
        "head_edge_colors": list(HEAD_EDGE_COLORS),
        "agents": agents_payload,
        "frames": frames_payload,
    }


def save_game_html(
    trajectory: list[GameState],
    agent_cards: list[AgentCard],
    win_probs_per_frame: list[NDArray[np.float64]],
    path: str | Path,
    title: str = "TerritoryTakeover",
    fps: int = 4,
) -> None:
    """Write a self-contained interactive HTML viewer for ``trajectory``.

    ``trajectory`` is a list of :class:`GameState` snapshots (one per half-
    move — see ``scripts/record_demo.play_demo_game``). ``agent_cards`` and
    ``win_probs_per_frame`` carry the per-seat and per-frame metadata the
    viewer needs; see :class:`AgentCard` and :func:`heuristic_win_probs` /
    :func:`alphazero_win_probs` for ready-made producers.

    The emitted file has no external assets — open it directly in a browser.
    """
    payload = build_payload(
        trajectory=trajectory,
        agent_cards=agent_cards,
        win_probs_per_frame=win_probs_per_frame,
        title=title,
        fps=fps,
    )
    html_text = render_html(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


def render_html(payload: dict[str, object]) -> str:
    """Inline ``payload`` (built by :func:`build_payload`) into the HTML template."""
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # ``</script>`` inside a JSON string payload would close the embedding
    # script tag early; the standard escape is to split the sequence.
    data_json = data_json.replace("</", "<\\/")
    title = str(payload.get("title", "TerritoryTakeover"))
    # Replace the data placeholder first so an escaped title can never
    # accidentally introduce the data marker into the output.
    out = _TEMPLATE.replace(_DATA_MARKER, data_json)
    out = out.replace(_TITLE_MARKER, html.escape(title))
    return out


_TITLE_MARKER: Final[str] = "__TT_TITLE_MARKER_5c8f__"
_DATA_MARKER: Final[str] = "__TT_DATA_MARKER_5c8f__"


_TEMPLATE: Final[str] = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__TT_TITLE_MARKER_5c8f__</title>
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
  input[type=range] { flex: 1 1 200px; }
  label { font-size: 12px; color: #9aa0a6; }
  .turn-readout { font-size: 12px; color: #c4c7cc; min-width: 110px; }
  .agent-card {
    background: #1a1d23;
    border: 1px solid #2a2f38;
    border-left-width: 6px;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 10px;
  }
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
    transition: width 240ms ease-out, background-color 240ms ease-out;
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
    <h1>__TT_TITLE_MARKER_5c8f__</h1>
    <div class="meta" id="meta"></div>
    <canvas id="board"></canvas>
    <div class="controls">
      <button id="prev" title="Previous frame">&laquo;</button>
      <button id="play" title="Play/pause">Play</button>
      <button id="next" title="Next frame">&raquo;</button>
      <input type="range" id="scrub" min="0" max="0" step="1" value="0" />
      <span class="turn-readout" id="turn-readout"></span>
      <label>FPS <select id="fps-select">
        <option>2</option><option>4</option><option>8</option><option>16</option>
      </select></label>
    </div>
    <div class="winner-banner" id="winner-banner">Game in progress.</div>
  </div>
  <div class="side-col" id="side-col"></div>
</div>
<script id="tt-data" type="application/json">__TT_DATA_MARKER_5c8f__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("tt-data").textContent);
  var TILE = DATA.tile_colors;
  var EDGE = DATA.head_edge_colors;
  var H = DATA.board_height, W = DATA.board_width;
  var NUM = DATA.num_players;
  var FRAMES = DATA.frames;
  var AGENTS = DATA.agents;

  var canvas = document.getElementById("board");
  var ctx = canvas.getContext("2d");
  var CELL = 24;
  canvas.width = W * CELL;
  canvas.height = H * CELL;

  function drawFrame(f) {
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
    for (var p = 0; p < NUM; p++) {
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

  // Side panel
  var sideCol = document.getElementById("side-col");
  var cards = [];
  for (var s = 0; s < NUM; s++) {
    var agent = AGENTS[s];
    var pathColor = TILE[s + 1];
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
    cards.push({
      root: card,
      status: card.querySelector(".status"),
      claimed: card.querySelector(".claimed"),
      pathLen: card.querySelector(".path-len"),
      pct: card.querySelector(".prob-pct"),
      fill: card.querySelector(".prob-fill")
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {
        "&": "&amp;", "<": "&lt;", ">": "&gt;",
        '"': "&quot;", "'": "&#39;"
      }[c];
    });
  }

  function updateSide(f) {
    for (var s = 0; s < NUM; s++) {
      var c = cards[s];
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
    }
  }

  var meta = document.getElementById("meta");
  meta.textContent = H + "x" + W + " board, " + NUM + " players, "
      + FRAMES.length + " frames";

  var banner = document.getElementById("winner-banner");
  var turnReadout = document.getElementById("turn-readout");
  var scrub = document.getElementById("scrub");
  scrub.max = String(FRAMES.length - 1);

  var idx = 0;
  function render() {
    var f = FRAMES[idx];
    drawFrame(f);
    updateSide(f);
    turnReadout.textContent = "Turn " + f.turn
        + " · Frame " + (idx + 1) + "/" + FRAMES.length;
    scrub.value = String(idx);
    if (f.done) {
      banner.classList.add("active");
      if (f.winner === null || f.winner === undefined) {
        banner.textContent = "Game over — tie.";
      } else {
        var w = AGENTS[f.winner];
        banner.textContent = "Winner: " + w.name + " (seat " + (f.winner + 1) + ")";
      }
    } else {
      banner.classList.remove("active");
      banner.textContent = "Game in progress.";
    }
  }

  var playing = false;
  var fps = DATA.fps;
  var timer = null;
  function stop() {
    playing = false;
    document.getElementById("play").textContent = "Play";
    if (timer) { clearInterval(timer); timer = null; }
  }
  function start() {
    if (idx >= FRAMES.length - 1) idx = 0;
    playing = true;
    document.getElementById("play").textContent = "Pause";
    timer = setInterval(function () {
      if (idx >= FRAMES.length - 1) { stop(); return; }
      idx += 1;
      render();
    }, Math.max(30, Math.round(1000 / fps)));
  }

  document.getElementById("play").onclick = function () {
    if (playing) stop(); else start();
  };
  document.getElementById("prev").onclick = function () {
    stop();
    if (idx > 0) { idx -= 1; render(); }
  };
  document.getElementById("next").onclick = function () {
    stop();
    if (idx < FRAMES.length - 1) { idx += 1; render(); }
  };
  scrub.oninput = function () {
    stop();
    idx = parseInt(scrub.value, 10) || 0;
    render();
  };
  var fpsSelect = document.getElementById("fps-select");
  // Pick the closest preset to the payload's fps.
  var options = Array.prototype.map.call(fpsSelect.options, function (o) {
    return parseInt(o.value, 10);
  });
  var best = options[0], bestDiff = Math.abs(options[0] - fps);
  for (var k = 1; k < options.length; k++) {
    var d = Math.abs(options[k] - fps);
    if (d < bestDiff) { best = options[k]; bestDiff = d; }
  }
  fpsSelect.value = String(best);
  fps = best;
  fpsSelect.onchange = function () {
    fps = parseInt(fpsSelect.value, 10) || 4;
    if (playing) { stop(); start(); }
  };

  render();
})();
</script>
</body>
</html>
"""


__all__ = [
    "AgentCard",
    "alphazero_win_probs",
    "build_payload",
    "heuristic_win_probs",
    "load_elo_csv",
    "render_html",
    "save_game_html",
]
