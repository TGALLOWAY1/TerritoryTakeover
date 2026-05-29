# Backlog

Loosely-prioritized ideas not yet scheduled.

## Arena front end

- **Touch play on mobile.** The Arena is watch-friendly on phones, but the
  human-seat mode (Settings → "Play seat 1 myself") is **keyboard-only**: input
  is read from `KEY_MAP` (arrow keys / WASD) in `src/territory_takeover/viz_interactive.py`.
  Add touch controls so a seat can be played from a phone:
  - an on-screen D-pad overlay (4 large tap targets), and/or
  - swipe-to-turn gestures on the board canvas,
  - posting the chosen direction to the existing `POST /action` endpoint
    (`{"action": 0|1|2|3}`) — no engine changes needed.
  Show the touch controls only when a human seat is active and hide the
  keyboard hint on touch devices.
