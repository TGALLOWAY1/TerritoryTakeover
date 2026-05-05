"""Launch the interactive TerritoryTakeover frontend.

Opens a browser-based UI where you can:
  - Watch agents of configurable strength play each other, or
  - Take control of seat 1 with arrow keys against AI opponents (Tron-style).

Usage::

    # Default: localhost:8000, auto-opens browser.
    python scripts/play_interactive.py

    # Custom port, no auto-open.
    python scripts/play_interactive.py --port 8765 --no-browser

    # Bind to all interfaces (useful for remote access).
    python scripts/play_interactive.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve the interactive TerritoryTakeover frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Once running, open the printed URL in a browser.\n"
            "Use the setup form to choose mode, board size, and opponent strength."
        ),
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="TCP port (default: 8000)")
    p.add_argument("--title", default="TerritoryTakeover", help="Page title")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the default browser.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from territory_takeover.viz_interactive import InteractiveServer

    server = InteractiveServer(host=args.host, port=args.port, title=args.title)
    server.start()

    print(f"[interactive] serving at {server.url}")
    print("[interactive] use the setup form in the browser to start a game")
    print("[interactive] press Ctrl-C to stop")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(server.url)).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[interactive] stopping…")
    finally:
        server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
