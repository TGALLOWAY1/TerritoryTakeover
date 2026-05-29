"""Launch the Territory Takeover *Arena* frontend.

Opens a browser-based UI that auto-starts a four-agent match where you can:
  - Watch agents battle, swapping any agent's strategy from a dropdown,
  - Drive the simulation with play / pause / step / speed / reset, and
  - Optionally take control of seat 1 with arrow keys / WASD (Settings → Play).

Usage::

    # Default: localhost:8000, auto-opens browser.
    python scripts/play_interactive.py

    # Custom port, no auto-open.
    python scripts/play_interactive.py --port 8765 --no-browser

    # Bind to all interfaces (useful for remote access).
    python scripts/play_interactive.py --host 0.0.0.0 --port 8000

Environment variables (handy for container/cloud deploys):
  TT_HOST          default bind address (overrides 127.0.0.1)
  PORT             default TCP port (set by most PaaS platforms)
  TT_ARENA_TOKEN   shared access token; when set, every request must present it
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve the Territory Takeover Arena frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Once running, open the printed URL in a browser.\n"
            "A match auto-starts; swap agents from the selector bar and use the\n"
            "play / pause / step / speed / reset controls below the board."
        ),
    )
    p.add_argument(
        "--host",
        default=os.environ.get("TT_HOST", "127.0.0.1"),
        help="Bind address (default: 127.0.0.1, or $TT_HOST)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="TCP port (default: 8000, or $PORT)",
    )
    p.add_argument("--title", default="Territory Takeover", help="Page title")
    p.add_argument(
        "--token",
        default=os.environ.get("TT_ARENA_TOKEN"),
        help="Shared access token; when set, every request must present it "
        "(default: $TT_ARENA_TOKEN).",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the default browser.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from territory_takeover.viz_interactive import InteractiveServer

    server = InteractiveServer(
        host=args.host, port=args.port, title=args.title, token=args.token
    )
    server.start()

    print(f"[arena] serving at {server.url}")
    print("[arena] a match auto-starts; swap agents and use the controls in the browser")
    if args.token:
        print("[arena] access token required — open with ?token=<token>")
    print("[arena] press Ctrl-C to stop")

    if not args.no_browser:
        open_url = server.url + (f"?token={args.token}" if args.token else "")
        threading.Timer(0.4, lambda: webbrowser.open(open_url)).start()

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
