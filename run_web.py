#!/usr/bin/env python3
"""Start the dashboard.

    python run_web.py                 # http://127.0.0.1:8000
    python run_web.py --port 9000
    python run_web.py --host 0.0.0.0  # only behind HTTPS; see the security note

Binding to 0.0.0.0 exposes a credential vault to your network. Do that only
behind a TLS-terminating proxy, and set JP_INSECURE_COOKIES back to false.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the job pipeline dashboard.")
    parser.add_argument("--host", default=settings.WEB_HOST)
    parser.add_argument("--port", type=int, default=settings.WEB_PORT)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost") and settings.COOKIE_SECURE is False:
        print("Refusing to bind a non-loopback host with insecure cookies enabled.\n"
              "Unset JP_INSECURE_COOKIES and put HTTPS in front of this.", file=sys.stderr)
        return 2

    import uvicorn

    settings.load_or_create_secret_key()   # generate before workers start
    print(f"Dashboard on http://{args.host}:{args.port}")
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
