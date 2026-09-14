#!/usr/bin/env python3
"""Keep the legacy localhost:18080 API while ES80Z runs on the companion box."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


STATIC_DIR = Path(__file__).resolve().parent / "static"
UI_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def make_handler(upstream: str, timeout: float):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SuctionAPIProxy/1.0"

        def send_html(self) -> None:
            payload = UI_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def proxy(self) -> None:
            body = None
            if self.command in {"POST", "PUT", "PATCH"}:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request = Request(
                upstream.rstrip("/") + self.path,
                data=body,
                method=self.command,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
            except HTTPError as exc:
                payload = exc.read()
                status = exc.code
                content_type = exc.headers.get("Content-Type", "application/json")
            except (URLError, TimeoutError, OSError) as exc:
                payload = json.dumps(
                    {"ok": False, "service": "suction-api-proxy", "error": f"suction backend unavailable: {exc}"},
                    separators=(",", ":"),
                ).encode("utf-8")
                status = 502
                content_type = "application/json"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            assets = {"/static/suction.css": ("suction.css", "text/css; charset=utf-8"),
                      "/static/suction.js": ("suction.js", "text/javascript; charset=utf-8")}
            if path in assets:
                filename, content_type = assets[path]
                payload = (STATIC_DIR / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if path in {"/", "/ui", "/index.html"}:
                self.send_html()
                return
            self.proxy()

        do_POST = proxy

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SUCTION_PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SUCTION_PROXY_PORT", "18080")))
    parser.add_argument("--upstream", default=os.environ.get("SUCTION_PROXY_UPSTREAM", "http://192.168.123.5:18089"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("SUCTION_PROXY_TIMEOUT", "30")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.upstream, args.timeout))
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
