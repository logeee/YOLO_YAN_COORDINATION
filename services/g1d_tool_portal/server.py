#!/usr/bin/env python3
"""Small dependency-free directory for G1-D debugging web tools."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.error import HTTPError
from urllib.request import Request, urlopen


from .catalog import TOOLS


STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")


class Portal:
    def __init__(self, box_public_host: str, box_internal_host: str, timeout: float) -> None:
        self.box_public_host = box_public_host
        self.box_internal_host = box_internal_host
        self.timeout = max(0.1, float(timeout))

    @staticmethod
    def _body_public_host(host_header: str) -> str:
        value = (host_header or "").strip()
        if value.startswith("["):
            return value.split("]", 1)[0] + "]"
        return value.rsplit(":", 1)[0] if ":" in value else value or "127.0.0.1"

    def _probe(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        internal_host = "127.0.0.1" if tool["location"] == "body" else self.box_internal_host
        url = f"http://{internal_host}:{tool['port']}{tool['health_path']}"
        started = time.perf_counter()
        try:
            request = Request(url, headers={"User-Agent": "g1d-tool-portal/1.0"})
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(65536)
                status_code = int(response.status)
                content_type = response.headers.get("Content-Type", "")
            state = "online"
            if "json" in content_type.lower() and raw:
                try:
                    payload = json.loads(raw.decode("utf-8", errors="replace"))
                    if isinstance(payload, dict) and payload.get("ok") is False:
                        state = "degraded"
                except Exception:
                    pass
            return {
                "state": state,
                "http_status": status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": None,
            }
        except HTTPError as exc:
            return {
                "state": "degraded",
                "http_status": int(exc.code),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "state": "offline",
                "http_status": None,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": str(exc),
            }

    def catalog(self, host_header: str) -> Dict[str, Any]:
        body_public_host = self._body_public_host(host_header)
        with ThreadPoolExecutor(max_workers=8) as executor:
            probes = list(executor.map(self._probe, TOOLS))
        result = []
        for source, probe in zip(TOOLS, probes):
            item = dict(source)
            public_host = body_public_host if item["location"] == "body" else self.box_public_host
            item["public_host"] = public_host
            item["url"] = f"http://{public_host}:{item['port']}{item['path']}"
            item.update(probe)
            result.append(item)
        return {
            "ok": True,
            "service": "g1d_tool_portal",
            "body_public_host": body_public_host,
            "box_public_host": self.box_public_host,
            "box_internal_host": self.box_internal_host,
            "tools": result,
            "timestamp": time.time(),
        }


def make_handler(portal: Portal):
    class Handler(BaseHTTPRequestHandler):
        server_version = "G1DToolPortal/1.0"

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            assets = {"/static/portal.css": ("portal.css", "text/css; charset=utf-8"),
                      "/static/portal.js": ("portal.js", "text/javascript; charset=utf-8")}
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
            if path == "/":
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
                return
            if path == "/health":
                payload = {"ok": True, "service": "g1d_tool_portal", "port": self.server.server_port}
                self._send(200, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))
                return
            if path == "/api/catalog":
                payload = portal.catalog(self.headers.get("Host", ""))
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
                return
            self._send(404, "application/json; charset=utf-8", b'{"ok":false,"error":"not found"}')

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {self.client_address[0]} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="G1-D debugging tool directory")
    parser.add_argument("--host", default=os.environ.get("G1D_PORTAL_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("G1D_PORTAL_PORT", "18079")))
    parser.add_argument("--box-public-host", default=os.environ.get("G1D_BOX_PUBLIC_HOST", "192.168.61.132"))
    parser.add_argument("--box-internal-host", default=os.environ.get("G1D_BOX_INTERNAL_HOST", "192.168.123.5"))
    parser.add_argument("--probe-timeout", type=float, default=float(os.environ.get("G1D_PORTAL_PROBE_TIMEOUT", "1.2")))
    args = parser.parse_args()
    portal = Portal(args.box_public_host, args.box_internal_host, args.probe_timeout)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(portal))
    print(f"G1-D tool portal listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
