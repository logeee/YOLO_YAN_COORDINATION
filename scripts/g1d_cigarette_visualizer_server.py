#!/usr/bin/env python3
"""Serve the G1-D cigarette relative-position visualizer."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = 18085
DEFAULT_XYZ_URL = "http://127.0.0.1:18081/xyz"
DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
VIEWER_DIR = Path(__file__).resolve().parents[1] / "visualization" / "g1d_cigarette_viewer"
DEFAULT_ROBOT_STATE: dict[str, Any] = {
    "ok": True,
    "source": "visualizer_default",
    "column_extension_mm": 420.0,
    "joints": {
        "LZ_mt_Joint": 0.21,
        "LZ_it_Joint": 0.21,
    },
}


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _fetch_json(url: str, timeout_sec: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=float(timeout_sec)) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _parse_rostopic_joint_state_csv(text: str, topic: str) -> dict[str, Any]:
    rows = [row for row in csv.reader(text.splitlines()) if row]
    if len(rows) < 2:
        raise RuntimeError(f"no joint state sample from {topic}")
    headers = rows[0]
    values = rows[1]
    field_values = {header: values[index] for index, header in enumerate(headers) if index < len(values)}
    names: list[str] = []
    positions: list[float] = []
    velocities: list[float] = []
    efforts: list[float] = []

    for header, name in field_values.items():
        if not header.startswith("field.name"):
            continue
        suffix = header.removeprefix("field.name")
        if not name:
            continue
        position_text = field_values.get(f"field.position{suffix}")
        if position_text in (None, ""):
            continue
        names.append(name)
        positions.append(float(position_text))
        velocity_text = field_values.get(f"field.velocity{suffix}")
        effort_text = field_values.get(f"field.effort{suffix}")
        if velocity_text not in (None, ""):
            velocities.append(float(velocity_text))
        if effort_text not in (None, ""):
            efforts.append(float(effort_text))

    if not names:
        raise RuntimeError(f"joint state sample from {topic} did not contain name/position arrays")

    joint_states: dict[str, Any] = {"name": names, "position": positions}
    if len(velocities) == len(names):
        joint_states["velocity"] = velocities
    if len(efforts) == len(names):
        joint_states["effort"] = efforts
    return {
        "ok": True,
        "source": f"ros1:{topic}",
        "joint_states": joint_states,
        "joints": dict(zip(names, positions)),
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
    }


def _read_ros_joint_states(topic: str, timeout_sec: float) -> dict[str, Any]:
    rostopic = shutil.which("rostopic")
    if not rostopic:
        raise RuntimeError("rostopic not found; source ROS setup.bash before starting the visualizer")
    completed = subprocess.run(
        [rostopic, "echo", "-n", "1", "-p", topic],
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_sec),
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(message or f"rostopic echo failed for {topic}")
    return _parse_rostopic_joint_state_csv(completed.stdout, topic)


def _merge_query(url: str, params: dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    query.update({key: value for key, value in params.items() if value})
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def _default_robot_state(warning: str | None = None) -> dict[str, Any]:
    payload = json.loads(json.dumps(DEFAULT_ROBOT_STATE))
    payload["updated_at"] = datetime.now().isoformat(timespec="milliseconds")
    if warning:
        payload["warning"] = warning
    return payload


def _read_robot_state(
    robot_state_url: str | None,
    robot_state_file: Path | None,
    joint_states_topic: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    if robot_state_url:
        payload = _fetch_json(robot_state_url, timeout_sec)
        payload.setdefault("source", robot_state_url)
        return payload
    if robot_state_file and robot_state_file.exists():
        payload = json.loads(robot_state_file.read_text(encoding="utf-8"))
        payload.setdefault("source", str(robot_state_file))
        return payload
    if joint_states_topic:
        try:
            return _read_ros_joint_states(joint_states_topic, timeout_sec)
        except Exception as exc:
            return _default_robot_state(f"joint states unavailable: {exc}")
    return _default_robot_state()


def make_handler(
    default_xyz_url: str,
    timeout_sec: float,
    robot_state_url: str | None,
    robot_state_file: Path | None,
    joint_states_topic: str | None,
) -> type[SimpleHTTPRequestHandler]:
    class VisualizerHandler(SimpleHTTPRequestHandler):
        server_version = "G1DCigaretteVisualizer/1.0"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "service": "g1d_cigarette_visualizer",
                        "viewer_dir": str(VIEWER_DIR),
                        "default_xyz_url": default_xyz_url,
                        "robot_state_url": robot_state_url,
                        "robot_state_file": str(robot_state_file) if robot_state_file else None,
                        "joint_states_topic": joint_states_topic,
                    },
                )
                return
            if parsed.path == "/api/xyz":
                query = urllib.parse.parse_qs(parsed.query)
                source_url = query.get("url", [default_xyz_url])[-1]
                label = query.get("label", [""])[-1]
                request_url = _merge_query(source_url, {"label": label})
                try:
                    payload = _fetch_json(request_url, timeout_sec)
                    _json_response(self, 200 if payload.get("ok", True) else 502, {"ok": True, "url": request_url, "pose": payload})
                except Exception as exc:
                    _json_response(self, 502, {"ok": False, "url": request_url, "error": str(exc)})
                return
            if parsed.path == "/api/robot_state":
                query = urllib.parse.parse_qs(parsed.query)
                source_url = query.get("url", [robot_state_url or ""])[-1] or None
                topic = query.get("joint_states_topic", [joint_states_topic or ""])[-1] or None
                try:
                    payload = _read_robot_state(source_url, robot_state_file, topic, timeout_sec)
                    _json_response(self, 200 if payload.get("ok", True) else 502, {"ok": True, "state": payload})
                except Exception as exc:
                    _json_response(self, 502, {"ok": False, "error": str(exc)})
                return
            super().do_GET()

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    return VisualizerHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the G1-D cigarette visualization page.")
    parser.add_argument("--bind", default=DEFAULT_BIND)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--xyz-url", default=DEFAULT_XYZ_URL)
    parser.add_argument("--robot-state-url", default="")
    parser.add_argument("--robot-state-file", type=Path)
    parser.add_argument("--joint-states-topic", default=DEFAULT_JOINT_STATES_TOPIC)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not VIEWER_DIR.exists():
        raise FileNotFoundError(f"viewer directory not found: {VIEWER_DIR}")
    server = ThreadingHTTPServer(
        (args.bind, int(args.port)),
        make_handler(
            args.xyz_url,
            args.timeout_sec,
            args.robot_state_url or None,
            args.robot_state_file,
            args.joint_states_topic or None,
        ),
    )
    print(f"serving G1-D cigarette visualizer on http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
