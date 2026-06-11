#!/usr/bin/env python3
"""Small HTTP service for G1D base pose micro-adjustment.

This service is intentionally separate from the YOLO pose service. It reads
YOLO /xyz, builds one safe adjustment step, and only executes when confirm=1.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_YOLO_XYZ_URL = "http://127.0.0.1:18081/xyz"


@dataclass(frozen=True)
class AdjustConfig:
    bind: str = "0.0.0.0"
    port: int = 18084
    yolo_xyz_url: str = DEFAULT_YOLO_XYZ_URL
    sdk_build_dir: Path = Path("/home/unitree/unitree_sdk2/build")
    interface: str = "eth0"
    target_near_edge_forward_mm: float = 200.0
    yaw_tolerance_deg: float = 2.0
    distance_tolerance_mm: float = 15.0
    turn_speed: float = 0.1
    drive_speed: float = 0.1
    min_duration_sec: float = 0.15
    max_duration_sec: float = 0.5
    turn_duration_per_deg: float = 0.035
    drive_duration_per_mm: float = 0.004
    request_timeout_sec: float = 8.0


OVERRIDE_TYPES: dict[str, type] = {
    "label": str,
    "yolo_label": str,
    "target_near_edge_forward_mm": float,
    "yaw_tolerance_deg": float,
    "distance_tolerance_mm": float,
    "turn_speed": float,
    "drive_speed": float,
    "min_duration_sec": float,
    "max_duration_sec": float,
    "turn_duration_per_deg": float,
    "drive_duration_per_mm": float,
}


def _round(value: Any, ndigits: int = 3) -> float | None:
    try:
        return round(float(value), ndigits)
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _request_values(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(handler.path)
    values: dict[str, Any] = {}
    for key, raw_values in urllib.parse.parse_qs(parsed.query).items():
        normalized = key.replace("-", "_")
        if normalized in OVERRIDE_TYPES and raw_values:
            values[normalized] = OVERRIDE_TYPES[normalized](raw_values[-1])
        elif normalized in ("confirm", "execute") and raw_values:
            values[normalized] = _parse_bool(raw_values[-1])

    length = int(handler.headers.get("Content-Length") or 0)
    if length > 0:
        body = handler.rfile.read(length)
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request JSON body must be an object")
        for key, value in data.items():
            normalized = key.replace("-", "_")
            if normalized in OVERRIDE_TYPES:
                values[normalized] = OVERRIDE_TYPES[normalized](value)
            elif normalized in ("confirm", "execute"):
                values[normalized] = _parse_bool(value)
    return values


def _config_with_overrides(config: AdjustConfig, values: dict[str, Any]) -> AdjustConfig:
    allowed = {
        "target_near_edge_forward_mm",
        "yaw_tolerance_deg",
        "distance_tolerance_mm",
        "turn_speed",
        "drive_speed",
        "min_duration_sec",
        "max_duration_sec",
        "turn_duration_per_deg",
        "drive_duration_per_mm",
    }
    updates = {key: values[key] for key in allowed if key in values}
    return replace(config, **updates)


def _url_with_pose_overrides(base_url: str, values: dict[str, Any]) -> str:
    parsed = urllib.parse.urlparse(base_url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    label = values.get("label") or values.get("yolo_label")
    if label:
        query["label"] = str(label)
    new_query = urllib.parse.urlencode(query)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _fetch_pose(config: AdjustConfig, values: dict[str, Any]) -> dict[str, Any]:
    url = _url_with_pose_overrides(config.yolo_xyz_url, values)
    with urllib.request.urlopen(url, timeout=float(config.request_timeout_sec)) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _nested(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _duration_for_error(error: float, per_unit: float, config: AdjustConfig) -> float:
    duration = abs(float(error)) * float(per_unit)
    return round(_clamp(duration, config.min_duration_sec, config.max_duration_sec), 3)


def build_plan(pose: dict[str, Any], config: AdjustConfig) -> dict[str, Any]:
    if not pose.get("ok"):
        return {
            "ok": False,
            "error": pose.get("error") or "YOLO pose is not ok",
            "pose_ok": False,
            "command": None,
        }

    yaw_deg = _nested(pose, "robot_alignment", "control_hint", "box_parallel_yaw_deg")
    near_forward_mm = _nested(pose, "near_edge_robot_alignment", "target", "ground_forward_mm")
    center_forward_mm = _nested(pose, "robot_alignment", "target", "ground_forward_mm")
    near_xyz = pose.get("near_edge_midpoint_xyz_mm")
    if yaw_deg is None:
        return {"ok": False, "error": "missing robot_alignment.control_hint.box_parallel_yaw_deg", "command": None}
    if near_forward_mm is None:
        return {"ok": False, "error": "missing near_edge_robot_alignment.target.ground_forward_mm", "command": None}

    yaw_deg = float(yaw_deg)
    near_forward_mm = float(near_forward_mm)
    distance_error_mm = near_forward_mm - float(config.target_near_edge_forward_mm)

    command: dict[str, Any] | None = None
    if abs(yaw_deg) > float(config.yaw_tolerance_deg):
        action = "turn_left" if yaw_deg > 0.0 else "turn_right"
        command = {
            "phase": "align_box_long_axis",
            "action": action,
            "speed": round(float(config.turn_speed), 3),
            "duration_sec": _duration_for_error(yaw_deg, config.turn_duration_per_deg, config),
            "reason": "make box_parallel_yaw_deg close to 0",
            "error_deg": round(yaw_deg, 3),
        }
    elif abs(distance_error_mm) > float(config.distance_tolerance_mm):
        action = "forward" if distance_error_mm > 0.0 else "back"
        command = {
            "phase": "adjust_near_edge_forward_distance",
            "action": action,
            "speed": round(float(config.drive_speed), 3),
            "duration_sec": _duration_for_error(distance_error_mm, config.drive_duration_per_mm, config),
            "reason": "make near-edge forward distance close to target",
            "error_mm": round(distance_error_mm, 1),
        }
    else:
        command = {
            "phase": "done",
            "action": "none",
            "reason": "yaw and near-edge forward distance are within tolerance",
        }

    return {
        "ok": True,
        "pose_ok": True,
        "selected_yolo_label": pose.get("selected_yolo_label"),
        "selected_orientation": pose.get("selected_orientation"),
        "box_parallel_yaw_deg": round(yaw_deg, 3),
        "near_edge_forward_mm": round(near_forward_mm, 1),
        "center_forward_mm": _round(center_forward_mm, 1),
        "target_near_edge_forward_mm": round(float(config.target_near_edge_forward_mm), 1),
        "distance_error_mm": round(distance_error_mm, 1),
        "near_edge_midpoint_xyz_mm": near_xyz,
        "tolerances": {
            "yaw_tolerance_deg": round(float(config.yaw_tolerance_deg), 3),
            "distance_tolerance_mm": round(float(config.distance_tolerance_mm), 1),
        },
        "command": command,
    }


def _command_argv(config: AdjustConfig, command: dict[str, Any]) -> list[str]:
    binary = config.sdk_build_dir / "bin" / "g1d_simple_control"
    action = str(command.get("action"))
    if action == "stop":
        return [str(binary), config.interface, "stop"]
    return [
        str(binary),
        config.interface,
        action,
        str(command.get("speed")),
        str(command.get("duration_sec")),
    ]


def execute_command(config: AdjustConfig, command: dict[str, Any]) -> dict[str, Any]:
    action = str(command.get("action") or "none")
    if action == "none":
        return {"executed": False, "reason": "no movement needed"}
    argv = _command_argv(config, command)
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=str(config.sdk_build_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(command.get("duration_sec") or 0.0) + 5.0,
        check=False,
    )
    return {
        "executed": True,
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


def make_handler(base_config: AdjustConfig) -> type[BaseHTTPRequestHandler]:
    class AdjustHandler(BaseHTTPRequestHandler):
        server_version = "G1DPoseAdjust/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:
            self._handle()

        def do_POST(self) -> None:
            self._handle()

        def _handle(self) -> None:
            try:
                path = urllib.parse.urlparse(self.path).path
                values = _request_values(self)
                config = _config_with_overrides(base_config, values)
                if path == "/health":
                    _json_response(
                        self,
                        200,
                        {
                            "ok": True,
                            "service": "g1d_pose_adjust_service",
                            "bind": config.bind,
                            "port": config.port,
                            "yolo_xyz_url": config.yolo_xyz_url,
                            "sdk_build_dir": str(config.sdk_build_dir),
                            "interface": config.interface,
                            "endpoints": ["/health", "/plan", "/step", "/stop"],
                            "safety": "/plan never moves; /step and /stop require confirm=1",
                        },
                    )
                    return
                if path == "/plan":
                    pose = _fetch_pose(config, values)
                    plan = build_plan(pose, config)
                    _json_response(self, 200 if plan.get("ok") else 500, {"ok": bool(plan.get("ok")), "plan": plan})
                    return
                if path == "/step":
                    pose = _fetch_pose(config, values)
                    plan = build_plan(pose, config)
                    confirm = bool(values.get("confirm") or values.get("execute"))
                    execution = {"executed": False, "reason": "missing confirm=1"}
                    if confirm and plan.get("ok") and isinstance(plan.get("command"), dict):
                        execution = execute_command(config, plan["command"])
                    _json_response(
                        self,
                        200 if plan.get("ok") else 500,
                        {"ok": bool(plan.get("ok")), "confirmed": confirm, "plan": plan, "execution": execution},
                    )
                    return
                if path == "/stop":
                    confirm = bool(values.get("confirm") or values.get("execute"))
                    command = {"action": "stop"}
                    execution = {"executed": False, "reason": "missing confirm=1"}
                    if confirm:
                        execution = execute_command(config, command)
                    _json_response(self, 200, {"ok": True, "confirmed": confirm, "command": command, "execution": execution})
                    return
                _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {path}"})
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})

    return AdjustHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1D pose micro-adjustment service.")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--yolo-xyz-url", default=DEFAULT_YOLO_XYZ_URL)
    parser.add_argument("--sdk-build-dir", type=Path, default=Path("/home/unitree/unitree_sdk2/build"))
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--target-near-edge-forward-mm", type=float, default=200.0)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--distance-tolerance-mm", type=float, default=15.0)
    parser.add_argument("--turn-speed", type=float, default=0.1)
    parser.add_argument("--drive-speed", type=float, default=0.1)
    parser.add_argument("--min-duration-sec", type=float, default=0.15)
    parser.add_argument("--max-duration-sec", type=float, default=0.5)
    parser.add_argument("--turn-duration-per-deg", type=float, default=0.035)
    parser.add_argument("--drive-duration-per-mm", type=float, default=0.004)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = AdjustConfig(
        bind=args.bind,
        port=args.port,
        yolo_xyz_url=args.yolo_xyz_url,
        sdk_build_dir=args.sdk_build_dir,
        interface=args.interface,
        target_near_edge_forward_mm=args.target_near_edge_forward_mm,
        yaw_tolerance_deg=args.yaw_tolerance_deg,
        distance_tolerance_mm=args.distance_tolerance_mm,
        turn_speed=args.turn_speed,
        drive_speed=args.drive_speed,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        turn_duration_per_deg=args.turn_duration_per_deg,
        drive_duration_per_mm=args.drive_duration_per_mm,
    )
    server = ThreadingHTTPServer((config.bind, config.port), make_handler(config))
    print(f"serving on http://{config.bind}:{config.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
