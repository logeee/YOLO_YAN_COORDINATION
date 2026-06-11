#!/usr/bin/env python3
"""Small HTTP service for G1D base pose micro-adjustment.

This service is intentionally separate from the YOLO pose service. It reads
YOLO /xyz, builds small base-control commands, and exposes both single-step
and combined adjustment endpoints.
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
from datetime import datetime
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
    max_duration_sec: float = 5.0
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


def _start_ndjson_response(handler: BaseHTTPRequestHandler, status: int = 200) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()


def _write_ndjson(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    handler.wfile.write(data)
    handler.wfile.flush()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _request_values(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(handler.path)
    values: dict[str, Any] = {}
    for key, raw_values in urllib.parse.parse_qs(parsed.query).items():
        normalized = key.replace("-", "_")
        if normalized in OVERRIDE_TYPES and raw_values:
            values[normalized] = OVERRIDE_TYPES[normalized](raw_values[-1])
        elif normalized in ("confirm", "execute", "dry_run", "stream") and raw_values:
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
            elif normalized in ("confirm", "execute", "dry_run", "stream"):
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


def _duration_for_turn_deg(yaw_deg: float, config: AdjustConfig) -> float:
    speed = max(abs(float(config.turn_speed)), 1e-6)
    duration = math.radians(abs(float(yaw_deg))) / speed
    return round(_clamp(duration, config.min_duration_sec, config.max_duration_sec), 3)


def _duration_for_distance_mm(distance_error_mm: float, config: AdjustConfig) -> float:
    speed = max(abs(float(config.drive_speed)), 1e-6)
    duration = abs(float(distance_error_mm)) / 1000.0 / speed
    return round(_clamp(duration, config.min_duration_sec, config.max_duration_sec), 3)


def _pose_metrics(pose: dict[str, Any], config: AdjustConfig) -> dict[str, Any]:
    if not pose.get("ok"):
        return {
            "ok": False,
            "error": pose.get("error") or "YOLO pose is not ok",
            "pose_ok": False,
        }

    yaw_deg = _nested(pose, "robot_alignment", "control_hint", "box_parallel_yaw_deg")
    near_forward_mm = _nested(pose, "near_edge_robot_alignment", "target", "ground_forward_mm")
    near_right_mm = _nested(pose, "near_edge_robot_alignment", "target", "right_mm")
    center_forward_mm = _nested(pose, "robot_alignment", "target", "ground_forward_mm")
    near_xyz = pose.get("near_edge_midpoint_xyz_mm")
    if yaw_deg is None:
        return {"ok": False, "error": "missing robot_alignment.control_hint.box_parallel_yaw_deg"}
    if near_forward_mm is None:
        return {"ok": False, "error": "missing near_edge_robot_alignment.target.ground_forward_mm"}

    yaw_deg = float(yaw_deg)
    near_forward_mm = float(near_forward_mm)
    if near_right_mm is None and isinstance(near_xyz, list) and near_xyz:
        near_right_mm = near_xyz[0]
    distance_error_mm = near_forward_mm - float(config.target_near_edge_forward_mm)
    return {
        "ok": True,
        "pose_ok": True,
        "selected_yolo_label": pose.get("selected_yolo_label"),
        "selected_orientation": pose.get("selected_orientation"),
        "box_parallel_yaw_deg": round(yaw_deg, 3),
        "near_edge_forward_mm": round(near_forward_mm, 1),
        "near_edge_right_mm": _round(near_right_mm, 1),
        "center_forward_mm": _round(center_forward_mm, 1),
        "target_near_edge_forward_mm": round(float(config.target_near_edge_forward_mm), 1),
        "distance_error_mm": round(distance_error_mm, 1),
        "near_edge_midpoint_xyz_mm": near_xyz,
        "tolerances": {
            "yaw_tolerance_deg": round(float(config.yaw_tolerance_deg), 3),
            "distance_tolerance_mm": round(float(config.distance_tolerance_mm), 1),
        },
    }


def _planned_turn_yaw_rad(command: dict[str, Any] | None) -> float:
    if not command or command.get("phase") != "align_box_long_axis":
        return 0.0
    action = str(command.get("action") or "")
    sign = 1.0 if action == "turn_left" else -1.0 if action == "turn_right" else 0.0
    return sign * abs(float(command.get("speed") or 0.0)) * abs(float(command.get("duration_sec") or 0.0))


def _metrics_after_planned_turn(
    metrics: dict[str, Any],
    config: AdjustConfig,
    yaw_command: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(metrics)
    raw_error_mm = float(metrics["distance_error_mm"])
    updated["control_distance_error_mm"] = round(raw_error_mm, 1)
    updated["predicted_after_turn_forward_mm"] = metrics.get("near_edge_forward_mm")
    updated["forward_delta_from_planned_turn_mm"] = 0.0
    planned_yaw_rad = _planned_turn_yaw_rad(yaw_command)
    updated["planned_turn_yaw_deg"] = round(math.degrees(planned_yaw_rad), 3)
    updated["planned_turn_yaw_rad"] = round(planned_yaw_rad, 6)

    if planned_yaw_rad == 0.0:
        return updated

    right_mm = metrics.get("near_edge_right_mm")
    if right_mm is None:
        return updated

    forward_mm = float(metrics["near_edge_forward_mm"])
    predicted_forward_mm = forward_mm * math.cos(planned_yaw_rad) - float(right_mm) * math.sin(planned_yaw_rad)
    forward_delta_mm = predicted_forward_mm - forward_mm
    control_error_mm = predicted_forward_mm - float(config.target_near_edge_forward_mm)
    updated["control_distance_error_mm"] = round(control_error_mm, 1)
    updated["forward_delta_from_planned_turn_mm"] = round(forward_delta_mm, 1)
    updated["predicted_after_turn_forward_mm"] = round(predicted_forward_mm, 1)
    return updated


def _yaw_command(metrics: dict[str, Any], config: AdjustConfig) -> dict[str, Any] | None:
    yaw_deg = float(metrics["box_parallel_yaw_deg"])
    if abs(yaw_deg) <= float(config.yaw_tolerance_deg):
        return None
    action = "turn_left" if yaw_deg > 0.0 else "turn_right"
    return {
        "phase": "align_box_long_axis",
        "action": action,
        "speed": round(float(config.turn_speed), 3),
        "duration_sec": _duration_for_turn_deg(yaw_deg, config),
        "reason": "make box_parallel_yaw_deg close to 0",
        "error_deg": round(yaw_deg, 3),
    }


def _distance_command(metrics: dict[str, Any], config: AdjustConfig) -> dict[str, Any] | None:
    distance_error_mm = float(metrics.get("control_distance_error_mm", metrics["distance_error_mm"]))
    if abs(distance_error_mm) <= float(config.distance_tolerance_mm):
        return None
    action = "forward" if distance_error_mm > 0.0 else "back"
    return {
        "phase": "adjust_near_edge_forward_distance",
        "action": action,
        "speed": round(float(config.drive_speed), 3),
        "duration_sec": _duration_for_distance_mm(distance_error_mm, config),
        "reason": "make near-edge forward distance close to target",
        "error_mm": round(distance_error_mm, 1),
        "raw_error_mm": metrics.get("distance_error_mm"),
        "planned_turn_yaw_deg": metrics.get("planned_turn_yaw_deg", 0.0),
        "forward_delta_from_planned_turn_mm": metrics.get("forward_delta_from_planned_turn_mm", 0.0),
        "predicted_after_turn_forward_mm": metrics.get("predicted_after_turn_forward_mm"),
    }


def build_plan(pose: dict[str, Any], config: AdjustConfig) -> dict[str, Any]:
    metrics = _pose_metrics(pose, config)
    if not metrics.get("ok"):
        metrics["command"] = None
        return metrics
    command = _yaw_command(metrics, config)
    if command is None:
        command = _distance_command(metrics, config)
    if command is None:
        command = {
            "phase": "done",
            "action": "none",
            "reason": "yaw and near-edge forward distance are within tolerance",
        }
    plan = dict(metrics)
    plan["command"] = command
    return plan


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
        return {"ok": True, "executed": False, "reason": "no movement needed"}
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
        "ok": completed.returncode == 0,
        "executed": True,
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


def _execution_ok(execution: dict[str, Any] | None) -> bool:
    return execution is not None and bool(execution.get("ok"))


def _single_calculation_commands(metrics: dict[str, Any], config: AdjustConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    yaw_cmd = _yaw_command(metrics, config)
    if yaw_cmd is not None:
        commands.append(yaw_cmd)
    control_metrics = _metrics_after_planned_turn(metrics, config, yaw_cmd)
    distance_cmd = _distance_command(control_metrics, config)
    if distance_cmd is not None:
        commands.append(distance_cmd)
    return commands, control_metrics


def run_adjust_sequence(config: AdjustConfig, values: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    started_at = _now_iso()
    started = time.perf_counter()
    pose = _fetch_pose(config, values)
    metrics = _pose_metrics(pose, config)
    if not metrics.get("ok"):
        return {
            "ok": False,
            "dry_run": dry_run,
            "started_at": started_at,
            "finished_at": _now_iso(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "error": metrics.get("error"),
            "target_near_edge_forward_mm": round(float(config.target_near_edge_forward_mm), 1),
            "stages": [],
            "final_plan": None,
        }

    commands, control_metrics = _single_calculation_commands(metrics, config)
    executions: list[dict[str, Any]] = []
    if dry_run:
        executions = [{"executed": False, "reason": "dry_run=1"} for _command in commands]
    else:
        for command in commands:
            execution = execute_command(config, command)
            executions.append(execution)
            if not _execution_ok(execution):
                break

    ok = all(_execution_ok(execution) for execution in executions) if commands and not dry_run else True
    if commands and dry_run:
        ok = True
    if not commands:
        ok = True

    result: dict[str, Any] = {
        "ok": ok,
        "dry_run": dry_run,
        "mode": "single_calculation_single_control_batch",
        "started_at": started_at,
        "finished_at": _now_iso(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "target_near_edge_forward_mm": round(float(config.target_near_edge_forward_mm), 1),
        "stages": [
            {
                "stage": "single_calculation_control",
                "pose_ok": bool(pose.get("ok")),
                "metrics": control_metrics,
                "commands": commands,
                "executions": executions,
            }
        ],
        "final_plan": None,
        "note": "Only one YOLO /xyz result is used; /adjust does not refetch after movement.",
    }
    if not ok:
        result["error"] = "control command failed"
    return result


def stream_adjust_sequence(
    handler: BaseHTTPRequestHandler,
    config: AdjustConfig,
    values: dict[str, Any],
    dry_run: bool = False,
) -> None:
    started_at = _now_iso()
    started = time.perf_counter()
    target_mm = round(float(config.target_near_edge_forward_mm), 1)

    def emit(event: str, payload: dict[str, Any] | None = None) -> None:
        message = {"event": event, "ts": _now_iso()}
        if payload:
            message.update(payload)
        _write_ndjson(handler, message)

    emit(
        "adjust_started",
        {
            "ok": True,
            "dry_run": dry_run,
            "mode": "single_calculation_single_control_batch",
            "target_near_edge_forward_mm": target_mm,
        },
    )

    try:
        pose = _fetch_pose(config, values)
        metrics = _pose_metrics(pose, config)
        if not metrics.get("ok"):
            result = {
                "ok": False,
                "dry_run": dry_run,
                "started_at": started_at,
                "finished_at": _now_iso(),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": metrics.get("error"),
                "target_near_edge_forward_mm": target_mm,
                "stages": [],
                "final_plan": None,
            }
            emit("adjust_finished", {"ok": False, "error": result["error"], "result": result})
            return

        commands, control_metrics = _single_calculation_commands(metrics, config)
        emit(
            "plan_ready",
            {
                "ok": True,
                "metrics": control_metrics,
                "commands": commands,
                "command_count": len(commands),
            },
        )

        executions: list[dict[str, Any]] = []
        if dry_run:
            executions = [{"executed": False, "reason": "dry_run=1"} for _command in commands]
            for index, command in enumerate(commands):
                emit(
                    "command_skipped",
                    {"ok": True, "index": index, "command": command, "execution": executions[index]},
                )
        else:
            for index, command in enumerate(commands):
                emit("command_started", {"ok": True, "index": index, "command": command})
                execution = execute_command(config, command)
                executions.append(execution)
                emit(
                    "command_finished",
                    {
                        "ok": _execution_ok(execution),
                        "index": index,
                        "command": command,
                        "execution": execution,
                    },
                )
                if not _execution_ok(execution):
                    break

        ok = all(_execution_ok(execution) for execution in executions) if commands and not dry_run else True
        if commands and dry_run:
            ok = True
        if not commands:
            ok = True

        result: dict[str, Any] = {
            "ok": ok,
            "dry_run": dry_run,
            "mode": "single_calculation_single_control_batch",
            "started_at": started_at,
            "finished_at": _now_iso(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "target_near_edge_forward_mm": target_mm,
            "stages": [
                {
                    "stage": "single_calculation_control",
                    "pose_ok": bool(pose.get("ok")),
                    "metrics": control_metrics,
                    "commands": commands,
                    "executions": executions,
                }
            ],
            "final_plan": None,
            "note": "Only one YOLO /xyz result is used; /adjust does not refetch after movement.",
        }
        if not ok:
            result["error"] = "control command failed"
        emit("adjust_finished", {"ok": ok, "error": result.get("error"), "result": result})
    except Exception as exc:
        emit(
            "adjust_finished",
            {
                "ok": False,
                "error": str(exc),
                "started_at": started_at,
                "finished_at": _now_iso(),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            },
        )


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
                            "endpoints": ["/health", "/plan", "/step", "/adjust", "/stop"],
                            "streaming": "/adjust?stream=1 returns NDJSON progress events",
                            "safety": "/plan never moves; /adjust uses one YOLO calculation and sends one control batch; use /adjust?dry_run=1 to preview",
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
                if path == "/adjust":
                    dry_run = bool(values.get("dry_run"))
                    if bool(values.get("stream")):
                        _start_ndjson_response(self)
                        stream_adjust_sequence(self, config, values, dry_run=dry_run)
                        return
                    result = run_adjust_sequence(config, values, dry_run=dry_run)
                    _json_response(self, 200 if result.get("ok") else 500, result)
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
    parser.add_argument("--max-duration-sec", type=float, default=5.0)
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
    )
    server = ThreadingHTTPServer((config.bind, config.port), make_handler(config))
    print(f"serving on http://{config.bind}:{config.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
