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
import urllib.error
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
    target_turn_to_target_yaw_deg: float = 30.0
    target_turn_tolerance_deg: float = 3.0
    planner_turn_step_deg: float = 1.0
    right_entry_prealign_forward_mm: float = 400.0
    right_entry_final_forward_mm: float = 200.0
    right_entry_target_right_mm: float = 200.0
    right_entry_lateral_tolerance_mm: float = 15.0
    right_entry_side_turn_base_deg: float = 90.0
    right_entry_side_turn_max_duration_sec: float = 20.0
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
    "target_turn_to_target_yaw_deg": float,
    "target_turn_tolerance_deg": float,
    "target_angle_deg": float,
    "planner_turn_step_deg": float,
    "right_entry_prealign_forward_mm": float,
    "right_entry_final_forward_mm": float,
    "right_entry_target_right_mm": float,
    "right_entry_lateral_tolerance_mm": float,
    "right_entry_side_turn_base_deg": float,
    "right_entry_side_turn_max_duration_sec": float,
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


def _wrap_deg_pm180(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _wrap_axis_yaw_deg(value: float) -> float:
    wrapped = _wrap_deg_pm180(value)
    if wrapped >= 90.0:
        wrapped -= 180.0
    if wrapped < -90.0:
        wrapped += 180.0
    return wrapped


def _bearing_yaw_deg(forward_mm: float, right_mm: float) -> float:
    # Positive yaw follows ROS angular.z: target on the robot-left side is positive.
    return math.degrees(math.atan2(-float(right_mm), float(forward_mm)))


def _rotate_forward_right(forward_mm: float, right_mm: float, yaw_delta_rad: float) -> tuple[float, float]:
    cos_yaw = math.cos(float(yaw_delta_rad))
    sin_yaw = math.sin(float(yaw_delta_rad))
    forward_after = float(forward_mm) * cos_yaw - float(right_mm) * sin_yaw
    right_after = float(forward_mm) * sin_yaw + float(right_mm) * cos_yaw
    return forward_after, right_after


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
        "target_turn_to_target_yaw_deg",
        "target_turn_tolerance_deg",
        "planner_turn_step_deg",
        "right_entry_prealign_forward_mm",
        "right_entry_final_forward_mm",
        "right_entry_target_right_mm",
        "right_entry_lateral_tolerance_mm",
        "right_entry_side_turn_base_deg",
        "right_entry_side_turn_max_duration_sec",
        "yaw_tolerance_deg",
        "distance_tolerance_mm",
        "turn_speed",
        "drive_speed",
        "min_duration_sec",
        "max_duration_sec",
    }
    updates = {key: values[key] for key in allowed if key in values}
    if "target_angle_deg" in values:
        updates["target_turn_to_target_yaw_deg"] = values["target_angle_deg"]
    return replace(config, **updates)


def _requested_yolo_label(values: dict[str, Any]) -> str | None:
    label = values.get("label") or values.get("yolo_label")
    if label is None:
        return None
    label = str(label).strip()
    return label or None


def _label_matches(requested_label: str | None, selected_label: Any) -> bool:
    if not requested_label:
        return True
    if selected_label is None:
        return False
    requested = str(requested_label).strip().lower()
    selected = str(selected_label).strip().lower()
    return requested == selected or requested in selected or selected in requested


def _url_with_pose_overrides(base_url: str, values: dict[str, Any]) -> str:
    parsed = urllib.parse.urlparse(base_url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    label = _requested_yolo_label(values)
    if label:
        query["label"] = label
    new_query = urllib.parse.urlencode(query)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _fetch_pose(config: AdjustConfig, values: dict[str, Any]) -> dict[str, Any]:
    url = _url_with_pose_overrides(config.yolo_xyz_url, values)
    try:
        with urllib.request.urlopen(url, timeout=float(config.request_timeout_sec)) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"ok": False, "error": body.strip() or f"YOLO /xyz HTTP {exc.code}"}
        if not isinstance(payload, dict):
            payload = {"ok": False, "error": f"YOLO /xyz HTTP {exc.code}: non-object JSON response"}
        payload.setdefault("ok", False)
        payload.setdefault("error", f"YOLO /xyz HTTP {exc.code}")
        return payload


def _nested(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _duration_for_turn_deg(
    yaw_deg: float,
    config: AdjustConfig,
    *,
    max_duration_sec: float | None = None,
) -> float:
    speed = max(abs(float(config.turn_speed)), 1e-6)
    duration = math.radians(abs(float(yaw_deg))) / speed
    max_duration = float(config.max_duration_sec if max_duration_sec is None else max_duration_sec)
    return round(_clamp(duration, config.min_duration_sec, max_duration), 3)


def _duration_for_distance_mm(distance_error_mm: float, config: AdjustConfig) -> float:
    speed = max(abs(float(config.drive_speed)), 1e-6)
    duration = abs(float(distance_error_mm)) / 1000.0 / speed
    return round(_clamp(duration, config.min_duration_sec, config.max_duration_sec), 3)


def _pose_metrics(pose: dict[str, Any], config: AdjustConfig, values: dict[str, Any] | None = None) -> dict[str, Any]:
    requested_label = _requested_yolo_label(values or {})
    selected_label = pose.get("selected_yolo_label")
    if not pose.get("ok"):
        return {
            "ok": False,
            "error": pose.get("error") or "YOLO pose is not ok",
            "pose_ok": False,
            "requested_yolo_label": requested_label,
            "selected_yolo_label": selected_label,
        }

    yaw_deg = _nested(pose, "robot_alignment", "control_hint", "box_parallel_yaw_deg")
    turn_to_target_yaw_deg = _nested(pose, "robot_alignment", "control_hint", "turn_first_yaw_deg")
    if turn_to_target_yaw_deg is None:
        turn_to_target_yaw_deg = _nested(pose, "robot_alignment", "target", "cmd_vel_yaw_to_center_deg")
    near_forward_mm = _nested(pose, "near_edge_robot_alignment", "target", "ground_forward_mm")
    near_right_mm = _nested(pose, "near_edge_robot_alignment", "target", "right_mm")
    center_forward_mm = _nested(pose, "robot_alignment", "target", "ground_forward_mm")
    center_right_mm = _nested(pose, "robot_alignment", "target", "right_mm")
    near_xyz = pose.get("near_edge_midpoint_xyz_mm")
    if yaw_deg is None:
        return {"ok": False, "error": "missing robot_alignment.control_hint.box_parallel_yaw_deg"}
    if near_forward_mm is None:
        return {"ok": False, "error": "missing near_edge_robot_alignment.target.ground_forward_mm"}
    if center_forward_mm is None:
        return {"ok": False, "error": "missing robot_alignment.target.ground_forward_mm"}

    yaw_deg = float(yaw_deg)
    near_forward_mm = float(near_forward_mm)
    if near_right_mm is None and isinstance(near_xyz, list) and near_xyz:
        near_right_mm = near_xyz[0]
    if center_right_mm is None:
        center_xyz = pose.get("center_xyz_mm")
        if isinstance(center_xyz, list) and center_xyz:
            center_right_mm = center_xyz[0]
    if turn_to_target_yaw_deg is None and center_right_mm is not None:
        turn_to_target_yaw_deg = _bearing_yaw_deg(float(center_forward_mm), float(center_right_mm))
    distance_error_mm = near_forward_mm - float(config.target_near_edge_forward_mm)
    return {
        "ok": True,
        "pose_ok": True,
        "requested_yolo_label": requested_label,
        "selected_yolo_label": pose.get("selected_yolo_label"),
        "selected_yolo_class_id": pose.get("selected_yolo_class_id"),
        "selected_yolo_confidence": pose.get("selected_yolo_confidence"),
        "yolo_label_matched": _label_matches(requested_label, selected_label),
        "selected_orientation": pose.get("selected_orientation"),
        "box_parallel_yaw_deg": round(yaw_deg, 3),
        "turn_to_target_yaw_deg": _round(turn_to_target_yaw_deg, 3),
        "near_edge_forward_mm": round(near_forward_mm, 1),
        "near_edge_right_mm": _round(near_right_mm, 1),
        "center_forward_mm": _round(center_forward_mm, 1),
        "center_right_mm": _round(center_right_mm, 1),
        "target_near_edge_forward_mm": round(float(config.target_near_edge_forward_mm), 1),
        "target_turn_to_target_yaw_deg": round(float(config.target_turn_to_target_yaw_deg), 3),
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


def _turn_command_from_delta_deg(
    yaw_delta_deg: float,
    config: AdjustConfig,
    *,
    phase: str,
    reason: str,
    max_duration_sec: float | None = None,
) -> dict[str, Any] | None:
    yaw_delta_deg = float(yaw_delta_deg)
    if abs(yaw_delta_deg) <= 0.5:
        return None
    action = "turn_left" if yaw_delta_deg > 0.0 else "turn_right"
    return {
        "phase": phase,
        "action": action,
        "speed": round(float(config.turn_speed), 3),
        "duration_sec": _duration_for_turn_deg(yaw_delta_deg, config, max_duration_sec=max_duration_sec),
        "reason": reason,
        "planned_delta_deg": round(yaw_delta_deg, 3),
        "error_deg": round(yaw_delta_deg, 3),
    }


def _drive_command_from_delta_mm(
    drive_delta_mm: float,
    config: AdjustConfig,
    *,
    phase: str,
    reason: str,
) -> dict[str, Any] | None:
    drive_delta_mm = float(drive_delta_mm)
    if abs(drive_delta_mm) <= float(config.distance_tolerance_mm):
        return None
    action = "forward" if drive_delta_mm > 0.0 else "back"
    return {
        "phase": phase,
        "action": action,
        "speed": round(float(config.drive_speed), 3),
        "duration_sec": _duration_for_distance_mm(drive_delta_mm, config),
        "reason": reason,
        "planned_delta_mm": round(drive_delta_mm, 1),
        "error_mm": round(drive_delta_mm, 1),
    }


def _max_turn_delta_deg(config: AdjustConfig) -> float:
    return math.degrees(max(abs(float(config.turn_speed)), 1e-6) * float(config.max_duration_sec))


def _max_drive_delta_mm(config: AdjustConfig) -> float:
    return max(abs(float(config.drive_speed)), 1e-6) * float(config.max_duration_sec) * 1000.0


def _candidate_turn_deltas_deg(metrics: dict[str, Any], config: AdjustConfig) -> list[float]:
    max_turn_deg = _max_turn_delta_deg(config)
    step_deg = _clamp(float(config.planner_turn_step_deg), 0.25, 5.0)
    candidates: set[float] = {0.0, -max_turn_deg, max_turn_deg}
    count = int(math.floor((max_turn_deg * 2.0) / step_deg))
    for index in range(count + 1):
        candidates.add(round(-max_turn_deg + index * step_deg, 6))

    turn_to_target = metrics.get("turn_to_target_yaw_deg")
    if turn_to_target is not None:
        target_yaw = float(config.target_turn_to_target_yaw_deg)
        exact_target_delta = float(turn_to_target) - target_yaw
        tolerance = float(config.target_turn_tolerance_deg)
        for delta in (exact_target_delta, exact_target_delta - tolerance, exact_target_delta + tolerance):
            candidates.add(_clamp(delta, -max_turn_deg, max_turn_deg))

    box_yaw = metrics.get("box_parallel_yaw_deg")
    if box_yaw is not None:
        candidates.add(_clamp(float(box_yaw), -max_turn_deg, max_turn_deg))

    return sorted(candidates)


def _target_angle_candidate(
    metrics: dict[str, Any],
    config: AdjustConfig,
    turn_delta_deg: float,
) -> dict[str, Any]:
    center_forward = float(metrics["center_forward_mm"])
    center_right = float(metrics["center_right_mm"])
    near_forward = float(metrics["near_edge_forward_mm"])
    near_right = float(metrics["near_edge_right_mm"])
    box_yaw = float(metrics["box_parallel_yaw_deg"])

    turn_delta_rad = math.radians(float(turn_delta_deg))
    center_forward_after_turn, center_right_after_turn = _rotate_forward_right(
        center_forward,
        center_right,
        turn_delta_rad,
    )
    near_forward_after_turn, near_right_after_turn = _rotate_forward_right(
        near_forward,
        near_right,
        turn_delta_rad,
    )

    raw_drive_delta_mm = near_forward_after_turn - float(config.target_near_edge_forward_mm)
    max_drive_mm = _max_drive_delta_mm(config)
    drive_delta_mm = _clamp(raw_drive_delta_mm, -max_drive_mm, max_drive_mm)
    center_forward_final = center_forward_after_turn - drive_delta_mm
    near_forward_final = near_forward_after_turn - drive_delta_mm
    target_yaw_final = _bearing_yaw_deg(center_forward_final, center_right_after_turn)
    target_yaw_error_deg = abs(_wrap_deg_pm180(target_yaw_final - float(config.target_turn_to_target_yaw_deg)))
    box_yaw_final = _wrap_axis_yaw_deg(box_yaw - float(turn_delta_deg))
    near_forward_error_mm = near_forward_final - float(config.target_near_edge_forward_mm)
    distance_error_mm = abs(near_forward_error_mm)

    in_target_band = target_yaw_error_deg <= float(config.target_turn_tolerance_deg)
    if in_target_band:
        sort_key = (
            0,
            round(abs(box_yaw_final), 6),
            round(distance_error_mm, 6),
            round(target_yaw_error_deg, 6),
            round(abs(turn_delta_deg), 6),
            round(abs(drive_delta_mm), 6),
        )
    else:
        sort_key = (
            1,
            round(target_yaw_error_deg, 6),
            round(abs(box_yaw_final), 6),
            round(distance_error_mm, 6),
            round(abs(turn_delta_deg), 6),
            round(abs(drive_delta_mm), 6),
        )

    return {
        "turn_delta_deg": round(float(turn_delta_deg), 3),
        "turn_delta_rad": round(turn_delta_rad, 6),
        "raw_drive_delta_mm": round(raw_drive_delta_mm, 1),
        "drive_delta_mm": round(drive_delta_mm, 1),
        "drive_delta_clipped": abs(raw_drive_delta_mm - drive_delta_mm) > 1e-6,
        "center_forward_after_turn_mm": round(center_forward_after_turn, 1),
        "center_right_after_turn_mm": round(center_right_after_turn, 1),
        "near_edge_forward_after_turn_mm": round(near_forward_after_turn, 1),
        "near_edge_right_after_turn_mm": round(near_right_after_turn, 1),
        "center_forward_final_mm": round(center_forward_final, 1),
        "center_right_final_mm": round(center_right_after_turn, 1),
        "near_edge_forward_final_mm": round(near_forward_final, 1),
        "near_edge_forward_error_mm": round(near_forward_error_mm, 1),
        "turn_to_target_yaw_final_deg": round(target_yaw_final, 3),
        "turn_to_target_yaw_error_deg": round(target_yaw_error_deg, 3),
        "box_parallel_yaw_final_deg": round(box_yaw_final, 3),
        "box_parallel_yaw_error_deg": round(abs(box_yaw_final), 3),
        "in_target_turn_band": in_target_band,
        "sort_key": sort_key,
    }


def _target_angle_plan_commands(
    candidate: dict[str, Any],
    config: AdjustConfig,
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    turn_command = _turn_command_from_delta_deg(
        float(candidate["turn_delta_deg"]),
        config,
        phase="set_target_turn_angle",
        reason="make turn_to_target_yaw_deg close to target angle first",
    )
    if turn_command is not None:
        commands.append(turn_command)
    drive_command = _drive_command_from_delta_mm(
        float(candidate["drive_delta_mm"]),
        config,
        phase="adjust_near_edge_forward_distance_after_target_angle",
        reason="move forward/back so near-edge forward distance is close to target",
    )
    if drive_command is not None:
        drive_command["raw_drive_delta_mm"] = candidate.get("raw_drive_delta_mm")
        drive_command["drive_delta_clipped"] = candidate.get("drive_delta_clipped")
        commands.append(drive_command)
    return commands


def build_target_angle_plan(
    pose: dict[str, Any],
    config: AdjustConfig,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _pose_metrics(pose, config, values)
    if not metrics.get("ok"):
        metrics["commands"] = []
        return metrics
    required = (
        "center_forward_mm",
        "center_right_mm",
        "near_edge_forward_mm",
        "near_edge_right_mm",
        "box_parallel_yaw_deg",
        "turn_to_target_yaw_deg",
    )
    missing = [key for key in required if metrics.get(key) is None]
    if missing:
        return {
            **metrics,
            "ok": False,
            "error": "missing metrics for target-angle planner: " + ", ".join(missing),
            "commands": [],
        }

    candidates = [_target_angle_candidate(metrics, config, delta) for delta in _candidate_turn_deltas_deg(metrics, config)]
    selected = min(candidates, key=lambda item: item["sort_key"])
    commands = _target_angle_plan_commands(selected, config)
    if not commands:
        commands = [
            {
                "phase": "done",
                "action": "none",
                "reason": "target angle, box yaw, and near-edge forward distance are within rule tolerances",
            }
        ]
    plan = dict(metrics)
    plan.update(
        {
            "strategy": "target_angle_rule_planner",
            "target_turn_to_target_yaw_deg": round(float(config.target_turn_to_target_yaw_deg), 3),
            "target_turn_tolerance_deg": round(float(config.target_turn_tolerance_deg), 3),
            "planner_turn_step_deg": round(float(config.planner_turn_step_deg), 3),
            "planner_max_turn_delta_deg": round(_max_turn_delta_deg(config), 3),
            "planner_max_drive_delta_mm": round(_max_drive_delta_mm(config), 1),
            "planner_rule": (
                "enumerate bounded turn deltas; after each candidate, predict one forward/back move "
                "to near-edge target distance; prefer target yaw band first, then smaller box yaw, then distance"
            ),
            "selected_candidate": selected,
            "candidate_count": len(candidates),
            "commands": commands,
        }
    )
    return plan


def _single_adjust_plan_from_pose(
    pose: dict[str, Any],
    config: AdjustConfig,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _pose_metrics(pose, config, values)
    if not metrics.get("ok"):
        return {
            **metrics,
            "commands": [],
        }
    commands, control_metrics = _single_calculation_commands(metrics, config)
    return {
        **control_metrics,
        "strategy": "single_calculation_single_control_batch",
        "commands": commands,
    }


def build_right_entry_side_shift_plan(
    pose: dict[str, Any],
    config: AdjustConfig,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _pose_metrics(pose, config, values)
    if not metrics.get("ok"):
        return {
            **metrics,
            "commands": [],
        }

    required = ("center_forward_mm", "center_right_mm", "box_parallel_yaw_deg")
    missing = [key for key in required if metrics.get(key) is None]
    if missing:
        return {
            **metrics,
            "ok": False,
            "error": "missing metrics for right-entry side shift: " + ", ".join(missing),
            "commands": [],
        }

    center_forward_mm = float(metrics["center_forward_mm"])
    center_right_mm = float(metrics["center_right_mm"])
    box_yaw_deg = float(metrics["box_parallel_yaw_deg"])
    target_right_mm = float(config.right_entry_target_right_mm)
    lateral_error_mm = target_right_mm - center_right_mm
    turn_delta_deg = float(config.right_entry_side_turn_base_deg) + box_yaw_deg
    turn_delta_rad = math.radians(turn_delta_deg)
    sin_turn = math.sin(turn_delta_rad)
    cos_turn = math.cos(turn_delta_rad)

    if abs(sin_turn) < 0.2:
        return {
            **metrics,
            "ok": False,
            "strategy": "right_hand_side_shift",
            "error": "side-turn angle is too close to forward/back direction for safe lateral adjustment",
            "turn_delta_deg": round(turn_delta_deg, 3),
            "commands": [],
        }

    commands: list[dict[str, Any]] = []
    predicted_drive_delta_mm = 0.0
    if abs(lateral_error_mm) > float(config.right_entry_lateral_tolerance_mm):
        predicted_drive_delta_mm = lateral_error_mm / sin_turn
        side_turn = _turn_command_from_delta_deg(
            turn_delta_deg,
            config,
            phase="right_entry_turn_sideways",
            reason="turn about 90 degrees so forward/back motion becomes a controlled lateral shift",
            max_duration_sec=float(config.right_entry_side_turn_max_duration_sec),
        )
        lateral_drive = _drive_command_from_delta_mm(
            predicted_drive_delta_mm,
            config,
            phase="right_entry_lateral_drive",
            reason="move forward/back after sideways turn to place the target at the right-hand lateral offset",
        )
        turn_back = _turn_command_from_delta_deg(
            -turn_delta_deg,
            config,
            phase="right_entry_turn_back",
            reason="return to the original operating heading after lateral shift",
            max_duration_sec=float(config.right_entry_side_turn_max_duration_sec),
        )
        commands = [command for command in (side_turn, lateral_drive, turn_back) if command is not None]

    predicted_center_right_mm = center_right_mm + predicted_drive_delta_mm * sin_turn
    predicted_center_forward_mm = center_forward_mm - predicted_drive_delta_mm * cos_turn
    if not commands:
        commands = [
            {
                "phase": "right_entry_side_shift_done",
                "action": "none",
                "reason": "center lateral offset is already within right-entry tolerance",
            }
        ]

    return {
        **metrics,
        "strategy": "right_hand_side_shift",
        "right_entry_rule": (
            "after safe pre-align, refetch YOLO; turn left by 90deg plus residual box yaw, "
            "drive forward/back to correct center_right_mm to the right-hand target, then turn back"
        ),
        "right_entry_target_right_mm": round(target_right_mm, 1),
        "right_entry_lateral_tolerance_mm": round(float(config.right_entry_lateral_tolerance_mm), 1),
        "right_entry_side_turn_base_deg": round(float(config.right_entry_side_turn_base_deg), 3),
        "side_turn_delta_deg": round(turn_delta_deg, 3),
        "side_turn_delta_rad": round(turn_delta_rad, 6),
        "side_turn_sin": round(sin_turn, 6),
        "side_turn_cos": round(cos_turn, 6),
        "lateral_error_mm": round(lateral_error_mm, 1),
        "predicted_drive_delta_mm": round(predicted_drive_delta_mm, 1),
        "predicted_center_right_after_side_shift_mm": round(predicted_center_right_mm, 1),
        "predicted_center_forward_after_side_shift_mm": round(predicted_center_forward_mm, 1),
        "commands": commands,
    }


def build_plan(pose: dict[str, Any], config: AdjustConfig, values: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = _pose_metrics(pose, config, values)
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


def _execute_command_batch(
    config: AdjustConfig,
    commands: list[dict[str, Any]],
    dry_run: bool,
) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    if dry_run:
        return [{"executed": False, "reason": "dry_run=1"} for _command in commands]

    for command in commands:
        execution = execute_command(config, command)
        executions.append(execution)
        if not _execution_ok(execution):
            break
    return executions


def _commands_ok(commands: list[dict[str, Any]], executions: list[dict[str, Any]], dry_run: bool) -> bool:
    if not commands:
        return True
    if dry_run:
        return True
    return all(_execution_ok(execution) for execution in executions)


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
    requested_label = _requested_yolo_label(values)
    pose = _fetch_pose(config, values)
    metrics = _pose_metrics(pose, config, values)
    if not metrics.get("ok"):
        return {
            "ok": False,
            "dry_run": dry_run,
            "requested_yolo_label": requested_label,
            "selected_yolo_label": metrics.get("selected_yolo_label"),
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
        "requested_yolo_label": requested_label,
        "selected_yolo_label": control_metrics.get("selected_yolo_label"),
        "yolo_label_matched": control_metrics.get("yolo_label_matched"),
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


def run_target_angle_adjust_sequence(config: AdjustConfig, values: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    started_at = _now_iso()
    started = time.perf_counter()
    requested_label = _requested_yolo_label(values)
    pose = _fetch_pose(config, values)
    plan = build_target_angle_plan(pose, config, values)
    if not plan.get("ok"):
        return {
            "ok": False,
            "dry_run": dry_run,
            "mode": "target_angle_rule_planner",
            "requested_yolo_label": requested_label,
            "selected_yolo_label": plan.get("selected_yolo_label"),
            "started_at": started_at,
            "finished_at": _now_iso(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "error": plan.get("error"),
            "target_near_edge_forward_mm": round(float(config.target_near_edge_forward_mm), 1),
            "target_turn_to_target_yaw_deg": round(float(config.target_turn_to_target_yaw_deg), 3),
            "plan": plan,
            "stages": [],
        }

    commands = [command for command in plan.get("commands", []) if isinstance(command, dict)]
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
        "mode": "target_angle_rule_planner",
        "requested_yolo_label": requested_label,
        "selected_yolo_label": plan.get("selected_yolo_label"),
        "yolo_label_matched": plan.get("yolo_label_matched"),
        "started_at": started_at,
        "finished_at": _now_iso(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "target_near_edge_forward_mm": round(float(config.target_near_edge_forward_mm), 1),
        "target_turn_to_target_yaw_deg": round(float(config.target_turn_to_target_yaw_deg), 3),
        "stages": [
            {
                "stage": "target_angle_rule_control",
                "pose_ok": bool(pose.get("ok")),
                "metrics": plan,
                "commands": commands,
                "executions": executions,
            }
        ],
        "final_plan": None,
        "note": (
            "Rule-based target-angle planner. It uses one YOLO /xyz result, predicts one bounded turn "
            "and one forward/back move, and does not refetch after movement."
        ),
    }
    if not ok:
        result["error"] = "control command failed"
    return result


def run_right_entry_adjust_sequence(config: AdjustConfig, values: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    started_at = _now_iso()
    started = time.perf_counter()
    requested_label = _requested_yolo_label(values)
    stages: list[dict[str, Any]] = []

    def finish(ok: bool, error: str | None = None, note: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": ok,
            "dry_run": dry_run,
            "mode": "right_hand_safe_entry",
            "requested_yolo_label": requested_label,
            "started_at": started_at,
            "finished_at": _now_iso(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "right_entry_targets": {
                "prealign_near_edge_forward_mm": round(float(config.right_entry_prealign_forward_mm), 1),
                "final_near_edge_forward_mm": round(float(config.right_entry_final_forward_mm), 1),
                "center_right_mm": round(float(config.right_entry_target_right_mm), 1),
            },
            "stages": stages,
        }
        if error:
            result["error"] = error
        if note:
            result["note"] = note
        return result

    def add_single_adjust_stage(stage_name: str, stage_config: AdjustConfig) -> bool:
        pose = _fetch_pose(stage_config, values)
        plan = _single_adjust_plan_from_pose(pose, stage_config, values)
        commands = [command for command in plan.get("commands", []) if isinstance(command, dict)]
        executions = _execute_command_batch(stage_config, commands, dry_run)
        stage = {
            "stage": stage_name,
            "pose_ok": bool(pose.get("ok")),
            "metrics": plan,
            "commands": commands,
            "executions": executions,
        }
        stages.append(stage)
        return bool(plan.get("ok")) and _commands_ok(commands, executions, dry_run)

    prealign_config = replace(config, target_near_edge_forward_mm=float(config.right_entry_prealign_forward_mm))
    if not add_single_adjust_stage("safe_prealign_to_target", prealign_config):
        return finish(False, "safe pre-align failed")

    if dry_run:
        stages.append(
            {
                "stage": "side_shift_and_final_adjust_skipped",
                "pose_ok": None,
                "metrics": {
                    "ok": True,
                    "reason": (
                        "dry_run=1 only previews the first safe pre-align stage; the side shift and final "
                        "200mm adjust require fresh YOLO after real movement"
                    ),
                },
                "commands": [],
                "executions": [],
            }
        )
        return finish(
            True,
            note=(
                "Dry run shows stage 1 only. Stage 2 and stage 3 deliberately refetch YOLO after real movement, "
                "so they are not simulated from stale data."
            ),
        )

    side_pose = _fetch_pose(config, values)
    side_plan = build_right_entry_side_shift_plan(side_pose, config, values)
    side_commands = [command for command in side_plan.get("commands", []) if isinstance(command, dict)]
    side_executions = _execute_command_batch(config, side_commands, dry_run=False)
    stages.append(
        {
            "stage": "pseudo_lateral_shift_to_right_hand",
            "pose_ok": bool(side_pose.get("ok")),
            "metrics": side_plan,
            "commands": side_commands,
            "executions": side_executions,
        }
    )
    if not side_plan.get("ok") or not _commands_ok(side_commands, side_executions, dry_run=False):
        return finish(False, "pseudo lateral shift failed")

    final_config = replace(config, target_near_edge_forward_mm=float(config.right_entry_final_forward_mm))
    if not add_single_adjust_stage("final_adjust_to_200mm", final_config):
        return finish(False, "final operation adjust failed")

    return finish(
        True,
        note=(
            "Right-hand safe entry uses fresh YOLO before each major stage: pre-align to 400mm, "
            "pseudo-lateral side shift, then final 200mm operation adjust."
        ),
    )


def stream_adjust_sequence(
    handler: BaseHTTPRequestHandler,
    config: AdjustConfig,
    values: dict[str, Any],
    dry_run: bool = False,
) -> None:
    started_at = _now_iso()
    started = time.perf_counter()
    target_mm = round(float(config.target_near_edge_forward_mm), 1)
    requested_label = _requested_yolo_label(values)

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
            "requested_yolo_label": requested_label,
            "target_near_edge_forward_mm": target_mm,
        },
    )

    try:
        pose = _fetch_pose(config, values)
        metrics = _pose_metrics(pose, config, values)
        if not metrics.get("ok"):
            result = {
                "ok": False,
                "dry_run": dry_run,
                "requested_yolo_label": requested_label,
                "selected_yolo_label": metrics.get("selected_yolo_label"),
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
            "requested_yolo_label": requested_label,
            "selected_yolo_label": control_metrics.get("selected_yolo_label"),
            "yolo_label_matched": control_metrics.get("yolo_label_matched"),
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
                            "endpoints": [
                                "/health",
                                "/plan",
                                "/step",
                                "/adjust",
                                "/plan_target_angle",
                                "/adjust_target_angle",
                                "/plan_right_entry",
                                "/adjust_right_entry",
                                "/stop",
                            ],
                            "label_usage": "pass ?label=XiongMao or ?label=Xizi_Liqun to make YOLO select that class before adjustment",
                            "streaming": "/adjust?stream=1 returns NDJSON progress events",
                            "target_angle_rule": "use /adjust_target_angle to prefer turn_to_target_yaw_deg=30deg while keeping box yaw and near-edge distance controlled",
                            "right_entry_rule": "use /adjust_right_entry for right-hand safe entry: pre-align to 400mm, refetch YOLO, side-shift, refetch YOLO, final adjust to 200mm",
                            "safety": "/plan, /plan_target_angle, /plan_right_entry, and dry_run=1 never move the robot",
                        },
                    )
                    return
                if path == "/plan":
                    pose = _fetch_pose(config, values)
                    plan = build_plan(pose, config, values)
                    _json_response(
                        self,
                        200 if plan.get("ok") else 500,
                        {
                            "ok": bool(plan.get("ok")),
                            "requested_yolo_label": _requested_yolo_label(values),
                            "selected_yolo_label": plan.get("selected_yolo_label"),
                            "yolo_label_matched": plan.get("yolo_label_matched"),
                            "plan": plan,
                        },
                    )
                    return
                if path == "/step":
                    pose = _fetch_pose(config, values)
                    plan = build_plan(pose, config, values)
                    confirm = bool(values.get("confirm") or values.get("execute"))
                    execution = {"executed": False, "reason": "missing confirm=1"}
                    if confirm and plan.get("ok") and isinstance(plan.get("command"), dict):
                        execution = execute_command(config, plan["command"])
                    _json_response(
                        self,
                        200 if plan.get("ok") else 500,
                        {
                            "ok": bool(plan.get("ok")),
                            "confirmed": confirm,
                            "requested_yolo_label": _requested_yolo_label(values),
                            "selected_yolo_label": plan.get("selected_yolo_label"),
                            "yolo_label_matched": plan.get("yolo_label_matched"),
                            "plan": plan,
                            "execution": execution,
                        },
                    )
                    return
                if path == "/plan_target_angle":
                    pose = _fetch_pose(config, values)
                    plan = build_target_angle_plan(pose, config, values)
                    _json_response(
                        self,
                        200 if plan.get("ok") else 500,
                        {
                            "ok": bool(plan.get("ok")),
                            "requested_yolo_label": _requested_yolo_label(values),
                            "selected_yolo_label": plan.get("selected_yolo_label"),
                            "yolo_label_matched": plan.get("yolo_label_matched"),
                            "plan": plan,
                        },
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
                if path == "/adjust_target_angle":
                    dry_run = bool(values.get("dry_run"))
                    result = run_target_angle_adjust_sequence(config, values, dry_run=dry_run)
                    _json_response(self, 200 if result.get("ok") else 500, result)
                    return
                if path == "/plan_right_entry":
                    result = run_right_entry_adjust_sequence(config, values, dry_run=True)
                    _json_response(self, 200 if result.get("ok") else 500, result)
                    return
                if path == "/adjust_right_entry":
                    dry_run = bool(values.get("dry_run"))
                    result = run_right_entry_adjust_sequence(config, values, dry_run=dry_run)
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
    parser.add_argument("--target-turn-to-target-yaw-deg", type=float, default=30.0)
    parser.add_argument("--target-turn-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--planner-turn-step-deg", type=float, default=1.0)
    parser.add_argument("--right-entry-prealign-forward-mm", type=float, default=400.0)
    parser.add_argument("--right-entry-final-forward-mm", type=float, default=200.0)
    parser.add_argument("--right-entry-target-right-mm", type=float, default=200.0)
    parser.add_argument("--right-entry-lateral-tolerance-mm", type=float, default=15.0)
    parser.add_argument("--right-entry-side-turn-base-deg", type=float, default=90.0)
    parser.add_argument("--right-entry-side-turn-max-duration-sec", type=float, default=20.0)
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
        target_turn_to_target_yaw_deg=args.target_turn_to_target_yaw_deg,
        target_turn_tolerance_deg=args.target_turn_tolerance_deg,
        planner_turn_step_deg=args.planner_turn_step_deg,
        right_entry_prealign_forward_mm=args.right_entry_prealign_forward_mm,
        right_entry_final_forward_mm=args.right_entry_final_forward_mm,
        right_entry_target_right_mm=args.right_entry_target_right_mm,
        right_entry_lateral_tolerance_mm=args.right_entry_lateral_tolerance_mm,
        right_entry_side_turn_base_deg=args.right_entry_side_turn_base_deg,
        right_entry_side_turn_max_duration_sec=args.right_entry_side_turn_max_duration_sec,
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
