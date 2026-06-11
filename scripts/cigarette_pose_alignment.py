#!/usr/bin/env python3
"""Compute robot-facing alignment hints from the YOLO/PnP cigarette pose.

Input pose coordinates are in left_camera_optical:
  +X image right, +Y image down, +Z camera forward.

The output is perception-only. It does not publish robot commands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from typing import Any, List


DEFAULT_POSE_URL = "http://127.0.0.1:18081/pose"
DEFAULT_CAMERA_TO_VERTICAL_DEG = 42.4


def _round(value: float, ndigits: int = 3) -> float:
    return round(float(value), ndigits)


def _round_list(values: Any, ndigits: int = 6) -> list[float]:
    return [round(float(value), ndigits) for value in values]


def _normalize_angle_deg(angle_deg: float) -> float:
    value = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if value == -180.0:
        return 180.0
    return value


def _normalize_axis_angle_deg(angle_deg: float) -> float:
    """Return undirected axis angle in [-90, 90)."""
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


Vector3 = List[float]
Vector2 = List[float]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(float(left[i]) * float(right[i]) for i in range(3))


def _norm(vector: Vector3 | Vector2) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return [float(value) * float(scalar) for value in vector]


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return [float(left[i]) - float(right[i]) for i in range(3)]


def _normalize(vector: Vector3) -> Vector3 | None:
    length = _norm(vector)
    if length <= 1e-9:
        return None
    return [float(value) / length for value in vector]


def configured_ground_basis(camera_to_vertical_deg: float) -> dict[str, Vector3]:
    theta = math.radians(float(camera_to_vertical_deg))
    x_right = [1.0, 0.0, 0.0]
    vertical_down = [0.0, math.sin(theta), math.cos(theta)]
    vertical_up = [-value for value in vertical_down]
    ground_forward = [0.0, -math.cos(theta), math.sin(theta)]
    return {
        "x_right_unit": x_right,
        "vertical_down_unit": vertical_down,
        "vertical_up_unit": vertical_up,
        "ground_forward_unit": ground_forward,
    }


def _as_vector(values: Any, name: str) -> Vector3:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name} must be a 3-value list")
    return [float(value) for value in values]


def _project_to_ground(vector: Vector3, vertical_down_unit: Vector3) -> Vector3 | None:
    projected = _sub(vector, _scale(vertical_down_unit, _dot(vector, vertical_down_unit)))
    return _normalize(projected)


def _dot_mm(point: Vector3, unit: Vector3) -> float:
    return _dot(point, unit)


def _image_axis_angle(points_px: Any, long_axis: str | None) -> dict[str, Any] | None:
    if not isinstance(points_px, list) or len(points_px) != 4:
        return None
    points = [[float(x), float(y)] for x, y in points_px]
    edge_x = [
        ((points[1][0] - points[0][0]) + (points[2][0] - points[3][0])) * 0.5,
        ((points[1][1] - points[0][1]) + (points[2][1] - points[3][1])) * 0.5,
    ]
    edge_y = [
        ((points[2][0] - points[1][0]) + (points[3][0] - points[0][0])) * 0.5,
        ((points[2][1] - points[1][1]) + (points[3][1] - points[0][1])) * 0.5,
    ]
    if long_axis == "object_x":
        vector = edge_x
        source = "object_x_edge"
    elif long_axis == "object_y":
        vector = edge_y
        source = "object_y_edge"
    else:
        edge_x_len = _norm(edge_x)
        edge_y_len = _norm(edge_y)
        vector = edge_x if edge_x_len >= edge_y_len else edge_y
        source = "longer_pixel_edge"
    angle_deg = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    return {
        "source": source,
        "image_axis_angle_deg": _round(_normalize_angle_deg(angle_deg), 3),
        "image_axis_angle_mod180_deg": _round(_normalize_axis_angle_deg(angle_deg), 3),
        "image_axis_vector_px": [_round(float(vector[0]), 3), _round(float(vector[1]), 3)],
        "note": "image angle uses +X right and +Y down; mod180 ignores head/tail direction",
    }


def compute_robot_alignment(
    pose: dict[str, Any],
    camera_to_vertical_deg: float = DEFAULT_CAMERA_TO_VERTICAL_DEG,
    target_key: str = "center_xyz_mm",
) -> dict[str, Any]:
    basis = configured_ground_basis(camera_to_vertical_deg)
    x_right = basis["x_right_unit"]
    vertical_down = basis["vertical_down_unit"]
    vertical_up = basis["vertical_up_unit"]
    ground_forward = basis["ground_forward_unit"]

    target = _as_vector(pose.get(target_key), target_key)
    target_right_mm = _dot_mm(target, x_right)
    target_forward_mm = _dot_mm(target, ground_forward)
    target_vertical_down_mm = _dot_mm(target, vertical_down)
    target_ground_distance_mm = math.hypot(target_right_mm, target_forward_mm)
    target_bearing_right_deg = math.degrees(math.atan2(target_right_mm, target_forward_mm))
    target_bearing_right_norm_deg = _normalize_angle_deg(target_bearing_right_deg)

    box_head_point = pose.get("box_head_point")
    if not isinstance(box_head_point, dict):
        box_head_point = {}
    head_to_tail = box_head_point.get("head_to_tail_unit_xyz")
    long_axis = box_head_point.get("long_axis") if isinstance(box_head_point.get("long_axis"), str) else None

    box_axis: dict[str, Any] | None = None
    if isinstance(head_to_tail, list) and len(head_to_tail) == 3:
        axis = _as_vector(head_to_tail, "box_head_point.head_to_tail_unit_xyz")
        axis_ground = _project_to_ground(axis, vertical_down)
        if axis_ground is not None:
            axis_right = _dot(axis_ground, x_right)
            axis_forward = _dot(axis_ground, ground_forward)
            axis_yaw_deg = math.degrees(math.atan2(axis_right, axis_forward))
            axis_yaw_norm_deg = _normalize_angle_deg(axis_yaw_deg)
            axis_yaw_mod180 = _normalize_axis_angle_deg(axis_yaw_deg)
            box_axis = {
                "source": "box_head_point.head_to_tail_unit_xyz",
                "long_axis": long_axis,
                "head_to_tail_unit_xyz": _round_list(axis, 6),
                "ground_projected_unit_xyz": _round_list(axis_ground, 6),
                "ground_right_component": _round(axis_right, 6),
                "ground_forward_component": _round(axis_forward, 6),
                "axis_yaw_head_to_tail_deg": _round(axis_yaw_norm_deg, 3),
                "axis_yaw_head_to_tail_rad": _round(math.radians(axis_yaw_norm_deg), 6),
                "axis_yaw_mod180_deg": _round(axis_yaw_mod180, 3),
                "axis_yaw_mod180_rad": _round(math.radians(axis_yaw_mod180), 6),
                "cmd_vel_yaw_to_parallel_axis_deg": _round(-axis_yaw_mod180, 3),
                "cmd_vel_yaw_to_parallel_axis_rad": _round(math.radians(-axis_yaw_mod180), 6),
                "note": "axis_yaw_mod180 ignores head/tail direction; cmd_vel yaw sign assumes positive angular.z turns left",
            }

    image_axis = _image_axis_angle(pose.get("points_px"), long_axis)

    range_mm = pose.get("range_from_left_camera_mm")
    try:
        range_mm_value = float(range_mm)
    except Exception:
        range_mm_value = _norm(target)

    return {
        "ok": True,
        "frame": "left_camera_optical",
        "target_key": target_key,
        "camera_to_vertical_deg": _round(camera_to_vertical_deg, 3),
        "basis": {
            "x_right_unit_xyz": _round_list(x_right, 6),
            "ground_forward_unit_xyz": _round_list(ground_forward, 6),
            "vertical_down_unit_xyz": _round_list(vertical_down, 6),
            "vertical_up_unit_xyz": _round_list(vertical_up, 6),
            "ground_forward_source": "projection_of_camera_plus_z_using_configured_camera_angle",
        },
        "target": {
            "xyz_mm": _round_list(target, 1),
            "right_mm": _round(target_right_mm, 1),
            "ground_forward_mm": _round(target_forward_mm, 1),
            "ground_forward_formula": "z_mm*sin(camera_to_vertical_deg) - y_mm*cos(camera_to_vertical_deg)",
            "ground_forward_formula_inputs": {
                "y_mm": _round(target[1], 1),
                "z_mm": _round(target[2], 1),
                "camera_to_vertical_deg": _round(camera_to_vertical_deg, 3),
            },
            "vertical_down_mm": _round(target_vertical_down_mm, 1),
            "ground_distance_mm": _round(target_ground_distance_mm, 1),
            "range_from_left_camera_mm": _round(range_mm_value, 1),
            "bearing_right_deg": _round(target_bearing_right_norm_deg, 3),
            "bearing_right_rad": _round(math.radians(target_bearing_right_norm_deg), 6),
            "cmd_vel_yaw_to_center_deg": _round(-target_bearing_right_deg, 3),
            "cmd_vel_yaw_to_center_rad": _round(math.radians(-target_bearing_right_deg), 6),
            "note": "bearing_right_deg > 0 means target is to camera/robot right; cmd_vel positive angular.z is assumed left turn",
        },
        "box_axis": box_axis,
        "image_axis": image_axis,
        "control_hint": {
            "turn_first_yaw_deg": _round(-target_bearing_right_deg, 3),
            "turn_first_yaw_rad": _round(math.radians(-target_bearing_right_deg), 6),
            "forward_distance_m": _round(target_forward_mm / 1000.0, 4),
            "lateral_error_m": _round(target_right_mm / 1000.0, 4),
            "height_down_m": _round(target_vertical_down_mm / 1000.0, 4),
            "box_parallel_yaw_deg": box_axis["cmd_vel_yaw_to_parallel_axis_deg"] if box_axis else None,
            "box_parallel_yaw_rad": box_axis["cmd_vel_yaw_to_parallel_axis_rad"] if box_axis else None,
            "ros1_sdk_topic": "/cmd_vel",
            "ros1_sdk_mapping": "Twist.linear.x -> forward speed; Twist.angular.z -> yaw speed; current SDK clamps yaw to +/-0.6 rad/s",
            "safe_to_execute": False,
        },
    }


def _fetch_json(url: str, timeout_sec: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=float(timeout_sec)) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute robot alignment hints from YOLO/PnP cigarette pose JSON.")
    parser.add_argument("--url", default=DEFAULT_POSE_URL)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--camera-to-vertical-deg", type=float, default=DEFAULT_CAMERA_TO_VERTICAL_DEG)
    parser.add_argument("--target-key", default="center_xyz_mm")
    parser.add_argument("--json", help="pose JSON string; if omitted, fetch --url")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.json:
        pose = json.loads(args.json)
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        pose = json.loads(text) if text else _fetch_json(args.url, args.timeout_sec)
    else:
        pose = _fetch_json(args.url, args.timeout_sec)
    alignment = compute_robot_alignment(
        pose,
        camera_to_vertical_deg=args.camera_to_vertical_deg,
        target_key=args.target_key,
    )
    print(json.dumps(alignment, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
