#!/usr/bin/env python3
"""Convert a detected box center to a point above it in the camera frame.

Input/output coordinates use the left camera optical frame:
    +X: image right
    +Y: image down
    +Z: camera forward/depth

The default ground relation is the robot's current camera installation:
the +Z optical axis is 47.6 degrees away from the vertical-down ground
direction in the camera Y-Z plane. Therefore vertical-up in the camera frame is:
    [0, -sin(47.6 deg), -cos(47.6 deg)]

The ground-parallel direction perpendicular to the camera X axis is the
projection of camera +Z onto the ground plane:
    [0, -cos(47.6 deg), sin(47.6 deg)]

This script is intentionally perception-only. It never sends robot commands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from typing import Any


DEFAULT_POSE_URL = "http://127.0.0.1:18081/xyz"
DEFAULT_CAMERA_TO_VERTICAL_DEG = 47.6
DEFAULT_GROUND_OFFSET_MM = 0.0


def _round_list(values: list[float], ndigits: int = 1) -> list[float]:
    return [round(float(value), ndigits) for value in values]


def _parse_xyz(text: str) -> list[float]:
    cleaned = text.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if "center_xyz_mm" in data:
                data = data["center_xyz_mm"]
            else:
                data = [data["x_mm"], data["y_mm"], data["z_mm"]]
        values = [float(value) for value in data]
    except Exception:
        values = [float(part.strip()) for part in cleaned.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("--xyz must contain exactly three values: x,y,z")
    return values


def _fetch_pose(url: str, timeout_sec: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=float(timeout_sec)) as response:
        raw = response.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    if not data.get("ok", False):
        raise RuntimeError(f"pose API returned not ok: {data.get('error') or data}")
    return data


def _extract_center_xyz_mm(pose: dict[str, Any]) -> list[float]:
    if "center_xyz_mm" in pose:
        values = pose["center_xyz_mm"]
    else:
        values = [pose["x_mm"], pose["y_mm"], pose["z_mm"]]
    if len(values) != 3:
        raise ValueError(f"pose center_xyz_mm must have three values, got: {values}")
    return [float(value) for value in values]


def vertical_vectors(camera_to_vertical_deg: float) -> tuple[list[float], list[float]]:
    """Return vertical-down and vertical-up unit vectors in optical frame."""
    theta = math.radians(float(camera_to_vertical_deg))
    down = [0.0, math.sin(theta), math.cos(theta)]
    up = [-value for value in down]
    return down, up


def ground_parallel_x_perp_vectors(camera_to_vertical_deg: float) -> tuple[list[float], list[float]]:
    """Return ground-parallel unit vectors perpendicular to the camera X axis.

    Positive ground-forward is the projection of camera +Z onto the ground
    plane. Negative distance moves in the opposite direction.
    """
    theta = math.radians(float(camera_to_vertical_deg))
    forward = [0.0, -math.cos(theta), math.sin(theta)]
    backward = [-value for value in forward]
    return forward, backward


def _add_vectors(*vectors: list[float]) -> list[float]:
    return [sum(float(vector[i]) for vector in vectors) for i in range(3)]


def above_center_yz_vector(
    center_xyz_mm: list[float],
    height_mm: float = 100.0,
    camera_to_vertical_deg: float = DEFAULT_CAMERA_TO_VERTICAL_DEG,
    ground_offset_mm: float = DEFAULT_GROUND_OFFSET_MM,
    x_offset_mm: float = 0.0,
) -> dict[str, Any]:
    """Move above center, then add a ground-parallel offset."""
    down_unit, up_unit = vertical_vectors(camera_to_vertical_deg)
    ground_forward_unit, ground_backward_unit = ground_parallel_x_perp_vectors(camera_to_vertical_deg)
    vertical_offset = [float(height_mm) * value for value in up_unit]
    ground_offset = [float(ground_offset_mm) * value for value in ground_forward_unit]
    x_offset = [float(x_offset_mm), 0.0, 0.0]
    offset = _add_vectors(vertical_offset, ground_offset, x_offset)
    vertical_only = _add_vectors(center_xyz_mm, vertical_offset)
    above = [float(center_xyz_mm[i]) + offset[i] for i in range(3)]
    center_vertical_down_mm = sum(float(center_xyz_mm[i]) * down_unit[i] for i in range(3))
    above_vertical_down_mm = sum(float(above[i]) * down_unit[i] for i in range(3))
    center_ground_forward_mm = sum(float(center_xyz_mm[i]) * ground_forward_unit[i] for i in range(3))
    above_ground_forward_mm = sum(float(above[i]) * ground_forward_unit[i] for i in range(3))
    return {
        "method": "ground_vertical_yz_vector",
        "camera_to_vertical_deg": round(float(camera_to_vertical_deg), 3),
        "height_above_center_mm": round(float(height_mm), 1),
        "ground_offset_mm": round(float(ground_offset_mm), 1),
        "ground_offset_direction": "projection_of_camera_plus_z_onto_ground_plane; use negative ground_offset_mm for the opposite direction",
        "x_offset_mm": round(float(x_offset_mm), 1),
        "vertical_down_unit_xyz": _round_list(down_unit, 6),
        "vertical_up_unit_xyz": _round_list(up_unit, 6),
        "ground_forward_unit_xyz": _round_list(ground_forward_unit, 6),
        "ground_backward_unit_xyz": _round_list(ground_backward_unit, 6),
        "vertical_offset_xyz_mm": _round_list(vertical_offset),
        "ground_offset_xyz_mm": _round_list(ground_offset),
        "x_offset_xyz_mm": _round_list(x_offset),
        "offset_xyz_mm": _round_list(offset),
        "center_xyz_mm": _round_list(center_xyz_mm),
        "above_vertical_only_xyz_mm": _round_list(vertical_only),
        "above_xyz_mm": _round_list(above),
        "vertical_down_projection_mm": {
            "center": round(center_vertical_down_mm, 1),
            "above": round(above_vertical_down_mm, 1),
            "delta": round(above_vertical_down_mm - center_vertical_down_mm, 1),
        },
        "ground_forward_projection_mm": {
            "center": round(center_ground_forward_mm, 1),
            "above": round(above_ground_forward_mm, 1),
            "delta": round(above_ground_forward_mm - center_ground_forward_mm, 1),
        },
    }


def above_center_z_only(
    center_xyz_mm: list[float],
    height_mm: float = 100.0,
    camera_to_vertical_deg: float = DEFAULT_CAMERA_TO_VERTICAL_DEG,
    ground_offset_mm: float = DEFAULT_GROUND_OFFSET_MM,
    x_offset_mm: float = 0.0,
) -> dict[str, Any]:
    """Legacy helper: change Z for height, then add optional ground/X offsets."""
    theta = math.radians(float(camera_to_vertical_deg))
    cos_theta = math.cos(theta)
    if abs(cos_theta) <= 1e-9:
        raise ValueError("camera_to_vertical_deg makes cos(theta) too small")
    ground_forward_unit, _ground_backward_unit = ground_parallel_x_perp_vectors(camera_to_vertical_deg)
    z_height_offset = [0.0, 0.0, -float(height_mm) / cos_theta]
    ground_offset = [float(ground_offset_mm) * value for value in ground_forward_unit]
    x_offset = [float(x_offset_mm), 0.0, 0.0]
    offset = _add_vectors(z_height_offset, ground_offset, x_offset)
    above = [float(center_xyz_mm[i]) + offset[i] for i in range(3)]
    return {
        "method": "z_only_projection",
        "note": "legacy height mode changes only Z first; optional ground_offset_mm is still added along the ground-parallel direction",
        "camera_to_vertical_deg": round(float(camera_to_vertical_deg), 3),
        "height_above_center_mm": round(float(height_mm), 1),
        "ground_offset_mm": round(float(ground_offset_mm), 1),
        "x_offset_mm": round(float(x_offset_mm), 1),
        "z_height_offset_xyz_mm": _round_list(z_height_offset),
        "ground_offset_xyz_mm": _round_list(ground_offset),
        "x_offset_xyz_mm": _round_list(x_offset),
        "offset_xyz_mm": _round_list(offset),
        "center_xyz_mm": _round_list(center_xyz_mm),
        "above_xyz_mm": _round_list(above),
        "z_cos_projection_mm": {
            "center": round(float(center_xyz_mm[2]) * cos_theta, 1),
            "above": round(float(above[2]) * cos_theta, 1),
            "delta": round((float(above[2]) - float(center_xyz_mm[2])) * cos_theta, 1),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Return the point above the detected cigarette-box center in left-camera optical coordinates."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_POSE_URL,
        help="pose service URL; ignored when --xyz is provided",
    )
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--xyz",
        help="manual center xyz in mm, for example '-24.6,107.0,655.3' or JSON",
    )
    parser.add_argument("--height-mm", type=float, default=100.0)
    parser.add_argument(
        "--ground-offset-mm",
        type=float,
        default=DEFAULT_GROUND_OFFSET_MM,
        help=(
            "offset along the ground-parallel direction perpendicular to camera X; "
            "positive follows the ground projection of camera +Z, negative reverses it"
        ),
    )
    parser.add_argument(
        "--x-offset-mm",
        type=float,
        default=0.0,
        help="optional extra offset along camera +X after vertical/ground offsets",
    )
    parser.add_argument(
        "--camera-to-vertical-deg",
        type=float,
        default=DEFAULT_CAMERA_TO_VERTICAL_DEG,
        help="angle between camera +Z and vertical-down ground direction",
    )
    parser.add_argument(
        "--method",
        choices=("ground_vertical_yz_vector", "z_only_projection"),
        default="ground_vertical_yz_vector",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    pose: dict[str, Any] | None = None
    if args.xyz:
        center_xyz_mm = _parse_xyz(args.xyz)
    else:
        pose = _fetch_pose(args.url, args.timeout_sec)
        center_xyz_mm = _extract_center_xyz_mm(pose)

    if args.method == "ground_vertical_yz_vector":
        result = above_center_yz_vector(
            center_xyz_mm,
            height_mm=args.height_mm,
            camera_to_vertical_deg=args.camera_to_vertical_deg,
            ground_offset_mm=args.ground_offset_mm,
            x_offset_mm=args.x_offset_mm,
        )
    else:
        result = above_center_z_only(
            center_xyz_mm,
            height_mm=args.height_mm,
            camera_to_vertical_deg=args.camera_to_vertical_deg,
            ground_offset_mm=args.ground_offset_mm,
            x_offset_mm=args.x_offset_mm,
        )

    result["coordinate_system"] = {
        "frame_id": "left_camera_optical",
        "x": "+X is image right",
        "y": "+Y is image down",
        "z": "+Z is camera forward/depth",
        "unit": "mm",
    }
    if pose is not None:
        result["source_pose"] = {
            "url": args.url,
            "range_from_left_camera_mm": pose.get("range_from_left_camera_mm"),
            "left_yolo_class_name": pose.get("left_yolo_class_name"),
            "left_yolo_selected_candidate_index": pose.get("left_yolo_selected_candidate_index"),
            "object_top_size_mm": pose.get("object_top_size_mm"),
            "server": pose.get("server"),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
