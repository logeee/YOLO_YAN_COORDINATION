#!/usr/bin/env python3
"""Cigarette-box pose API in the left camera optical frame.

This is the handoff-friendly entry point for downstream grasp code. It returns
the cigarette top-face center in the left camera optical coordinate frame:

    +X: image right
    +Y: image down
    +Z: camera forward/depth

The default detector is the YOLO segmentation model for the top face. Points
mode is included so a future keypoint model can provide the four top-face
corners directly and reuse the same PnP coordinate conversion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
TELEIMAGER_SRC = Path.home() / "unitree_eai_environment" / "service" / "teleimager" / "src"
if TELEIMAGER_SRC.exists() and str(TELEIMAGER_SRC) not in sys.path:
    sys.path.insert(0, str(TELEIMAGER_SRC))

from auto_pnp_cuboid_depth import (
    Detection,
    capture_head_images,
    detect_top_quad,
    draw_debug,
    order_quad,
    parse_roi,
    solve_depth,
)
from yolo_topface_detector import YOLO_SELECT_METHODS, detect_yolo_points_from_image
from cigarette_pose_alignment import compute_robot_alignment


ORIENTATIONS = ("long_x_short_y", "short_x_long_y")
DEFAULT_CAMERA_TO_VERTICAL_DEG = 42.4
DEFAULT_CENTER_ABOVE_HEIGHT_MM = 100.0
DEFAULT_BOX_HEAD_ABOVE_HEIGHT_MM = 100.0
DEFAULT_BOX_HEAD_FRACTION_FROM_HEAD = 0.2

# Physical top-face sizes used by YOLO class-aware PnP.
# Values are (long side, short side) in meters.
YOLO_CLASS_TOP_SIZES_M: dict[str, tuple[float, float]] = {
    "XiongMao": (0.161, 0.095),
    "Xizi_Liqun": (0.280, 0.089),
}


@dataclass(frozen=True)
class PoseConfig:
    long_side_m: float = 0.161
    short_side_m: float = 0.095
    orientation: str = "auto_by_stereo"
    focal_px: float = 260.0
    cx: float = 320.0
    cy: float = 240.0
    left_roi: tuple[int, int, int, int] | None = (190, 215, 500, 420)
    right_roi: tuple[int, int, int, int] | None = (170, 225, 385, 400)
    min_red_fraction: float = 0.0
    margin_px: int = 20
    max_reproj_px: float = 3.0
    max_depth_delta_mm: float = 100.0
    stereo_baseline_mm: float = 60.0


def optical_coordinate_convention() -> dict[str, str]:
    return {
        "frame_id": "left_camera_optical",
        "x": "+X is image right",
        "y": "+Y is image down",
        "z": "+Z is camera forward/depth",
        "left_depth_mm": "optical-axis depth, same as z_mm; not Euclidean range or robot-base forward distance",
        "range_from_left_camera_mm": "straight-line distance from left camera optical center to target center",
        "unit": "mm",
    }


def _mm(value_m: float) -> float:
    return round(float(value_m) * 1000.0, 1)


def _mm_list(values_m: list[float]) -> list[float]:
    return [_mm(value) for value in values_m]


def _round_list(values: list[float], ndigits: int = 1) -> list[float]:
    return [round(float(value), int(ndigits)) for value in values]


def _class_top_size_source(config: PoseConfig, yolo_info: dict[str, Any] | None) -> dict[str, Any]:
    class_name = str(yolo_info.get("class_name", "")) if isinstance(yolo_info, dict) else ""
    class_size = YOLO_CLASS_TOP_SIZES_M.get(class_name)
    if class_size is None:
        source = "args_long_short_side"
        long_side_m = float(config.long_side_m)
        short_side_m = float(config.short_side_m)
    else:
        source = "yolo_class_top_size"
        long_side_m, short_side_m = class_size
    return {
        "source": source,
        "class_name": class_name or None,
        "long_side_mm": _mm(long_side_m),
        "short_side_mm": _mm(short_side_m),
        "known_class_sizes_mm": {
            name: [_mm(long_side_m), _mm(short_side_m)]
            for name, (long_side_m, short_side_m) in YOLO_CLASS_TOP_SIZES_M.items()
        },
    }


def _config_for_yolo_class(
    config: PoseConfig,
    yolo_info: dict[str, Any] | None,
) -> tuple[PoseConfig, dict[str, Any]]:
    size_source = _class_top_size_source(config, yolo_info)
    if size_source["source"] != "yolo_class_top_size":
        return config, size_source
    return (
        replace(
            config,
            long_side_m=float(size_source["long_side_mm"]) / 1000.0,
            short_side_m=float(size_source["short_side_mm"]) / 1000.0,
        ),
        size_source,
    )


def _points_list(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points.tolist()]


def _opencv_camera_to_optical_xyz(opencv_xyz_mm: list[float]) -> list[float]:
    """Convert OpenCV camera coordinates to the team's optical frame.

    cv2.solvePnP returns OpenCV camera coordinates:
        X_cv = image right
        Y_cv = image down
        Z_cv = camera forward/depth

    The requested optical frame in the handoff image is:
        X = image right
        Y = image down
        Z = camera forward/depth
    """
    x_cv, y_cv, z_cv = opencv_xyz_mm
    return [x_cv, y_cv, z_cv]


def _optical_to_opencv_camera_xyz(optical_xyz_mm: list[float]) -> list[float]:
    x_opt, y_opt, z_opt = optical_xyz_mm
    return [x_opt, y_opt, z_opt]


def _range_mm(values_mm: list[float]) -> float:
    return round(math.sqrt(sum(float(value) * float(value) for value in values_mm)), 1)


def _direction_unit(values_mm: list[float]) -> list[float]:
    length = math.sqrt(sum(float(value) * float(value) for value in values_mm))
    if length <= 1e-9:
        raise ValueError("cannot compute direction for a zero-length vector")
    return [round(float(value) / length, 6) for value in values_mm]


def _xyz_from_range_and_direction(range_mm: float, direction_unit_xyz: list[float]) -> list[float]:
    return [round(float(range_mm) * float(value), 1) for value in direction_unit_xyz]


def _vertical_up_offset_xyz_mm(
    height_mm: float,
    camera_to_vertical_deg: float = DEFAULT_CAMERA_TO_VERTICAL_DEG,
) -> list[float]:
    theta = math.radians(float(camera_to_vertical_deg))
    return [
        0.0,
        -float(height_mm) * math.sin(theta),
        -float(height_mm) * math.cos(theta),
    ]


def _vertical_up_unit_from_camera_angle(camera_to_vertical_deg: float = DEFAULT_CAMERA_TO_VERTICAL_DEG) -> np.ndarray:
    theta = math.radians(float(camera_to_vertical_deg))
    return np.asarray([0.0, -math.sin(theta), -math.cos(theta)], dtype=np.float64)


def _camera_to_vertical_deg_from_up_unit(up_unit_xyz: list[float] | np.ndarray) -> float:
    up = np.asarray(up_unit_xyz, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(up))
    if norm <= 1e-9:
        return float("nan")
    up = up / norm
    vertical_down_z = float(-up[2])
    vertical_down_z = max(-1.0, min(1.0, vertical_down_z))
    return math.degrees(math.acos(vertical_down_z))


def _top_plane_up_unit_xyz(
    rotation_matrix: Any,
    camera_to_vertical_deg: float = DEFAULT_CAMERA_TO_VERTICAL_DEG,
) -> list[float]:
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    normal = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise ValueError("cannot compute top-plane normal from a zero-length vector")
    normal = normal / norm
    expected_up = _vertical_up_unit_from_camera_angle(camera_to_vertical_deg)
    if float(np.dot(normal, expected_up)) < 0.0:
        normal = -normal
    return [float(value) for value in normal.tolist()]


def _above_point_detail(
    base_xyz_mm: list[float],
    height_mm: float = DEFAULT_BOX_HEAD_ABOVE_HEIGHT_MM,
    camera_to_vertical_deg: float = DEFAULT_CAMERA_TO_VERTICAL_DEG,
    vertical_up_unit_xyz: list[float] | None = None,
    vertical_up_source: str = "configured_camera_pitch",
) -> dict[str, Any]:
    if vertical_up_unit_xyz is None:
        offset = _vertical_up_offset_xyz_mm(height_mm, camera_to_vertical_deg)
        up_unit = _vertical_up_unit_from_camera_angle(camera_to_vertical_deg)
        angle_deg = float(camera_to_vertical_deg)
        method = "ground_vertical_up_from_configured_camera_pitch"
    else:
        up_unit = np.asarray(vertical_up_unit_xyz, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(up_unit))
        if norm <= 1e-9:
            raise ValueError("vertical_up_unit_xyz cannot be zero-length")
        up_unit = up_unit / norm
        offset = [float(height_mm) * float(value) for value in up_unit.tolist()]
        angle_deg = _camera_to_vertical_deg_from_up_unit(up_unit)
        method = "ground_vertical_up_from_pnp_top_face_normal"
    above = [float(base_xyz_mm[i]) + float(offset[i]) for i in range(3)]
    return {
        "point_xyz_mm": _round_list(above),
        "base_xyz_mm": _round_list(base_xyz_mm),
        "height_above_base_mm": round(float(height_mm), 1),
        "camera_to_vertical_deg": round(float(angle_deg), 3),
        "configured_camera_to_vertical_deg": round(float(camera_to_vertical_deg), 3),
        "vertical_up_unit_xyz": _round_list(up_unit.tolist(), 6),
        "vertical_up_source": vertical_up_source,
        "vertical_up_offset_xyz_mm": _round_list(offset),
        "method": method,
    }


def estimate_rectified_stereo_depth(
    left_points: np.ndarray,
    right_points: np.ndarray,
    config: PoseConfig,
) -> dict[str, Any]:
    """Estimate depth from horizontal disparity as a stereo diagnostic.

    This assumes rectified, nearly parallel stereo images. The measured
    lens-surface distance is used as an approximate baseline, so this output is
    a consistency check unless a proper stereo calibration is available.
    """
    left = np.asarray(left_points, dtype=np.float64).reshape(4, 2)
    right = np.asarray(right_points, dtype=np.float64).reshape(4, 2)
    corner_disparities = left[:, 0] - right[:, 0]
    mean_disparity = float(np.mean(corner_disparities))
    center_left = np.mean(left, axis=0)
    center_right = np.mean(right, axis=0)
    center_disparity = float(center_left[0] - center_right[0])
    valid = mean_disparity > 1e-6 and config.stereo_baseline_mm > 0.0
    result: dict[str, Any] = {
        "enabled": True,
        "method": "rectified_disparity_diagnostic",
        "assumption": "Z = focal_px * baseline_mm / disparity_px; valid only for rectified/parallel stereo",
        "baseline_mm": round(float(config.stereo_baseline_mm), 1),
        "baseline_source": "measured lens-surface peak-to-peak distance; optical-center baseline may differ",
        "left_center_px": [round(float(center_left[0]), 2), round(float(center_left[1]), 2)],
        "right_center_px": [round(float(center_right[0]), 2), round(float(center_right[1]), 2)],
        "corner_disparities_px": [round(float(value), 2) for value in corner_disparities.tolist()],
        "mean_disparity_px": round(mean_disparity, 3),
        "center_disparity_px": round(center_disparity, 3),
        "valid": bool(valid),
    }
    if not valid:
        result["warning"] = "non-positive disparity or baseline; stereo depth not computed"
        return result

    z_mm = float(config.focal_px) * float(config.stereo_baseline_mm) / mean_disparity
    x_mm = (float(center_left[0]) - float(config.cx)) * z_mm / float(config.focal_px)
    y_mm = (float(center_left[1]) - float(config.cy)) * z_mm / float(config.focal_px)
    xyz_mm = [round(x_mm, 1), round(y_mm, 1), round(z_mm, 1)]
    result.update(
        {
            "stereo_z_mm": round(z_mm, 1),
            "stereo_center_xyz_mm": xyz_mm,
            "stereo_range_from_left_camera_mm": _range_mm(xyz_mm),
        }
    )
    return result


def scale_pose_to_known_range(
    pose: dict[str, Any],
    known_range_mm: float,
    lens_glass_to_optical_center_mm: float = 0.0,
) -> dict[str, Any]:
    """Scale a pose vector so its range matches a measured straight-line range.

    `known_range_mm` is the measured distance from the left lens glass surface
    to the cigarette top-face center. If the optical center offset behind the
    glass is known, pass it as `lens_glass_to_optical_center_mm`; otherwise the
    default treats the glass surface and optical center as coincident.
    """
    target_range_mm = float(known_range_mm) + float(lens_glass_to_optical_center_mm)
    if target_range_mm <= 0.0:
        raise ValueError("--known-range-mm must be positive")

    original_center = [float(value) for value in pose["center_xyz_mm"]]
    original_range = float(pose["range_from_left_camera_mm"])
    if original_range <= 1e-6:
        raise ValueError("cannot scale a zero-length pose vector")

    direction_unit_xyz = pose.get("direction_unit_xyz")
    if direction_unit_xyz is None:
        direction_unit_xyz = _direction_unit(original_center)
    scale = target_range_mm / original_range
    scaled_center = _xyz_from_range_and_direction(target_range_mm, direction_unit_xyz)
    scaled_opencv = [round(value, 1) for value in _optical_to_opencv_camera_xyz(scaled_center)]
    original_opencv = pose.get("opencv_camera_xyz_mm")

    updated = dict(pose)
    updated.update(
        {
            "center_xyz_mm": scaled_center,
            "x_mm": scaled_center[0],
            "y_mm": scaled_center[1],
            "z_mm": scaled_center[2],
            "depth_mm": scaled_center[2],
            "optical_axis_depth_mm": scaled_center[2],
            "range_from_left_camera_mm": round(target_range_mm, 1),
            "opencv_camera_xyz_mm": scaled_opencv,
            "direction_unit_xyz": direction_unit_xyz,
            "coordinate_method": "range_times_direction",
            "range_override": {
                "enabled": True,
                "known_range_from_left_lens_glass_mm": round(float(known_range_mm), 1),
                "lens_glass_to_optical_center_mm": round(float(lens_glass_to_optical_center_mm), 1),
                "applied_range_from_left_optical_center_mm": round(target_range_mm, 1),
                "scale": round(scale, 6),
                "pnp_unscaled_center_xyz_mm": [round(value, 1) for value in original_center],
                "pnp_unscaled_range_from_left_camera_mm": round(original_range, 1),
            },
        }
    )
    if original_opencv is not None:
        updated["range_override"]["pnp_unscaled_opencv_camera_xyz_mm"] = original_opencv
    for key in ("box_head_point_xyz_mm", "box_head_one_third_xyz_mm"):
        if key in pose:
            original_point = [float(value) for value in pose[key]]
            scaled_point = [round(float(value) * scale, 1) for value in original_point]
            updated[key] = scaled_point
            updated["range_override"][f"pnp_unscaled_{key}"] = [round(value, 1) for value in original_point]
    for detail_key in ("box_head_point", "box_head_one_third"):
        if isinstance(updated.get(detail_key), dict) and "box_head_point_xyz_mm" in updated:
            box_head = dict(updated[detail_key])
            scaled_point = updated["box_head_point_xyz_mm"]
            box_head["point_xyz_mm"] = scaled_point
            box_head["x_mm"] = scaled_point[0]
            box_head["y_mm"] = scaled_point[1]
            box_head["z_mm"] = scaled_point[2]
            updated[detail_key] = box_head
    return updated


def _orientation_size(config: PoseConfig, orientation: str) -> tuple[float, float]:
    if orientation == "long_x_short_y":
        return config.long_side_m, config.short_side_m
    if orientation == "short_x_long_y":
        return config.short_side_m, config.long_side_m
    raise ValueError(f"unsupported orientation: {orientation}")


def _transform_local_to_optical_mm(
    local_points_m: np.ndarray,
    rotation_matrix: Any,
    center_xyz_m: Any,
) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    center = np.asarray(center_xyz_m, dtype=np.float64).reshape(3, 1)
    local = np.asarray(local_points_m, dtype=np.float64).reshape(-1, 3)
    opencv_xyz_m = (rotation @ local.T + center).T
    return np.asarray([_opencv_camera_to_optical_xyz(_mm_list(row.tolist())) for row in opencv_xyz_m], dtype=np.float64)


def _box_head_one_third_point(
    depth: dict[str, Any],
    width_m: float,
    height_m: float,
    fraction_from_head: float = 1.0 / 3.0,
) -> dict[str, Any]:
    if not 0.0 <= float(fraction_from_head) <= 1.0:
        raise ValueError("fraction_from_head must be in [0, 1]")

    if float(width_m) >= float(height_m):
        long_axis = "object_x"
        long_side_m = float(width_m)
        end_locals = np.asarray(
            [
                [-width_m / 2.0, 0.0, 0.0],
                [width_m / 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    else:
        long_axis = "object_y"
        long_side_m = float(height_m)
        end_locals = np.asarray(
            [
                [0.0, -height_m / 2.0, 0.0],
                [0.0, height_m / 2.0, 0.0],
            ],
            dtype=np.float64,
        )

    ends_xyz_mm = _transform_local_to_optical_mm(
        end_locals,
        depth["rotation_matrix"],
        depth["center_xyz_m"],
    )
    ranges = np.linalg.norm(ends_xyz_mm, axis=1)
    head_index = int(np.argmax(ranges))
    tail_index = 1 - head_index
    head_xyz = ends_xyz_mm[head_index]
    tail_xyz = ends_xyz_mm[tail_index]
    point_xyz = head_xyz + float(fraction_from_head) * (tail_xyz - head_xyz)
    direction_head_to_tail = tail_xyz - head_xyz
    direction_length = float(np.linalg.norm(direction_head_to_tail))
    if direction_length > 1e-9:
        direction_unit = direction_head_to_tail / direction_length
    else:
        direction_unit = np.zeros(3, dtype=np.float64)

    return {
        "point_xyz_mm": _round_list(point_xyz.tolist()),
        "x_mm": round(float(point_xyz[0]), 1),
        "y_mm": round(float(point_xyz[1]), 1),
        "z_mm": round(float(point_xyz[2]), 1),
        "fraction_from_head": round(float(fraction_from_head), 6),
        "distance_from_head_mm": round(float(long_side_m) * 1000.0 * float(fraction_from_head), 1),
        "head_definition": "farther long-axis end by Euclidean range from left camera optical center",
        "long_axis": long_axis,
        "long_side_mm": round(float(long_side_m) * 1000.0, 1),
        "head_xyz_mm": _round_list(head_xyz.tolist()),
        "tail_xyz_mm": _round_list(tail_xyz.tolist()),
        "head_range_mm": round(float(ranges[head_index]), 1),
        "tail_range_mm": round(float(ranges[tail_index]), 1),
        "head_to_tail_unit_xyz": _round_list(direction_unit.tolist(), 6),
    }


def _parse_points_json(data: Any) -> np.ndarray:
    if isinstance(data, dict):
        for key in ("points_px", "left_points_px", "points", "keypoints"):
            if key in data:
                return _parse_points_json(data[key])
        raise ValueError("points JSON object must contain points_px, left_points_px, points, or keypoints")

    points = np.asarray(data, dtype=np.float64)
    if points.shape == (8,):
        points = points.reshape(4, 2)
    if points.shape != (4, 2):
        raise ValueError(f"expected four 2D points, got shape {points.shape}")
    return points


def parse_points(text: str) -> np.ndarray:
    """Parse four image points from JSON, a JSON file, or flat CSV.

    Accepted examples:
        [[342,270],[372,270],[382,315],[345,313]]
        {"points_px": [[342,270],[372,270],[382,315],[345,313]]}
        342,270,372,270,382,315,345,313
        /tmp/yolo_points.json
    """
    try:
        possible_path = Path(text)
        if possible_path.exists():
            text = possible_path.read_text(encoding="utf-8")
    except OSError:
        # Inline JSON can contain characters that are illegal in some local
        # path syntaxes. Treat it as data instead of a path.
        pass

    try:
        return _parse_points_json(json.loads(text))
    except json.JSONDecodeError:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
        if len(values) != 8:
            raise ValueError("flat point CSV must contain exactly 8 numbers")
        return np.asarray(values, dtype=np.float64).reshape(4, 2)


def normalize_points(points_px: Any, points_order: str = "auto") -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64).reshape(4, 2)
    if points_order == "auto":
        return order_quad(points)
    if points_order == "ordered":
        return points
    raise ValueError(f"unsupported points_order: {points_order}")


def apply_bottom_edge_offset(points: np.ndarray, offset_px: float) -> np.ndarray:
    """Move bottom-left and bottom-right image points vertically.

    This is a calibration/debug aid for mask detections where the bottom edge
    sits inside the top face. It should stay at 0 once YOLO/keypoint corners are
    available.
    """
    adjusted = np.asarray(points, dtype=np.float64).copy().reshape(4, 2)
    if abs(float(offset_px)) > 1e-9:
        adjusted[2, 1] += float(offset_px)
        adjusted[3, 1] += float(offset_px)
    return adjusted


def refine_bottom_edge_from_bbox(
    points: np.ndarray,
    bbox: tuple[int, int, int, int],
    bottom_fraction: float = 0.93,
    min_adjust_px: float = 2.0,
    max_adjust_fraction: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Automatically lower bottom points when mask corners sit inside the face.

    The color mask often locks onto the printed dark region and misses the true
    lower top-face edge. The full target bbox includes the lower visible side,
    so a stable automatic estimate is near the lower part of that bbox. Move
    points 2/3 along their side rays instead of adding a fixed vertical offset.
    """
    adjusted = np.asarray(points, dtype=np.float64).copy().reshape(4, 2)
    x, y, w, h = bbox
    target_y = float(y) + float(h) * float(bottom_fraction)
    current_bottom_y = float((adjusted[2, 1] + adjusted[3, 1]) / 2.0)
    requested_delta = target_y - current_bottom_y
    max_delta = max(0.0, float(h) * float(max_adjust_fraction))
    applied = requested_delta > float(min_adjust_px)
    if applied:
        target_y = current_bottom_y + min(requested_delta, max_delta)

        def extend_to_y(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
            dy = float(bottom[1] - top[1])
            if abs(dy) < 1e-6:
                out = bottom.copy()
                out[1] = target_y
                return out
            scale = (target_y - float(top[1])) / dy
            return top + scale * (bottom - top)

        adjusted[2] = extend_to_y(adjusted[1], adjusted[2])
        adjusted[3] = extend_to_y(adjusted[0], adjusted[3])

    info = {
        "mode": "auto_bbox_fraction",
        "applied": bool(applied),
        "bottom_fraction": round(float(bottom_fraction), 3),
        "target_y": round(float(target_y), 2),
        "original_bottom_y": round(current_bottom_y, 2),
        "applied_delta_px": round(float(target_y - current_bottom_y), 2) if applied else 0.0,
        "requested_delta_px": round(float(requested_delta), 2),
        "max_delta_px": round(float(max_delta), 2),
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "raw_points_px": _points_list(points),
        "adjusted_points_px": _points_list(adjusted),
    }
    return adjusted, info


def estimate_pose_from_left_points(
    points_px: Any,
    config: PoseConfig | None = None,
    orientation: str | None = None,
    points_order: str = "auto",
    known_range_mm: float | None = None,
    lens_glass_to_optical_center_mm: float = 0.0,
    box_head_fraction_from_head: float = DEFAULT_BOX_HEAD_FRACTION_FROM_HEAD,
) -> dict[str, Any]:
    """Return pose from four left-image top-face points.

    Points are ordered as top-left, top-right, bottom-right, bottom-left after
    optional normalization. This function is the intended YOLO/keypoint bridge.
    """
    config = config or PoseConfig()
    selected_orientation = orientation or config.orientation
    if selected_orientation == "auto_by_stereo":
        selected_orientation = "short_x_long_y"

    points = normalize_points(points_px, points_order)
    width_m, height_m = _orientation_size(config, selected_orientation)
    depth = solve_depth(points, width_m, height_m, config.focal_px, config.cx, config.cy)
    top_plane_up_unit_xyz = _top_plane_up_unit_xyz(depth["rotation_matrix"])
    box_head_point = _box_head_one_third_point(
        depth,
        width_m,
        height_m,
        fraction_from_head=box_head_fraction_from_head,
    )
    opencv_camera_xyz_mm = _mm_list(depth["center_xyz_m"])
    pnp_center_xyz_mm = _opencv_camera_to_optical_xyz(opencv_camera_xyz_mm)
    range_from_left_camera_mm = _range_mm(pnp_center_xyz_mm)
    direction_unit_xyz = _direction_unit(pnp_center_xyz_mm)
    center_xyz_mm = _xyz_from_range_and_direction(range_from_left_camera_mm, direction_unit_xyz)
    depth_mm = center_xyz_mm[2]
    result = {
        "orientation": selected_orientation,
        "object_top_size_mm": [round(width_m * 1000.0, 1), round(height_m * 1000.0, 1)],
        "center_xyz_mm": center_xyz_mm,
        "x_mm": center_xyz_mm[0],
        "y_mm": center_xyz_mm[1],
        "z_mm": center_xyz_mm[2],
        "depth_mm": depth_mm,
        "optical_axis_depth_mm": depth_mm,
        "range_from_left_camera_mm": range_from_left_camera_mm,
        "direction_unit_xyz": direction_unit_xyz,
        "coordinate_method": "range_times_direction",
        "pnp_center_xyz_mm": [round(value, 1) for value in pnp_center_xyz_mm],
        "opencv_camera_xyz_mm": opencv_camera_xyz_mm,
        "opencv_camera_convention": {
            "x": "+X_cv is image right",
            "y": "+Y_cv is image down",
            "z": "+Z_cv is camera forward/depth",
        },
        "corner_depth_range_mm": _mm_list(depth["corner_depth_range_m"]),
        "reprojection_error_px": round(float(depth["mean_reprojection_px"]), 3),
        "points_px": _points_list(points),
        "box_head_fraction_from_head": round(float(box_head_fraction_from_head), 6),
        "box_head_point_xyz_mm": box_head_point["point_xyz_mm"],
        "box_head_point": box_head_point,
        "top_plane_up_unit_xyz": _round_list(top_plane_up_unit_xyz, 6),
        "top_plane_up_source": "pnp_top_face_normal",
        "top_plane_camera_to_vertical_deg": round(_camera_to_vertical_deg_from_up_unit(top_plane_up_unit_xyz), 3),
        # Backward-compatible aliases. The value now follows box_head_fraction_from_head.
        "box_head_one_third_xyz_mm": box_head_point["point_xyz_mm"],
        "box_head_one_third": box_head_point,
    }
    if known_range_mm is not None:
        result = scale_pose_to_known_range(
            result,
            known_range_mm=known_range_mm,
            lens_glass_to_optical_center_mm=lens_glass_to_optical_center_mm,
        )
    return result


# Backward-compatible descriptive alias for callers that think in PnP terms.
solve_left_pose_from_points = estimate_pose_from_left_points


def calibrate_focal_from_known_range(
    points_px: Any,
    known_range_mm: float,
    config: PoseConfig | None = None,
    orientation: str | None = None,
    points_order: str = "auto",
    focal_min_px: float = 120.0,
    focal_max_px: float = 420.0,
) -> dict[str, Any]:
    """Find the focal length whose PnP range matches a measured range."""
    config = config or PoseConfig()
    selected_orientation = orientation or config.orientation
    if selected_orientation == "auto_by_stereo":
        selected_orientation = "short_x_long_y"

    points = normalize_points(points_px, points_order)
    target = float(known_range_mm)
    if target <= 0.0:
        raise ValueError("known_range_mm must be positive")

    def eval_focal(focal_px: float) -> dict[str, Any]:
        return estimate_pose_from_left_points(
            points,
            config=replace(config, focal_px=float(focal_px)),
            orientation=selected_orientation,
            points_order="ordered",
        )

    low, high = float(focal_min_px), float(focal_max_px)
    low_range = float(eval_focal(low)["range_from_left_camera_mm"])
    high_range = float(eval_focal(high)["range_from_left_camera_mm"])
    if not (min(low_range, high_range) <= target <= max(low_range, high_range)):
        raise ValueError(
            f"known range {target:.1f}mm is outside focal search range "
            f"[{low:.1f}px -> {low_range:.1f}mm, {high:.1f}px -> {high_range:.1f}mm]"
        )

    increasing = high_range >= low_range
    for _ in range(50):
        mid = (low + high) / 2.0
        mid_range = float(eval_focal(mid)["range_from_left_camera_mm"])
        if (mid_range < target) == increasing:
            low = mid
        else:
            high = mid

    focal_px = (low + high) / 2.0
    result = eval_focal(focal_px)
    return {
        "known_range_mm": round(target, 1),
        "focal_px": round(focal_px, 3),
        "result_at_focal": {
            "center_xyz_mm": result["center_xyz_mm"],
            "range_from_left_camera_mm": result["range_from_left_camera_mm"],
            "direction_unit_xyz": result["direction_unit_xyz"],
            "left_reprojection_error_px": result["reprojection_error_px"],
        },
        "search_range_px": [round(focal_min_px, 1), round(focal_max_px, 1)],
    }


def detect_left_points_from_image(image: np.ndarray, config: PoseConfig | None = None) -> Detection:
    config = config or PoseConfig()
    return detect_top_quad(
        image,
        search_roi=config.left_roi,
        min_red_fraction=config.min_red_fraction,
        margin=config.margin_px,
    )


def estimate_pose_from_left_image(
    image: np.ndarray,
    config: PoseConfig | None = None,
    orientation: str | None = None,
) -> dict[str, Any]:
    config = config or PoseConfig()
    detection = detect_left_points_from_image(image, config)
    result = estimate_pose_from_left_points(
        detection.points,
        config=config,
        orientation=orientation or config.orientation,
        points_order="ordered",
    )
    result.update(
        {
            "target_bbox_xywh": [int(v) for v in detection.target_bbox],
            "mask_score": round(float(detection.score), 3),
            "red_fraction": round(float(detection.red_fraction), 4),
            "dark_fraction": round(float(detection.dark_fraction), 4),
            "mask_area_px": round(float(detection.mask_area), 1),
        }
    )
    return result


def _solve_view_hypotheses(
    points: np.ndarray,
    config: PoseConfig,
    prefix: str,
    box_head_fraction_from_head: float = DEFAULT_BOX_HEAD_FRACTION_FROM_HEAD,
) -> dict[str, dict[str, Any]]:
    return {
        orientation: estimate_pose_from_left_points(
            points,
            config=config,
            orientation=orientation,
            points_order="ordered",
            box_head_fraction_from_head=box_head_fraction_from_head,
        )
        for orientation in ORIENTATIONS
    }


def _choose_orientation(
    left_by_orientation: dict[str, dict[str, Any]],
    right_by_orientation: dict[str, dict[str, Any]] | None,
    requested: str,
) -> str:
    if requested != "auto_by_stereo":
        return requested

    if right_by_orientation is None:
        return min(
            ORIENTATIONS,
            key=lambda item: float(left_by_orientation[item]["reprojection_error_px"]),
        )

    left_ranked = sorted(
        ORIENTATIONS,
        key=lambda item: float(left_by_orientation[item]["reprojection_error_px"]),
    )
    best_left = float(left_by_orientation[left_ranked[0]]["reprojection_error_px"])
    second_left = float(left_by_orientation[left_ranked[1]]["reprojection_error_px"])
    if best_left <= 3.0 and (second_left - best_left) >= 2.0:
        return left_ranked[0]

    right_depth_deltas = {
        item: abs(float(left_by_orientation[item]["depth_mm"]) - float(right_by_orientation[item]["depth_mm"]))
        for item in ORIENTATIONS
    }
    if min(right_depth_deltas.values()) > 100.0 and best_left <= 3.0:
        return left_ranked[0]

    def score(item: str) -> float:
        left = left_by_orientation[item]
        right = right_by_orientation[item]
        depth_delta = right_depth_deltas[item]
        reproj = float(left["reprojection_error_px"]) + float(right["reprojection_error_px"])
        return depth_delta + 25.0 * reproj

    return min(ORIENTATIONS, key=score)


def _evaluate_point_candidate(
    left_points: np.ndarray,
    right_points: np.ndarray | None,
    config: PoseConfig,
    box_head_fraction_from_head: float = DEFAULT_BOX_HEAD_FRACTION_FROM_HEAD,
) -> dict[str, Any]:
    left_by_orientation = _solve_view_hypotheses(
        left_points,
        config,
        "left",
        box_head_fraction_from_head=box_head_fraction_from_head,
    )
    right_by_orientation = (
        _solve_view_hypotheses(
            right_points,
            config,
            "right",
            box_head_fraction_from_head=box_head_fraction_from_head,
        )
        if right_points is not None
        else None
    )
    selected_orientation = _choose_orientation(
        left_by_orientation,
        right_by_orientation,
        config.orientation,
    )
    selected_left = left_by_orientation[selected_orientation]
    selected_right = right_by_orientation[selected_orientation] if right_by_orientation else None
    left_reproj = float(selected_left["reprojection_error_px"])
    right_reproj = float(selected_right["reprojection_error_px"]) if selected_right is not None else 0.0
    depth_delta_mm = (
        abs(float(selected_left["depth_mm"]) - float(selected_right["depth_mm"]))
        if selected_right is not None
        else 0.0
    )
    reproj_weight_mm_per_px = 25.0
    score = depth_delta_mm + reproj_weight_mm_per_px * (left_reproj + right_reproj)
    return {
        "selected_orientation": selected_orientation,
        "score": round(float(score), 3),
        "score_formula": "depth_delta_mm + 25*(left_reprojection_error_px + right_reprojection_error_px)",
        "left_depth_mm": selected_left["depth_mm"],
        "right_depth_mm": selected_right["depth_mm"] if selected_right is not None else None,
        "depth_delta_mm": round(float(depth_delta_mm), 1) if selected_right is not None else None,
        "left_reprojection_error_px": round(left_reproj, 3),
        "right_reprojection_error_px": round(right_reproj, 3) if selected_right is not None else None,
        "range_from_left_camera_mm": selected_left["range_from_left_camera_mm"],
    }


def _orientation_meaning(orientation: str) -> str:
    if orientation == "long_x_short_y":
        return "physical long side is assigned to ordered image edges 0-1 and 3-2"
    if orientation == "short_x_long_y":
        return "physical long side is assigned to ordered image edges 1-2 and 0-3"
    return "unknown orientation hypothesis"


def _build_robot_alignment_hypotheses(
    left_by_orientation: dict[str, dict[str, Any]],
    right_by_orientation: dict[str, dict[str, Any]] | None,
    selected_orientation: str,
    camera_to_vertical_deg: float,
    known_range_mm: float | None = None,
    lens_glass_to_optical_center_mm: float = 0.0,
) -> dict[str, dict[str, Any]]:
    hypotheses: dict[str, dict[str, Any]] = {}
    for orientation in ORIENTATIONS:
        pose = left_by_orientation[orientation]
        if known_range_mm is not None:
            pose = scale_pose_to_known_range(
                pose,
                known_range_mm=known_range_mm,
                lens_glass_to_optical_center_mm=lens_glass_to_optical_center_mm,
            )
        right_pose = right_by_orientation[orientation] if right_by_orientation else None
        try:
            alignment = compute_robot_alignment(
                pose,
                camera_to_vertical_deg=camera_to_vertical_deg,
                target_key="center_xyz_mm",
            )
        except Exception as exc:
            alignment = {
                "ok": False,
                "error": str(exc),
            }
        hypotheses[orientation] = {
            "selected": orientation == selected_orientation,
            "meaning": _orientation_meaning(orientation),
            "object_top_size_mm": pose.get("object_top_size_mm"),
            "center_xyz_mm": pose.get("center_xyz_mm"),
            "range_from_left_camera_mm": pose.get("range_from_left_camera_mm"),
            "left_reprojection_error_px": pose.get("reprojection_error_px"),
            "right_reprojection_error_px": right_pose.get("reprojection_error_px") if right_pose else None,
            "depth_delta_mm": (
                round(abs(float(pose["depth_mm"]) - float(right_pose["depth_mm"])), 1)
                if right_pose is not None
                else None
            ),
            "box_head_point": pose.get("box_head_point"),
            "robot_alignment": alignment,
        }
    return hypotheses


def _load_image(path: Path | None, label: str) -> np.ndarray | None:
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {label} image: {path}")
    return image


def _load_or_capture(args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None]:
    if args.capture:
        return capture_head_images(args.host, args.wait_sec)
    return _load_image(args.left_image, "left"), _load_image(args.right_image, "right")


def _bbox_from_points(points: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    return int(x), int(y), int(w), int(h)


def _save_debug(
    image: np.ndarray | None,
    points: np.ndarray | None,
    bbox: tuple[int, int, int, int] | None,
    path: Path,
) -> str | None:
    if image is None or points is None:
        return None
    if bbox is None:
        bbox = _bbox_from_points(points)
    detection = Detection(points, bbox, 0.0, 0.0, 0.0, 0.0)
    draw_debug(image, detection, path)
    return str(path)


def _roi_edge_warnings(
    label: str,
    points: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    tolerance_px: float = 1.5,
) -> list[str]:
    if roi is None:
        return []
    x1, y1, x2, y2 = roi
    warnings: list[str] = []
    if float(points[:, 0].min()) <= x1 + tolerance_px:
        warnings.append(f"{label} points touch the left ROI edge; expand ROI x1 if this repeats")
    if float(points[:, 0].max()) >= x2 - tolerance_px:
        warnings.append(f"{label} points touch the right ROI edge; expand ROI x2 carefully")
    if float(points[:, 1].min()) <= y1 + tolerance_px:
        warnings.append(f"{label} points touch the top ROI edge; expand ROI y1 if needed")
    if float(points[:, 1].max()) >= y2 - tolerance_px:
        warnings.append(f"{label} points touch the bottom ROI edge; expand ROI y2 if needed")
    return warnings


def _build_config(args: argparse.Namespace) -> PoseConfig:
    return PoseConfig(
        long_side_m=args.long_side_m,
        short_side_m=args.short_side_m,
        orientation=args.orientation,
        focal_px=args.focal_px,
        cx=args.cx,
        cy=args.cy,
        left_roi=args.left_roi,
        right_roi=args.right_roi,
        min_red_fraction=args.min_red_fraction,
        margin_px=args.margin_px,
        max_reproj_px=args.max_reproj_px,
        max_depth_delta_mm=args.max_depth_delta_mm,
        stereo_baseline_mm=args.stereo_baseline_mm,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Return cigarette top-face center XYZ in the left camera optical frame."
    )
    parser.add_argument("--mode", choices=("yolo", "points", "mask"), default="yolo")
    parser.add_argument("--capture", action="store_true", help="capture from teleimager head camera")
    parser.add_argument("--host", default="127.0.0.1", help="teleimager host for --capture")
    parser.add_argument("--wait-sec", type=float, default=5.0)
    parser.add_argument("--left-image", type=Path, help="offline left image")
    parser.add_argument("--right-image", type=Path, help="optional offline right image")
    parser.add_argument("--left-points", help="four left top-face points for --mode points")
    parser.add_argument("--right-points", help="optional four right top-face points for --mode points")
    parser.add_argument("--yolo-model", default="models/Liqun_Xiongmao.pt", help="Ultralytics YOLO segmentation model")
    parser.add_argument("--yolo-conf", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--yolo-device", default="auto", help="YOLO device, e.g. auto, cpu, or cuda:0")
    parser.add_argument("--yolo-mask-threshold", type=float, default=0.5, help="YOLO mask binarization threshold")
    parser.add_argument(
        "--yolo-select",
        choices=YOLO_SELECT_METHODS,
        default="confidence",
        help="how to order multiple YOLO masks before choosing one for PnP",
    )
    parser.add_argument(
        "--yolo-index",
        type=int,
        default=0,
        help="candidate index after --yolo-select ordering; only this candidate is used for PnP",
    )
    parser.add_argument(
        "--yolo-label",
        "--yolo-class-name",
        "--label",
        dest="yolo_label",
        help=(
            "only use YOLO detections matching this class name/id, then select the highest-confidence "
            "candidate by default; examples: XiongMao, Xizi_Liqun, Liqun"
        ),
    )
    parser.add_argument(
        "--points-order",
        choices=("auto", "ordered"),
        default="auto",
        help="auto orders points as top-left, top-right, bottom-right, bottom-left",
    )

    parser.add_argument("--long-side-m", type=float, default=0.161)
    parser.add_argument("--short-side-m", type=float, default=0.095)
    parser.add_argument(
        "--orientation",
        choices=("long_x_short_y", "short_x_long_y", "auto_by_stereo"),
        default="auto_by_stereo",
        help="physical side assigned to the image top edge",
    )
    parser.add_argument("--focal-px", type=float, default=260.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--left-roi", type=parse_roi, default=parse_roi("190,215,500,420"))
    parser.add_argument("--right-roi", type=parse_roi, default=parse_roi("170,225,385,400"))
    parser.add_argument("--min-red-fraction", type=float, default=0.0)
    parser.add_argument("--margin-px", type=int, default=20)
    parser.add_argument(
        "--bottom-edge-mode",
        choices=("auto", "fixed", "none"),
        default="none",
        help=(
            "auto refines mask bottom corners from the target bbox; fixed uses "
            "--bottom-edge-offset-px; none keeps raw points"
        ),
    )
    parser.add_argument(
        "--bottom-edge-bbox-fraction",
        type=float,
        default=0.93,
        help="for --bottom-edge-mode auto, target bottom edge y = bbox_y + fraction*bbox_h",
    )
    parser.add_argument(
        "--bottom-edge-offset-px",
        type=float,
        default=0.0,
        help=(
            "debug/calibration only: add this many pixels to points 2 and 3 "
            "when mask bottom corners are detected too high"
        ),
    )
    parser.add_argument("--max-reproj-px", type=float, default=3.0)
    parser.add_argument("--max-depth-delta-mm", type=float, default=100.0)
    parser.add_argument(
        "--stereo-baseline-mm",
        type=float,
        default=60.0,
        help="left/right lens-surface peak-to-peak distance; used as approximate stereo diagnostic baseline",
    )
    parser.add_argument(
        "--known-range-mm",
        type=float,
        help=(
            "measured straight-line distance from the left lens glass surface "
            "to the cigarette top-face center; when set, output coordinates "
            "are scaled to this range"
        ),
    )
    parser.add_argument(
        "--lens-glass-to-optical-center-mm",
        type=float,
        default=0.0,
        help="optional approximate offset from left lens glass surface back to the optical center",
    )
    parser.add_argument(
        "--calibrate-focal-from-known-range",
        action="store_true",
        help="also report the focal_px that would make the left PnP range match --known-range-mm",
    )
    parser.add_argument(
        "--camera-to-vertical-deg",
        type=float,
        default=DEFAULT_CAMERA_TO_VERTICAL_DEG,
        help="angle between camera +Z and the ground vertical-down direction",
    )
    parser.add_argument(
        "--center-above-height-mm",
        type=float,
        default=DEFAULT_CENTER_ABOVE_HEIGHT_MM,
        help="vertical height added above center_xyz_mm",
    )
    parser.add_argument(
        "--box-head-above-height-mm",
        type=float,
        default=DEFAULT_BOX_HEAD_ABOVE_HEIGHT_MM,
        help="vertical height added above box_head_point_xyz_mm",
    )
    parser.add_argument(
        "--box-head-fraction-from-head",
        type=float,
        default=DEFAULT_BOX_HEAD_FRACTION_FROM_HEAD,
        help="fraction of the long side measured inward from the farther head end; default 0.2 means 1/5",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(f"/tmp/cigarette_pose_optical_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def run_pose(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _build_config(args)
    object_top_size_source: dict[str, Any] = _class_top_size_source(config, None)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    left_image: np.ndarray | None = None
    right_image: np.ndarray | None = None
    debug_images: dict[str, str] = {}
    warnings: list[str] = []

    try:
        if args.mode in ("mask", "yolo") or args.capture or args.left_image or args.right_image:
            left_image, right_image = _load_or_capture(args)
            if left_image is not None:
                left_input = out_dir / "left_input.jpg"
                cv2.imwrite(str(left_input), left_image)
                debug_images["left_input"] = str(left_input)
            if right_image is not None:
                right_input = out_dir / "right_input.jpg"
                cv2.imwrite(str(right_input), right_image)
                debug_images["right_input"] = str(right_input)
    except Exception as exc:
        result = {
            "ok": False,
            "error_code": "CAPTURE_OR_IMAGE_FAILED",
            "error": str(exc),
            "frame": "left_camera_optical",
            "coordinate_system": optical_coordinate_convention(),
        }
        return result, 2

    try:
        if args.mode == "points":
            if not args.left_points:
                raise RuntimeError("--mode points requires --left-points")
            left_points = normalize_points(parse_points(args.left_points), args.points_order)
            right_points = (
                normalize_points(parse_points(args.right_points), args.points_order)
                if args.right_points
                else None
            )
            left_bbox = _bbox_from_points(left_points)
            right_bbox = _bbox_from_points(right_points) if right_points is not None else None
            point_adjustments: dict[str, Any] = {
                "bottom_edge_mode": "none_for_points",
                "bottom_edge_offset_px": 0.0,
            }
            if args.bottom_edge_mode == "fixed":
                left_points = apply_bottom_edge_offset(left_points, args.bottom_edge_offset_px)
                right_points = (
                    apply_bottom_edge_offset(right_points, args.bottom_edge_offset_px)
                    if right_points is not None
                    else None
                )
                left_bbox = _bbox_from_points(left_points)
                right_bbox = _bbox_from_points(right_points) if right_points is not None else None
                point_adjustments = {
                    "bottom_edge_mode": "fixed_for_points",
                    "bottom_edge_offset_px": round(float(args.bottom_edge_offset_px), 2),
                }
            elif args.bottom_edge_mode == "auto":
                point_adjustments = {
                    "bottom_edge_mode": "none_for_points",
                    "note": "auto bottom-edge refinement is only for mask detections",
                    "bottom_edge_offset_px": 0.0,
                }
        elif args.mode == "mask":
            if left_image is None:
                raise RuntimeError("--mode mask requires --capture or --left-image")
            left_det = detect_left_points_from_image(left_image, config)
            left_bbox = left_det.target_bbox
            right_points = None
            right_bbox = None
            left_raw_points = left_det.points
            if right_image is not None:
                right_det = detect_top_quad(
                    right_image,
                    search_roi=config.right_roi,
                    min_red_fraction=config.min_red_fraction,
                    margin=config.margin_px,
                )
                right_raw_points = right_det.points
                right_bbox = right_det.target_bbox
            else:
                right_raw_points = None

            point_adjustments = {
                "bottom_edge_mode": args.bottom_edge_mode,
                "bottom_edge_offset_px": round(float(args.bottom_edge_offset_px), 2),
            }
            if args.bottom_edge_mode == "auto":
                auto_left_points, left_adjust = refine_bottom_edge_from_bbox(
                    left_raw_points,
                    left_bbox,
                    bottom_fraction=args.bottom_edge_bbox_fraction,
                )
                point_adjustments["left_auto_bottom_edge"] = left_adjust
                auto_right_points = None
                if right_raw_points is not None and right_bbox is not None:
                    auto_right_points, right_adjust = refine_bottom_edge_from_bbox(
                        right_raw_points,
                        right_bbox,
                        bottom_fraction=args.bottom_edge_bbox_fraction,
                    )
                    point_adjustments["right_auto_bottom_edge"] = right_adjust

                raw_eval = _evaluate_point_candidate(
                    left_raw_points,
                    right_raw_points,
                    config,
                    box_head_fraction_from_head=args.box_head_fraction_from_head,
                )
                auto_eval = _evaluate_point_candidate(
                    auto_left_points,
                    auto_right_points,
                    config,
                    box_head_fraction_from_head=args.box_head_fraction_from_head,
                )
                if left_adjust["applied"]:
                    selected_candidate = "bbox_fraction"
                    selection_reason = (
                        "left_primary_uses_bbox_refinement_when_mask_bottom_edge_adjustment_applies"
                    )
                else:
                    selected_candidate = "raw"
                    selection_reason = "left_bottom_edge_adjustment_not_applied"

                if selected_candidate == "bbox_fraction":
                    left_points = auto_left_points
                    right_points = auto_right_points
                else:
                    left_points = left_raw_points
                    right_points = right_raw_points
                point_adjustments["auto_candidate_selection"] = {
                    "selected": selected_candidate,
                    "reason": selection_reason,
                    "raw": raw_eval,
                    "bbox_fraction": auto_eval,
                }
            elif args.bottom_edge_mode == "fixed":
                left_points = apply_bottom_edge_offset(left_raw_points, args.bottom_edge_offset_px)
                if right_raw_points is not None:
                    right_points = apply_bottom_edge_offset(right_raw_points, args.bottom_edge_offset_px)
                point_adjustments["left_fixed_bottom_edge"] = {
                    "raw_points_px": _points_list(left_raw_points),
                    "adjusted_points_px": _points_list(left_points),
                }
                if right_raw_points is not None and right_points is not None:
                    point_adjustments["right_fixed_bottom_edge"] = {
                        "raw_points_px": _points_list(right_raw_points),
                        "adjusted_points_px": _points_list(right_points),
                    }
            else:
                left_points = left_raw_points
                right_points = right_raw_points
                point_adjustments["left_raw_points_px"] = _points_list(left_raw_points)
                if right_raw_points is not None:
                    point_adjustments["right_raw_points_px"] = _points_list(right_raw_points)
        else:
            if left_image is None:
                raise RuntimeError("--mode yolo requires --capture or --left-image")
            left_det, left_yolo = detect_yolo_points_from_image(
                left_image,
                model_path=args.yolo_model,
                conf=args.yolo_conf,
                imgsz=args.yolo_imgsz,
                device=args.yolo_device,
                mask_threshold=args.yolo_mask_threshold,
                select=args.yolo_select,
                select_index=args.yolo_index,
                label_filter=args.yolo_label,
            )
            left_yolo_candidates = left_yolo.pop("candidates", [])
            left_raw_points = left_det.points
            left_bbox = left_det.target_bbox
            right_raw_points = None
            right_bbox = None
            right_yolo: dict[str, Any] | None = None
            right_yolo_candidates: list[dict[str, Any]] = []
            if right_image is not None:
                try:
                    right_det, right_yolo = detect_yolo_points_from_image(
                        right_image,
                        model_path=args.yolo_model,
                        conf=args.yolo_conf,
                        imgsz=args.yolo_imgsz,
                        device=args.yolo_device,
                        mask_threshold=args.yolo_mask_threshold,
                        select=args.yolo_select,
                        select_index=args.yolo_index,
                        label_filter=args.yolo_label,
                    )
                    right_yolo_candidates = right_yolo.pop("candidates", [])
                    right_raw_points = right_det.points
                    right_bbox = right_det.target_bbox
                except Exception as exc:
                    warnings.append(f"right YOLO detection failed; left result is still used: {exc}")

            left_points = left_raw_points
            right_points = right_raw_points
            point_adjustments = {
                "detector": "yolo_segment",
                "bottom_edge_mode": "none_for_yolo" if args.bottom_edge_mode != "fixed" else "fixed_for_yolo",
                "bottom_edge_offset_px": round(float(args.bottom_edge_offset_px), 2),
                "left_yolo": left_yolo,
                "left_yolo_candidates": left_yolo_candidates,
                "yolo_selection": {
                    "method": args.yolo_select,
                    "index": int(args.yolo_index),
                    "label_filter": args.yolo_label,
                    "note": "YOLO may return multiple masks; only the selected candidate is used for PnP/XYZ",
                },
            }
            config, object_top_size_source = _config_for_yolo_class(config, left_yolo)
            point_adjustments["object_top_size_source"] = object_top_size_source
            if object_top_size_source["source"] != "yolo_class_top_size":
                warnings.append(
                    "selected YOLO class has no class-specific top size; "
                    "using --long-side-m/--short-side-m for PnP"
                )
            if right_yolo is not None:
                point_adjustments["right_yolo"] = right_yolo
                point_adjustments["right_yolo_candidates"] = right_yolo_candidates
                if (
                    right_yolo.get("class_name") is not None
                    and left_yolo.get("class_name") is not None
                    and right_yolo.get("class_name") != left_yolo.get("class_name")
                ):
                    warnings.append(
                        "right selected YOLO class differs from left; "
                        "XYZ uses the left selected class size"
                    )
            if args.bottom_edge_mode == "fixed":
                left_points = apply_bottom_edge_offset(left_raw_points, args.bottom_edge_offset_px)
                left_bbox = _bbox_from_points(left_points)
                if right_raw_points is not None:
                    right_points = apply_bottom_edge_offset(right_raw_points, args.bottom_edge_offset_px)
                    right_bbox = _bbox_from_points(right_points)
                point_adjustments["left_fixed_bottom_edge"] = {
                    "raw_points_px": _points_list(left_raw_points),
                    "adjusted_points_px": _points_list(left_points),
                }
                if right_raw_points is not None and right_points is not None:
                    point_adjustments["right_fixed_bottom_edge"] = {
                        "raw_points_px": _points_list(right_raw_points),
                        "adjusted_points_px": _points_list(right_points),
                    }
            elif args.bottom_edge_mode == "auto":
                point_adjustments["note"] = "auto bottom-edge refinement is not applied to YOLO mask points"
    except Exception as exc:
        result = {
            "ok": False,
            "error_code": "NO_TARGET_OR_POINTS",
            "error": str(exc),
            "frame": "left_camera_optical",
            "coordinate_system": optical_coordinate_convention(),
            "debug_images": debug_images,
        }
        return result, 3

    left_by_orientation = _solve_view_hypotheses(
        left_points,
        config,
        "left",
        box_head_fraction_from_head=args.box_head_fraction_from_head,
    )
    right_by_orientation = (
        _solve_view_hypotheses(
            right_points,
            config,
            "right",
            box_head_fraction_from_head=args.box_head_fraction_from_head,
        )
        if right_points is not None
        else None
    )
    selected_orientation = _choose_orientation(left_by_orientation, right_by_orientation, config.orientation)
    selected_left = left_by_orientation[selected_orientation]
    selected_right = right_by_orientation[selected_orientation] if right_by_orientation else None

    if args.mode == "mask" and args.bottom_edge_mode in ("auto", "fixed"):
        debug_left_raw = _save_debug(left_image, left_raw_points, left_bbox, out_dir / "left_points_raw.jpg")
        if debug_left_raw:
            debug_images["left_points_raw"] = debug_left_raw
        debug_right_raw = _save_debug(
            right_image,
            right_raw_points,
            right_bbox,
            out_dir / "right_points_raw.jpg",
        )
        if debug_right_raw:
            debug_images["right_points_raw"] = debug_right_raw

    debug_left = _save_debug(left_image, left_points, left_bbox, out_dir / "left_points.jpg")
    if debug_left:
        debug_images["left_points"] = debug_left
    debug_right = _save_debug(right_image, right_points, right_bbox, out_dir / "right_points.jpg")
    if debug_right:
        debug_images["right_points"] = debug_right

    if args.mode == "mask":
        warnings.extend(_roi_edge_warnings("left", left_points, config.left_roi))
        if right_points is not None:
            warnings.extend(_roi_edge_warnings("right", right_points, config.right_roi))

    left_ok = float(selected_left["reprojection_error_px"]) <= config.max_reproj_px
    depth_delta_mm: float | None = None
    right_consistency_ok: bool | None = None
    if selected_right is not None:
        depth_delta_mm = round(abs(float(selected_left["depth_mm"]) - float(selected_right["depth_mm"])), 1)
        right_consistency_ok = depth_delta_mm <= config.max_depth_delta_mm
        if not right_consistency_ok:
            warnings.append(
                f"right-left depth delta {depth_delta_mm:.1f}mm exceeds "
                f"{config.max_depth_delta_mm:.1f}mm; inspect point images before trusting depth"
            )

    if args.known_range_mm is not None:
        focal_calibration = None
        if args.calibrate_focal_from_known_range:
            focal_calibration = calibrate_focal_from_known_range(
                left_points,
                known_range_mm=args.known_range_mm + args.lens_glass_to_optical_center_mm,
                config=config,
                orientation=selected_orientation,
                points_order="ordered",
            )
        selected_left = scale_pose_to_known_range(
            selected_left,
            known_range_mm=args.known_range_mm,
            lens_glass_to_optical_center_mm=args.lens_glass_to_optical_center_mm,
        )
        warnings.append(
            "left pose was scaled to --known-range-mm; right_depth_mm and depth_delta_mm "
            "remain unscaled PnP consistency checks"
        )
    else:
        focal_calibration = None

    center_above = _above_point_detail(
        selected_left["center_xyz_mm"],
        height_mm=args.center_above_height_mm,
        camera_to_vertical_deg=args.camera_to_vertical_deg,
        vertical_up_unit_xyz=selected_left.get("top_plane_up_unit_xyz"),
        vertical_up_source=selected_left.get("top_plane_up_source", "pnp_top_face_normal"),
    )
    box_head_one_third_above = _above_point_detail(
        selected_left["box_head_point_xyz_mm"],
        height_mm=args.box_head_above_height_mm,
        camera_to_vertical_deg=args.camera_to_vertical_deg,
        vertical_up_unit_xyz=selected_left.get("top_plane_up_unit_xyz"),
        vertical_up_source=selected_left.get("top_plane_up_source", "pnp_top_face_normal"),
    )

    stereo_check = None
    if right_points is not None:
        stereo_check = estimate_rectified_stereo_depth(left_points, right_points, config)
        if stereo_check.get("valid"):
            stereo_check["vs_left_pnp_z_delta_mm"] = round(
                abs(float(stereo_check["stereo_z_mm"]) - float(selected_left["optical_axis_depth_mm"])),
                1,
            )
            stereo_check["vs_left_pnp_range_delta_mm"] = round(
                abs(
                    float(stereo_check["stereo_range_from_left_camera_mm"])
                    - float(selected_left["range_from_left_camera_mm"])
                ),
                1,
            )
            if float(stereo_check["vs_left_pnp_z_delta_mm"]) > 150.0:
                stereo_check["warning"] = (
                    "large stereo-vs-PnP delta; images may not be rectified, "
                    "right/left points may not correspond, or lens-surface baseline may differ from optical-center baseline"
                )

    robot_alignment_hypotheses = _build_robot_alignment_hypotheses(
        left_by_orientation,
        right_by_orientation,
        selected_orientation,
        camera_to_vertical_deg=args.camera_to_vertical_deg,
        known_range_mm=args.known_range_mm,
        lens_glass_to_optical_center_mm=args.lens_glass_to_optical_center_mm,
    )

    algorithm_by_mode = {
        "yolo": "yolo_seg_pnp_left_primary",
        "points": "points_pnp_left_primary",
        "mask": "mask_pnp_left_primary",
    }
    result: dict[str, Any] = {
        "ok": bool(left_ok),
        "algorithm": algorithm_by_mode[args.mode],
        "frame": "left_camera_optical",
        "coordinate_system": optical_coordinate_convention(),
        "selected_orientation": selected_orientation,
        "requested_yolo_label": args.yolo_label,
        "selected_yolo_label": left_yolo.get("class_name") if "left_yolo" in locals() and isinstance(left_yolo, dict) else None,
        "selected_yolo_class_id": left_yolo.get("class_id") if "left_yolo" in locals() and isinstance(left_yolo, dict) else None,
        "selected_yolo_confidence": (
            left_yolo.get("confidence") if "left_yolo" in locals() and isinstance(left_yolo, dict) else None
        ),
        "center_xyz_mm": selected_left["center_xyz_mm"],
        "x_mm": selected_left["x_mm"],
        "y_mm": selected_left["y_mm"],
        "z_mm": selected_left["z_mm"],
        "center_above_xyz_mm": center_above["point_xyz_mm"],
        "center_above": center_above,
        "left_depth_mm": selected_left["depth_mm"],
        "optical_axis_depth_mm": selected_left["optical_axis_depth_mm"],
        "range_from_left_camera_mm": selected_left["range_from_left_camera_mm"],
        "direction_unit_xyz": selected_left["direction_unit_xyz"],
        "coordinate_method": selected_left["coordinate_method"],
        "vertical_up_unit_xyz": selected_left.get("top_plane_up_unit_xyz"),
        "vertical_up_source": selected_left.get("top_plane_up_source"),
        "top_plane_camera_to_vertical_deg": selected_left.get("top_plane_camera_to_vertical_deg"),
        "opencv_camera_xyz_mm": selected_left["opencv_camera_xyz_mm"],
        "pnp_center_xyz_mm": selected_left["pnp_center_xyz_mm"],
        "left_reprojection_error_px": selected_left["reprojection_error_px"],
        "points_px": selected_left["points_px"],
        "object_top_size_mm": selected_left["object_top_size_mm"],
        "object_top_size_source": object_top_size_source,
        "box_head_fraction_from_head": selected_left["box_head_fraction_from_head"],
        "box_head_point_xyz_mm": selected_left["box_head_point_xyz_mm"],
        "box_head_point": selected_left["box_head_point"],
        "box_head_point_above_xyz_mm": box_head_one_third_above["point_xyz_mm"],
        "box_head_point_above": box_head_one_third_above,
        # Backward-compatible aliases. These now follow box_head_fraction_from_head.
        "box_head_one_third_xyz_mm": selected_left["box_head_one_third_xyz_mm"],
        "box_head_one_third": selected_left["box_head_one_third"],
        "box_head_one_third_above_xyz_mm": box_head_one_third_above["point_xyz_mm"],
        "box_head_one_third_above": box_head_one_third_above,
        "intrinsics_assumption": {"focal_px": config.focal_px, "cx": config.cx, "cy": config.cy},
        "stereo_baseline_mm": round(float(config.stereo_baseline_mm), 1),
        "point_adjustments": point_adjustments,
        "roi": (
            {
                "left": list(config.left_roi) if config.left_roi else None,
                "right": list(config.right_roi) if config.right_roi else None,
            }
            if args.mode == "mask"
            else None
        ),
        "quality": {
            "left_reprojection_ok": bool(left_ok),
            "right_consistency_ok": right_consistency_ok,
            "max_reproj_px": config.max_reproj_px,
            "max_depth_delta_mm": config.max_depth_delta_mm,
        },
        "debug_images": debug_images,
        "hypotheses": {
            orientation: {
                "left_depth_mm": left_by_orientation[orientation]["depth_mm"],
                "left_reprojection_error_px": left_by_orientation[orientation]["reprojection_error_px"],
                "right_depth_mm": (
                    right_by_orientation[orientation]["depth_mm"] if right_by_orientation else None
                ),
                "depth_delta_mm": (
                    round(
                        abs(
                            float(left_by_orientation[orientation]["depth_mm"])
                            - float(right_by_orientation[orientation]["depth_mm"])
                        ),
                        1,
                    )
                    if right_by_orientation
                    else None
                ),
            }
            for orientation in ORIENTATIONS
        },
    }
    if selected_right is not None:
        result["right_depth_mm"] = selected_right["depth_mm"]
        result["right_reprojection_error_px"] = selected_right["reprojection_error_px"]
        result["depth_delta_mm"] = depth_delta_mm
    if stereo_check is not None:
        result["stereo_check"] = stereo_check
    if "range_override" in selected_left:
        result["range_override"] = selected_left["range_override"]
    if focal_calibration is not None:
        result["focal_calibration"] = focal_calibration
    try:
        result["robot_alignment"] = compute_robot_alignment(
            result,
            camera_to_vertical_deg=args.camera_to_vertical_deg,
            target_key="center_xyz_mm",
        )
    except Exception as exc:
        result["robot_alignment"] = {
            "ok": False,
            "error": str(exc),
        }
    result["robot_alignment_hypotheses"] = robot_alignment_hypotheses
    if warnings:
        result["warnings"] = warnings
    if not left_ok:
        result["error_code"] = "LOW_CONFIDENCE"
        result["error"] = (
            f"left reprojection {selected_left['reprojection_error_px']:.3f}px "
            f"> {config.max_reproj_px:.3f}px"
        )

    text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    (out_dir / "pose_optical_result.json").write_text(text + "\n", encoding="utf-8")
    return result, 0 if result["ok"] else 4


def main() -> int:
    args = build_arg_parser().parse_args()
    result, exit_code = run_pose(args)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
