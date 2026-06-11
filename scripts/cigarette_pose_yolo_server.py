#!/usr/bin/env python3
"""Resident YOLO pose service for the cigarette-box top-face center.

The command-line API is convenient for one-off tests, but it starts a new
Python process each time. This service keeps the process alive, so the YOLO
model stays cached on the Jetson GPU between requests.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cigarette_pose_optical_api import build_arg_parser, run_pose  # noqa: E402
from yolo_topface_detector import (  # noqa: E402
    YOLO_MODEL_CACHE,
    _ensure_torchvision_nms,
    get_yolo_model,
    resolve_model_path,
    resolve_yolo_device,
)


@dataclass(frozen=True)
class ServerConfig:
    bind: str = "127.0.0.1"
    port: int = 18081
    out_root: Path = Path("/tmp/cigarette_pose_yolo_server")
    yolo_model: str = "models/Liqun_Xiongmao.pt"
    yolo_device: str = "cuda:0"
    yolo_conf: float = 0.15
    yolo_imgsz: int = 640
    yolo_mask_threshold: float = 0.5
    focal_px: float = 260.0
    warmup: bool = True


POSE_LOCK = threading.Lock()
REQUEST_COUNTER = 0
REQUEST_COUNTER_LOCK = threading.Lock()
IMAGE_CLIENTS: dict[str, Any] = {}
LATEST_RESULT: dict[str, Any] | None = None
RESULT_CACHE: dict[str, dict[str, Any]] = {}
RESULT_CACHE_ORDER: list[str] = []
RESULT_CACHE_LOCK = threading.Lock()
MAX_CACHED_RESULTS = 20

OVERRIDE_TYPES: dict[str, type] = {
    "known_range_mm": float,
    "lens_glass_to_optical_center_mm": float,
    "focal_px": float,
    "yolo_conf": float,
    "yolo_imgsz": int,
    "yolo_mask_threshold": float,
    "yolo_select": str,
    "yolo_index": int,
    "yolo_label": str,
    "yolo_class_name": str,
    "label": str,
    "max_reproj_px": float,
    "max_depth_delta_mm": float,
    "stereo_baseline_mm": float,
    "wait_sec": float,
    "host": str,
    "calibrate_focal_from_known_range": bool,
    "camera_to_vertical_deg": float,
    "center_above_height_mm": float,
    "box_head_above_height_mm": float,
    "box_head_fraction_from_head": float,
}


def _next_request_id() -> str:
    global REQUEST_COUNTER
    with REQUEST_COUNTER_LOCK:
        REQUEST_COUNTER += 1
        counter = REQUEST_COUNTER
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{counter:04d}"


def _remember_result(result: dict[str, Any]) -> None:
    server = result.get("server")
    if not isinstance(server, dict):
        return
    request_id = str(server.get("request_id") or "")
    if not request_id:
        return
    with RESULT_CACHE_LOCK:
        if request_id not in RESULT_CACHE:
            RESULT_CACHE_ORDER.append(request_id)
        RESULT_CACHE[request_id] = result
        while len(RESULT_CACHE_ORDER) > MAX_CACHED_RESULTS:
            old_request_id = RESULT_CACHE_ORDER.pop(0)
            RESULT_CACHE.pop(old_request_id, None)


def _cached_result_for_request(handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
    parsed = urlparse(handler.path)
    query = parse_qs(parsed.query)
    request_ids = query.get("request_id") or query.get("rid") or []
    request_id = request_ids[-1] if request_ids else ""
    if not request_id:
        return LATEST_RESULT
    with RESULT_CACHE_LOCK:
        return RESULT_CACHE.get(request_id)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    _send_no_cache_headers(handler)
    handler.end_headers()
    handler.wfile.write(data)


def _send_no_cache_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")


def _file_response(handler: BaseHTTPRequestHandler, status: int, path: Path, content_type: str) -> None:
    data = path.read_bytes()
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    _send_no_cache_headers(handler)
    handler.end_headers()
    handler.wfile.write(data)


def _bytes_response(handler: BaseHTTPRequestHandler, status: int, data: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    _send_no_cache_headers(handler)
    handler.end_headers()
    handler.wfile.write(data)


def _html_response(handler: BaseHTTPRequestHandler, status: int, body: str) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    _send_no_cache_headers(handler)
    handler.end_headers()
    handler.wfile.write(data)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "y", "on")


def _coerce_override(name: str, value: Any) -> Any:
    target_type = OVERRIDE_TYPES[name]
    if target_type is bool:
        return _parse_bool(value)
    return target_type(value)


def _request_overrides(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    parsed = urlparse(handler.path)
    values: dict[str, Any] = {}
    for key, raw_values in parse_qs(parsed.query).items():
        normalized = key.replace("-", "_")
        if normalized in OVERRIDE_TYPES and raw_values:
            values[normalized] = _coerce_override(normalized, raw_values[-1])

    length = int(handler.headers.get("Content-Length") or 0)
    if length > 0:
        body = handler.rfile.read(length)
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request JSON body must be an object")
        for key, value in data.items():
            normalized = key.replace("-", "_")
            if normalized in OVERRIDE_TYPES:
                values[normalized] = _coerce_override(normalized, value)
    return values


def _append_arg(argv: list[str], name: str, value: Any) -> None:
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    argv.extend([flag, str(value)])


def _get_image_client(host: str) -> Any:
    if host not in IMAGE_CLIENTS:
        from teleimager.image_client import ImageClient

        IMAGE_CLIENTS[host] = ImageClient(host=host)
    return IMAGE_CLIENTS[host]


def _capture_head_images_persistent(host: str, wait_sec: float) -> tuple[np.ndarray, np.ndarray]:
    client = _get_image_client(host)
    deadline = time.monotonic() + float(wait_sec)
    while time.monotonic() < deadline:
        frame, _fps = client.get_head_frame()
        if frame is not None:
            height, width = frame.shape[:2]
            if width % 2 != 0:
                raise RuntimeError(f"expected side-by-side binocular image with even width, got {frame.shape}")
            return frame[:, : width // 2].copy(), frame[:, width // 2 :].copy()
        time.sleep(0.02)
    raise TimeoutError(f"no head camera frame after {float(wait_sec):.1f}s")


def _build_pose_args(
    config: ServerConfig,
    out_dir: Path,
    left_image: Path,
    right_image: Path,
    overrides: dict[str, Any],
) -> argparse.Namespace:
    argv = [
        "--mode",
        "yolo",
        "--left-image",
        str(left_image),
        "--right-image",
        str(right_image),
        "--yolo-model",
        config.yolo_model,
        "--yolo-device",
        config.yolo_device,
        "--yolo-conf",
        str(config.yolo_conf),
        "--yolo-imgsz",
        str(config.yolo_imgsz),
        "--yolo-mask-threshold",
        str(config.yolo_mask_threshold),
        "--focal-px",
        str(config.focal_px),
        "--out-dir",
        str(out_dir),
    ]
    for name, value in overrides.items():
        _append_arg(argv, name, value)
    return build_arg_parser().parse_args(argv)


def _device_from_result(result: dict[str, Any]) -> str | None:
    point_adjustments = result.get("point_adjustments")
    if not isinstance(point_adjustments, dict):
        return None
    left_yolo = point_adjustments.get("left_yolo")
    if not isinstance(left_yolo, dict):
        return None
    device = left_yolo.get("device")
    return str(device) if device is not None else None


def _left_yolo_selection_summary(result: dict[str, Any]) -> dict[str, Any]:
    point_adjustments = result.get("point_adjustments")
    if not isinstance(point_adjustments, dict):
        return {}
    left_yolo = point_adjustments.get("left_yolo")
    candidates = point_adjustments.get("left_yolo_candidates")
    if not isinstance(left_yolo, dict):
        return {}
    return {
        "left_yolo_candidate_count": len(candidates) if isinstance(candidates, list) else left_yolo.get("candidate_count"),
        "left_yolo_selected_candidate_index": left_yolo.get("selected_candidate_index"),
        "left_yolo_selection_method": left_yolo.get("selection_method"),
        "left_yolo_raw_yolo_index": left_yolo.get("raw_yolo_index"),
        "left_yolo_class_id": left_yolo.get("class_id"),
        "left_yolo_class_name": left_yolo.get("class_name"),
        "left_yolo_confidence": left_yolo.get("confidence"),
        "left_yolo_score": left_yolo.get("score"),
        "left_yolo_label_filter": left_yolo.get("label_filter"),
    }


def _compact_pose(result: dict[str, Any], exit_code: int) -> dict[str, Any]:
    compact = {
        "ok": bool(result.get("ok")),
        "exit_code": int(exit_code),
        "center_xyz_mm": result.get("center_xyz_mm"),
        "x_mm": result.get("x_mm"),
        "y_mm": result.get("y_mm"),
        "z_mm": result.get("z_mm"),
        "center_above_xyz_mm": result.get("center_above_xyz_mm"),
        "center_above": result.get("center_above"),
        "range_from_left_camera_mm": result.get("range_from_left_camera_mm"),
        "left_depth_mm": result.get("left_depth_mm"),
        "optical_axis_depth_mm": result.get("optical_axis_depth_mm"),
        "selected_orientation": result.get("selected_orientation"),
        "requested_yolo_label": result.get("requested_yolo_label"),
        "selected_yolo_label": result.get("selected_yolo_label"),
        "selected_yolo_class_id": result.get("selected_yolo_class_id"),
        "selected_yolo_confidence": result.get("selected_yolo_confidence"),
        "vertical_up_unit_xyz": result.get("vertical_up_unit_xyz"),
        "vertical_up_source": result.get("vertical_up_source"),
        "top_plane_camera_to_vertical_deg": result.get("top_plane_camera_to_vertical_deg"),
        "object_top_size_mm": result.get("object_top_size_mm"),
        "object_top_size_source": result.get("object_top_size_source"),
        "box_head_fraction_from_head": result.get("box_head_fraction_from_head"),
        "box_head_point_xyz_mm": result.get("box_head_point_xyz_mm"),
        "box_head_point": result.get("box_head_point"),
        "box_head_point_above_xyz_mm": result.get("box_head_point_above_xyz_mm"),
        "box_head_point_above": result.get("box_head_point_above"),
        "box_head_one_third_xyz_mm": result.get("box_head_one_third_xyz_mm"),
        "box_head_one_third": result.get("box_head_one_third"),
        "box_head_one_third_above_xyz_mm": result.get("box_head_one_third_above_xyz_mm"),
        "box_head_one_third_above": result.get("box_head_one_third_above"),
        "robot_alignment": result.get("robot_alignment"),
        "robot_alignment_hypotheses": result.get("robot_alignment_hypotheses"),
        "device": _device_from_result(result),
        "debug_images": result.get("debug_images"),
        "error_code": result.get("error_code"),
        "error": result.get("error"),
    }
    compact.update(_left_yolo_selection_summary(result))
    return compact


def _run_pose_request(config: ServerConfig, compact: bool, overrides: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    global LATEST_RESULT
    request_id = _next_request_id()
    started = time.perf_counter()
    with POSE_LOCK:
        out_dir = config.out_root / f"request_{request_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        host = str(overrides.get("host", "127.0.0.1"))
        wait_sec = float(overrides.get("wait_sec", 5.0))
        left_image, right_image = _capture_head_images_persistent(host, wait_sec)
        left_path = out_dir / "server_left_input.jpg"
        right_path = out_dir / "server_right_input.jpg"
        if not cv2.imwrite(str(left_path), left_image):
            raise RuntimeError(f"failed to write left image: {left_path}")
        if not cv2.imwrite(str(right_path), right_image):
            raise RuntimeError(f"failed to write right image: {right_path}")
        args = _build_pose_args(config, out_dir, left_path, right_path, overrides)
        result, exit_code = run_pose(args)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    server_info = {
        "resident": True,
        "pid": os.getpid(),
        "request_id": request_id,
        "elapsed_ms": elapsed_ms,
        "model_cache_size": len(YOLO_MODEL_CACHE),
    }
    result["server"] = server_info
    LATEST_RESULT = result
    _remember_result(result)
    if compact:
        response = _compact_pose(result, exit_code)
        response["server"] = server_info
    else:
        response = result
        response["exit_code"] = int(exit_code)
    return (200 if exit_code == 0 else 500), response


DEBUG_IMAGE_KEYS = {
    "/debug/left_points.jpg": "left_points",
    "/debug/right_points.jpg": "right_points",
    "/debug/left_input.jpg": "left_input",
    "/debug/right_input.jpg": "right_input",
    "/latest/left_points.jpg": "left_points",
    "/latest/right_points.jpg": "right_points",
    "/latest/left_input.jpg": "left_input",
    "/latest/right_input.jpg": "right_input",
}

PROJECTED_IMAGE_KEYS = {
    "/debug/left_projected.jpg": "left_points",
    "/debug/left_projected_zoom.jpg": "left_points",
    "/latest/left_projected.jpg": "left_points",
    "/latest/left_projected_zoom.jpg": "left_points",
}

CANDIDATE_IMAGE_KEYS = {
    "/debug/left_candidates.jpg": ("left_input", "left_yolo_candidates", "left_yolo"),
    "/debug/right_candidates.jpg": ("right_input", "right_yolo_candidates", "right_yolo"),
    "/latest/left_candidates.jpg": ("left_input", "left_yolo_candidates", "left_yolo"),
    "/latest/right_candidates.jpg": ("right_input", "right_yolo_candidates", "right_yolo"),
}

DEBUG_DASHBOARD_PATHS = {"/debug", "/debug/", "/debug/all", "/debug/all.html"}


def _project_xyz_to_px(payload: dict[str, Any], xyz_mm: Any) -> tuple[int, int] | None:
    values = _xyz_values(xyz_mm)
    if values is None:
        return None
    x_mm, y_mm, z_mm = values
    if abs(z_mm) <= 1e-9:
        return None
    intrinsics = payload.get("intrinsics_assumption")
    if not isinstance(intrinsics, dict):
        return None
    try:
        focal_px = float(intrinsics.get("focal_px"))
        cx = float(intrinsics.get("cx"))
        cy = float(intrinsics.get("cy"))
    except Exception:
        return None
    u = focal_px * x_mm / z_mm + cx
    v = focal_px * y_mm / z_mm + cy
    return int(round(u)), int(round(v))


def _xyz_values(xyz_mm: Any) -> tuple[float, float, float] | None:
    if not isinstance(xyz_mm, list) or len(xyz_mm) != 3:
        return None
    try:
        return tuple(float(value) for value in xyz_mm)  # type: ignore[return-value]
    except Exception:
        return None


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.52,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_label(image: np.ndarray, xy: tuple[int, int], label: str, color: tuple[int, int, int], dy: int = -10) -> None:
    x, y = xy
    cv2.circle(image, (x, y), 8, color, -1)
    cv2.circle(image, (x, y), 10, (255, 255, 255), 2)
    origin = (x + 10, max(18, y + dy))
    _draw_text(image, label, origin, color, scale=0.55)


def _projected_above_markers() -> list[tuple[str, str, tuple[int, int, int], int]]:
    return [
        ("C", "center_above_xyz_mm", (0, 0, 255), -18),
        ("H", "box_head_point_above_xyz_mm", (255, 0, 255), 20),
    ]


def _zoom_projected_overlay(image: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    xs: list[float] = []
    ys: list[float] = []
    for _, key, _, _ in _projected_above_markers():
        xy = _project_xyz_to_px(payload, payload.get(key))
        if xy is None:
            continue
        xs.append(float(xy[0]))
        ys.append(float(xy[1]))

    points = payload.get("points_px")
    if isinstance(points, list):
        for item in points:
            if isinstance(item, list) and len(item) >= 2:
                try:
                    xs.append(float(item[0]))
                    ys.append(float(item[1]))
                except Exception:
                    pass

    if not xs or not ys:
        return image

    h, w = image.shape[:2]
    padding = 80
    x1 = max(0, int(math.floor(min(xs) - padding)))
    y1 = max(0, int(math.floor(min(ys) - padding)))
    x2 = min(w, int(math.ceil(max(xs) + padding)))
    y2 = min(h, int(math.ceil(max(ys) + padding)))
    if x2 <= x1 or y2 <= y1:
        return image

    crop = image[y1:y2, x1:x2].copy()
    scale = 2.5
    zoomed = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _draw_text(zoomed, f"zoom x{scale:g}", (12, 28), (0, 255, 255), scale=0.7)
    return zoomed


def _draw_projected_points_overlay(payload: dict[str, Any], path: str) -> bytes:
    image_key = PROJECTED_IMAGE_KEYS[path]
    debug_images = payload.get("debug_images")
    if not isinstance(debug_images, dict):
        raise RuntimeError("debug_images not available")
    image_path = Path(str(debug_images.get(image_key, "")))
    if not image_path.exists():
        raise RuntimeError(f"image file not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    markers = _projected_above_markers()
    drawn = 0
    for label, key, color, dy in markers:
        xy = _project_xyz_to_px(payload, payload.get(key))
        if xy is None:
            continue
        _draw_label(image, xy, label, color, dy=dy)
        drawn += 1

    if drawn == 0:
        cv2.putText(image, "No projected XYZ points", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        _draw_text(image, "C=center_above  H=head_1/5_above", (12, 28), (255, 255, 255), scale=0.55)
        cv2.putText(
            image,
            "Projected from /xyz using u=fx*x/z+cx, v=fx*y/z+cy",
            (12, image.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Projected from /xyz using u=fx*x/z+cx, v=fx*y/z+cy",
            (12, image.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if "zoom" in path:
        image = _zoom_projected_overlay(image, payload)

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode projected overlay")
    return encoded.tobytes()


def _serve_debug_image(
    handler: BaseHTTPRequestHandler,
    config: ServerConfig,
    path: str,
    overrides: dict[str, Any],
) -> None:
    if path.startswith("/debug/"):
        status, payload = _run_pose_request(config, compact=False, overrides=overrides)
    else:
        status = 200
        payload = _cached_result_for_request(handler)
    if not isinstance(payload, dict):
        _json_response(handler, 404, {"ok": False, "error": "no latest pose result yet; call /pose or /debug/left_points.jpg first"})
        return
    image_key = DEBUG_IMAGE_KEYS[path]
    debug_images = payload.get("debug_images")
    if not isinstance(debug_images, dict) or image_key not in debug_images:
        _json_response(handler, 404, {"ok": False, "error": f"debug image not available: {image_key}", "result": payload})
        return
    image_path = Path(str(debug_images[image_key]))
    if not image_path.exists():
        _json_response(handler, 404, {"ok": False, "error": f"debug image file not found: {image_path}", "result": payload})
        return
    _file_response(handler, 200, image_path, "image/jpeg")


def _candidate_color(index: int) -> tuple[int, int, int]:
    colors = [
        (0, 255, 255),
        (0, 128, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 128, 0),
        (255, 0, 0),
    ]
    return colors[index % len(colors)]


def _draw_candidate_overlay(payload: dict[str, Any], path: str) -> bytes:
    image_key, candidates_key, selected_key = CANDIDATE_IMAGE_KEYS[path]
    debug_images = payload.get("debug_images")
    point_adjustments = payload.get("point_adjustments")
    if not isinstance(debug_images, dict):
        raise RuntimeError("debug_images not available")
    if not isinstance(point_adjustments, dict):
        raise RuntimeError("point_adjustments not available")
    image_path = Path(str(debug_images.get(image_key, "")))
    if not image_path.exists():
        raise RuntimeError(f"image file not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    candidates = point_adjustments.get(candidates_key)
    if not isinstance(candidates, list):
        candidates = []
    selected = point_adjustments.get(selected_key)
    selected_candidate_index = None
    if isinstance(selected, dict):
        selected_candidate_index = selected.get("selected_candidate_index")

    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        candidate_index = candidate.get("candidate_index", idx)
        color = _candidate_color(int(candidate_index or idx))
        thickness = 4 if candidate_index == selected_candidate_index else 2
        box = candidate.get("box_xyxy")
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        points = candidate.get("points_px")
        if isinstance(points, list) and len(points) == 4:
            pts = np.asarray(points, dtype=np.int32).reshape(4, 2)
            cv2.polylines(image, [pts], True, color, thickness)
            for point_idx, point in enumerate(pts):
                cv2.circle(image, tuple(point), 5, color, -1)
                cv2.putText(
                    image,
                    str(point_idx),
                    tuple(point + np.array([5, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        selected_mark = "* " if candidate_index == selected_candidate_index else ""
        class_name = candidate.get("class_name")
        label = f"{selected_mark}#{candidate_index} {class_name} conf={candidate.get('confidence')}"
        label_origin = (10, 24 + 24 * idx)
        if isinstance(box, list) and len(box) == 4:
            label_origin = (int(round(float(box[0]))), max(20, int(round(float(box[1]))) - 8))
        cv2.putText(image, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if not candidates:
        cv2.putText(image, "No YOLO candidates", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode candidate overlay")
    return encoded.tobytes()


def _serve_candidate_image(
    handler: BaseHTTPRequestHandler,
    config: ServerConfig,
    path: str,
    overrides: dict[str, Any],
) -> None:
    if path.startswith("/debug/"):
        status, payload = _run_pose_request(config, compact=False, overrides=overrides)
    else:
        status = 200
        payload = _cached_result_for_request(handler)
    if not isinstance(payload, dict):
        _json_response(handler, 404, {"ok": False, "error": "no latest pose result yet; call /pose or /debug first"})
        return
    try:
        data = _draw_candidate_overlay(payload, path)
    except Exception as exc:
        _json_response(handler, 500, {"ok": False, "error": str(exc), "result": payload})
        return
    _bytes_response(handler, 200, data, "image/jpeg")


def _serve_projected_image(
    handler: BaseHTTPRequestHandler,
    config: ServerConfig,
    path: str,
    overrides: dict[str, Any],
) -> None:
    if path.startswith("/debug/"):
        status, payload = _run_pose_request(config, compact=False, overrides=overrides)
    else:
        status = 200
        payload = _cached_result_for_request(handler)
    if not isinstance(payload, dict):
        _json_response(handler, 404, {"ok": False, "error": "no latest pose result yet; call /pose or /debug first"})
        return
    try:
        data = _draw_projected_points_overlay(payload, path)
    except Exception as exc:
        _json_response(handler, 500, {"ok": False, "error": str(exc), "result": payload})
        return
    _bytes_response(handler, 200, data, "image/jpeg")


def _candidate_table(candidates: Any) -> str:
    if not isinstance(candidates, list) or not candidates:
        return "<p>暂无 YOLO 候选。</p>"
    rows = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(item.get('candidate_index')))}</td>"
            f"<td>{html_lib.escape(str(item.get('raw_yolo_index')))}</td>"
            f"<td>{html_lib.escape(str(item.get('class_name')))}</td>"
            f"<td>{html_lib.escape(str(item.get('confidence')))}</td>"
            f"<td>{html_lib.escape(str(item.get('score')))}</td>"
            f"<td>{html_lib.escape(str(item.get('mask_area_px')))}</td>"
            f"<td>{html_lib.escape(str(item.get('box_xyxy')))}</td>"
            f"<td>{html_lib.escape(str(item.get('points_px')))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>候选序号</th><th>YOLO 原序号</th><th>类别</th><th>置信度</th><th>评分</th>"
        "<th>mask面积</th><th>检测框</th><th>四点像素</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _nested(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _fmt_number(value: Any, suffix: str = "", ndigits: int = 1) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    return f"{number:.{int(ndigits)}f}{suffix}"


def _fmt_list(values: Any, suffix: str = " mm") -> str:
    if not isinstance(values, (list, tuple)):
        return "-"
    return "[" + ", ".join(_fmt_number(value, "", 1) for value in values) + f"]{suffix}"


def _orientation_cn(value: Any) -> str:
    if value == "long_x_short_y":
        return "图中 0-1 / 3-2 是长边"
    if value == "short_x_long_y":
        return "图中 1-2 / 0-3 是长边"
    return str(value or "-")


def _error_cn(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if "YOLO found no detections" in text:
        return "未检测到烟盒目标"
    if "no head camera frame" in text:
        return "没有拿到头部相机画面"
    if "reprojection" in text:
        return "四点质量不足，重投影误差过大"
    return text


def _summary_tile(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="summary-tile">'
        f'<div class="summary-label">{html_lib.escape(label)}</div>'
        f'<div class="summary-value">{html_lib.escape(value)}</div>'
        f'<div class="summary-note">{html_lib.escape(note)}</div>'
        "</div>"
    )


def _key_summary_html(payload: dict[str, Any]) -> str:
    alignment = payload.get("robot_alignment")
    if not isinstance(alignment, dict):
        alignment = {}
    target = alignment.get("target")
    if not isinstance(target, dict):
        target = {}
    control = alignment.get("control_hint")
    if not isinstance(control, dict):
        control = {}

    label = payload.get("selected_yolo_label") or "-"
    confidence = payload.get("selected_yolo_confidence")
    selected = _orientation_cn(payload.get("selected_orientation"))
    camera_angle = _nested(alignment, "camera_to_vertical_deg")
    formula_inputs = target.get("ground_forward_formula_inputs")
    if not isinstance(formula_inputs, dict):
        formula_inputs = {}

    tiles = [
        _summary_tile("识别结果", f"{label} / conf {_fmt_number(confidence, '', 3)}", "当前用于计算的 YOLO 目标"),
        _summary_tile("长边判定", selected, "看左目四点图上的红色编号"),
        _summary_tile("中心点坐标", _fmt_list(payload.get("center_xyz_mm")), "left_camera_optical: X右 Y下 Z前"),
        _summary_tile("中心上方点", _fmt_list(payload.get("center_above_xyz_mm")), "默认中心点上方 100mm"),
        _summary_tile("直线距离", _fmt_number(target.get("range_from_left_camera_mm"), " mm", 1), "左目光心到中心点"),
        _summary_tile("地面前向", _fmt_number(target.get("ground_forward_mm"), " mm", 1), "Z*sin角度 - Y*cos角度"),
        _summary_tile("相机角度", _fmt_number(camera_angle, " deg", 1), "当前用于地面投影"),
        _summary_tile(
            "前向公式输入",
            f"Y={_fmt_number(formula_inputs.get('y_mm'), '', 1)} Z={_fmt_number(formula_inputs.get('z_mm'), '', 1)}",
            "单位 mm",
        ),
        _summary_tile("左右偏差", _fmt_number(target.get("right_mm"), " mm", 1), "正数在机器人右侧"),
        _summary_tile("朝目标转角", _fmt_number(control.get("turn_first_yaw_deg"), " deg", 2), "负数通常向右转"),
        _summary_tile("烟盒长轴角", _fmt_number(control.get("box_parallel_yaw_deg"), " deg", 2), "末端/身体对齐烟盒参考"),
        _summary_tile("服务耗时", _fmt_number(_nested(payload, "server", "elapsed_ms"), " ms", 1), "本次拍照+推理+计算"),
    ]
    return '<div class="summary-grid">' + "".join(tiles) + "</div>"


def _alignment_hypotheses_table(payload: dict[str, Any]) -> str:
    hypotheses = payload.get("robot_alignment_hypotheses")
    if not isinstance(hypotheses, dict) or not hypotheses:
        return "<p>暂无横/竖假设结果。</p>"

    rows = []
    for name in ORIENTATION_DISPLAY_ORDER:
        item = hypotheses.get(name)
        if not isinstance(item, dict):
            continue
        alignment = item.get("robot_alignment")
        if not isinstance(alignment, dict):
            alignment = {}
        target = alignment.get("target")
        if not isinstance(target, dict):
            target = {}
        control = alignment.get("control_hint")
        if not isinstance(control, dict):
            control = {}
        selected = "是" if item.get("selected") else ""
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(selected)}</td>"
            f"<td>{html_lib.escape(_orientation_cn(name))}</td>"
            f"<td>{html_lib.escape(_fmt_list(item.get('object_top_size_mm'), ' mm'))}</td>"
            f"<td>{html_lib.escape(_fmt_number(item.get('range_from_left_camera_mm'), ' mm', 1))}</td>"
            f"<td>{html_lib.escape(_fmt_number(target.get('ground_forward_mm'), ' mm', 1))}</td>"
            f"<td>{html_lib.escape(_fmt_number(target.get('right_mm'), ' mm', 1))}</td>"
            f"<td>{html_lib.escape(_fmt_number(control.get('turn_first_yaw_deg'), ' deg', 2))}</td>"
            f"<td>{html_lib.escape(_fmt_number(control.get('box_parallel_yaw_deg'), ' deg', 2))}</td>"
            f"<td>{html_lib.escape(_fmt_number(item.get('left_reprojection_error_px'), ' px', 3))}</td>"
            f"<td>{html_lib.escape(_fmt_number(item.get('depth_delta_mm'), ' mm', 1))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>选中</th><th>长边对应哪组点</th><th>上表面尺寸</th><th>直线距离</th>"
        "<th>地面前向</th><th>左右偏差</th><th>朝目标转角</th><th>烟盒长轴角</th>"
        "<th>重投影误差</th><th>左右深度差</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


ORIENTATION_DISPLAY_ORDER = ("long_x_short_y", "short_x_long_y")


def _debug_dashboard_html(payload: dict[str, Any]) -> str:
    point_adjustments = payload.get("point_adjustments") if isinstance(payload, dict) else {}
    if not isinstance(point_adjustments, dict):
        point_adjustments = {}
    left_candidates = point_adjustments.get("left_yolo_candidates")
    right_candidates = point_adjustments.get("right_yolo_candidates")
    selected_left = point_adjustments.get("left_yolo")
    server = payload.get("server") if isinstance(payload, dict) else {}
    if not isinstance(server, dict):
        server = {}
    json_text = html_lib.escape(json.dumps(payload, ensure_ascii=False, indent=2))
    selected_text = html_lib.escape(json.dumps(selected_left, ensure_ascii=False, indent=2))
    ok_text = html_lib.escape(str(payload.get("ok")))
    error_text = html_lib.escape(str(payload.get("error") or ""))
    request_id_raw = str(server.get("request_id") or "")
    request_id = html_lib.escape(request_id_raw)
    elapsed_ms = html_lib.escape(str(server.get("elapsed_ms") or ""))
    cache_token = quote(f"{request_id_raw}_{time.time_ns()}")
    if request_id_raw:
        image_query = f"?request_id={quote(request_id_raw)}&t={cache_token}"
    else:
        image_query = f"?t={cache_token}"

    def image_src(path: str) -> str:
        return html_lib.escape(path + image_query)

    image_cards = "\n".join(
        [
            f'<section><h2>左目原图</h2><img src="{image_src("/latest/left_input.jpg")}" alt="left input"></section>',
            f'<section><h2>左目四点</h2><img src="{image_src("/latest/left_points.jpg")}" alt="left points"></section>',
            f'<section><h2>左目上方点</h2><img src="{image_src("/latest/left_projected.jpg")}" alt="left projected above points"></section>',
            f'<section class="wide"><h2>左目上方点放大</h2><img src="{image_src("/latest/left_projected_zoom.jpg")}" alt="left projected above points zoom"></section>',
            f'<section><h2>左目全部候选</h2><img src="{image_src("/latest/left_candidates.jpg")}" alt="left candidates"></section>',
            f'<section><h2>右目原图</h2><img src="{image_src("/latest/right_input.jpg")}" alt="right input"></section>',
            f'<section><h2>右目四点</h2><img src="{image_src("/latest/right_points.jpg")}" alt="right points"></section>',
            f'<section><h2>右目全部候选</h2><img src="{image_src("/latest/right_candidates.jpg")}" alt="right candidates"></section>',
        ]
    )
    run_again_href = html_lib.escape(f"/debug?t={cache_token}")
    status_line = html_lib.escape(
        f"状态={ok_text} 请求={request_id} 耗时={elapsed_ms}ms 错误={_error_cn(payload.get('error'))}"
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>烟盒 YOLO 调试</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; color: #1f2933; }}
    header {{ display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }}
    a.button {{ display: inline-block; padding: 7px 10px; border: 1px solid #8795a1; color: #102a43; text-decoration: none; }}
    .status {{ margin: 12px 0; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 14px 0 18px; }}
    .summary-tile {{ border: 1px solid #bcccdc; border-radius: 6px; padding: 10px; background: #f8fbff; min-height: 76px; }}
    .summary-label {{ color: #52606d; font-size: 13px; margin-bottom: 6px; }}
    .summary-value {{ color: #102a43; font-size: 20px; font-weight: 700; line-height: 1.2; word-break: break-word; }}
    .summary-note {{ color: #627d98; font-size: 12px; margin-top: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    section {{ border: 1px solid #d9e2ec; padding: 10px; }}
    img {{ width: 100%; max-width: 640px; height: auto; display: block; background: #f0f4f8; }}
    .wide {{ grid-column: 1 / -1; }}
    .wide img {{ max-width: 1100px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 5px; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    pre {{ overflow: auto; background: #102a43; color: #f0f4f8; padding: 12px; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>烟盒 YOLO 调试</h1>
    <a class="button" href="{run_again_href}">重新拍照</a>
    <a class="button" href="/pose">完整 JSON</a>
    <a class="button" href="/xyz">坐标 JSON</a>
  </header>
  <div class="status">{status_line}</div>
  <h2>关键数据</h2>
  {_key_summary_html(payload)}
  <h2>横竖两套假设对比</h2>
  {_alignment_hypotheses_table(payload)}
  <div class="grid">{image_cards}</div>
  <h2>当前选中的左目候选</h2>
  <pre>{selected_text}</pre>
  <h2>左目 YOLO 候选</h2>
  {_candidate_table(left_candidates)}
  <h2>右目 YOLO 候选</h2>
  {_candidate_table(right_candidates)}
  <h2>完整 JSON</h2>
  <pre>{json_text}</pre>
</body>
</html>
"""


def _serve_debug_dashboard(
    handler: BaseHTTPRequestHandler,
    config: ServerConfig,
    overrides: dict[str, Any],
) -> None:
    _status, payload = _run_pose_request(config, compact=False, overrides=overrides)
    _html_response(handler, 200, _debug_dashboard_html(payload))


def _health_payload(config: ServerConfig) -> dict[str, Any]:
    resolved_device = resolve_yolo_device(config.yolo_device)
    cuda_available: bool | None = None
    torch_cuda: str | None = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_cuda = torch.version.cuda
    except Exception:
        pass
    return {
        "ok": True,
        "service": "cigarette_pose_yolo_server",
        "resident": True,
        "pid": os.getpid(),
        "bind": config.bind,
        "port": config.port,
        "yolo_model": config.yolo_model,
        "requested_device": config.yolo_device,
        "resolved_device": resolved_device,
        "cuda_available": cuda_available,
        "torch_cuda": torch_cuda,
        "model_cache_size": len(YOLO_MODEL_CACHE),
        "endpoints": [
            "/health",
            "/pose",
            "/xyz",
            "/debug/left_points.jpg",
            "/debug/right_points.jpg",
            "/debug/left_input.jpg",
            "/debug/right_input.jpg",
            "/debug/left_candidates.jpg",
            "/debug/right_candidates.jpg",
            "/debug/left_projected.jpg",
            "/debug/left_projected_zoom.jpg",
            "/latest/left_points.jpg",
            "/latest/right_points.jpg",
            "/latest/left_input.jpg",
            "/latest/right_input.jpg",
            "/latest/left_candidates.jpg",
            "/latest/right_candidates.jpg",
            "/latest/left_projected.jpg",
            "/latest/left_projected_zoom.jpg",
            "/debug",
        ],
    }


def _warmup(config: ServerConfig) -> None:
    resolved_model = resolve_model_path(config.yolo_model)
    model = get_yolo_model(resolved_model, task="segment")
    _ensure_torchvision_nms()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    resolved_device = resolve_yolo_device(config.yolo_device)
    model.predict(
        blank,
        conf=config.yolo_conf,
        imgsz=config.yolo_imgsz,
        retina_masks=True,
        verbose=False,
        device=resolved_device,
    )
    if resolved_device.startswith("cuda"):
        try:
            import torch

            torch.cuda.synchronize()
        except Exception:
            pass


def make_handler(config: ServerConfig) -> type[BaseHTTPRequestHandler]:
    class PoseHandler(BaseHTTPRequestHandler):
        server_version = "CigarettePoseYoloServer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:
            self._handle()

        def do_POST(self) -> None:
            self._handle()

        def _handle(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/health":
                    _json_response(self, 200, _health_payload(config))
                    return
                if path in DEBUG_IMAGE_KEYS:
                    overrides = _request_overrides(self)
                    _serve_debug_image(self, config, path, overrides)
                    return
                if path in PROJECTED_IMAGE_KEYS:
                    overrides = _request_overrides(self)
                    _serve_projected_image(self, config, path, overrides)
                    return
                if path in CANDIDATE_IMAGE_KEYS:
                    overrides = _request_overrides(self)
                    _serve_candidate_image(self, config, path, overrides)
                    return
                if path in DEBUG_DASHBOARD_PATHS:
                    overrides = _request_overrides(self)
                    _serve_debug_dashboard(self, config, overrides)
                    return
                if path not in ("/pose", "/xyz"):
                    _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {path}"})
                    return
                overrides = _request_overrides(self)
                status, payload = _run_pose_request(config, compact=(path == "/xyz"), overrides=overrides)
                _json_response(self, status, payload)
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})

    return PoseHandler


def build_arg_parser_server() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resident HTTP service for YOLO cigarette pose.")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/cigarette_pose_yolo_server"))
    parser.add_argument("--yolo-model", default="models/Liqun_Xiongmao.pt")
    parser.add_argument("--yolo-device", default="cuda:0")
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-mask-threshold", type=float, default=0.5)
    parser.add_argument("--focal-px", type=float, default=260.0)
    parser.add_argument("--no-warmup", action="store_true", help="skip model/GPU warmup at service startup")
    return parser


def main() -> int:
    args = build_arg_parser_server().parse_args()
    config = ServerConfig(
        bind=args.bind,
        port=args.port,
        out_root=args.out_root,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
        yolo_mask_threshold=args.yolo_mask_threshold,
        focal_px=args.focal_px,
        warmup=not args.no_warmup,
    )
    config.out_root.mkdir(parents=True, exist_ok=True)
    if config.warmup:
        started = time.perf_counter()
        _warmup(config)
        print(f"YOLO warmup finished in {(time.perf_counter() - started) * 1000.0:.1f} ms", flush=True)
    server = ThreadingHTTPServer((config.bind, config.port), make_handler(config))
    print(f"serving on http://{config.bind}:{config.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
