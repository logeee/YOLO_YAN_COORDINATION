#!/usr/bin/env python3
"""Resident YOLO pose service for the cigarette-box top-face center.

The command-line API is convenient for one-off tests, but it starts a new
Python process each time. This service keeps the process alive, so the YOLO
model stays cached on the Jetson GPU between requests.
"""

from __future__ import annotations

import argparse
import cgi
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
from urllib.parse import parse_qs, quote, urlencode, urlparse
import urllib.error
import urllib.request

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cigarette_pose_optical_api import (  # noqa: E402
    YOLO_CLASS_TOP_SIZES_M,
    build_arg_parser,
    parse_dist_coeffs,
    parse_float_list,
    run_pose,
)
from yolo_topface_detector import (  # noqa: E402
    YOLO_MODEL_CACHE,
    _ensure_torchvision_nms,
    get_yolo_model,
    resolve_model_path,
    resolve_yolo_device,
)

CALIBRATED_LEFT_FOCAL_PX = 275.06
CALIBRATED_LEFT_FX = 275.06
CALIBRATED_LEFT_FY = 275.39
CALIBRATED_LEFT_CX = 305.71
CALIBRATED_LEFT_CY = 268.34
CALIBRATED_LEFT_DIST_COEFFS = (0.05998239, -0.07112947, -0.00037432, 0.00015172, 0.01724672)

CALIBRATED_RIGHT_FX = 274.29699860633724
CALIBRATED_RIGHT_FY = 274.5716080713627
CALIBRATED_RIGHT_CX = 289.7163405945703
CALIBRATED_RIGHT_CY = 274.4892508669222
CALIBRATED_RIGHT_DIST_COEFFS = (
    0.06292257512401175,
    -0.07717484464783685,
    -0.000405354779537882,
    -0.00006950375556195126,
    0.019962308624586825,
)


@dataclass(frozen=True)
class ServerConfig:
    bind: str = "127.0.0.1"
    port: int = 18081
    out_root: Path = Path("/tmp/cigarette_pose_yolo_server")
    yolo_model: str = "models/YanHe20class.pt"
    yolo_device: str = "cuda:0"
    yolo_conf: float = 0.15
    yolo_imgsz: int = 640
    yolo_mask_threshold: float = 0.5
    focal_px: float = CALIBRATED_LEFT_FOCAL_PX
    fx: float | None = CALIBRATED_LEFT_FX
    fy: float | None = CALIBRATED_LEFT_FY
    cx: float = CALIBRATED_LEFT_CX
    cy: float = CALIBRATED_LEFT_CY
    dist_coeffs: tuple[float, ...] = CALIBRATED_LEFT_DIST_COEFFS
    fx_right: float | None = CALIBRATED_RIGHT_FX
    fy_right: float | None = CALIBRATED_RIGHT_FY
    cx_right: float | None = CALIBRATED_RIGHT_CX
    cy_right: float | None = CALIBRATED_RIGHT_CY
    dist_coeffs_right: tuple[float, ...] | None = CALIBRATED_RIGHT_DIST_COEFFS
    stereo_R: tuple[float, ...] | None = None
    stereo_T: tuple[float, ...] | None = None
    runtime_config_path: Path = Path("config/cigarette_pose_runtime.json")
    warmup: bool = True


POSE_LOCK = threading.Lock()
RUNTIME_CONFIG_LOCK = threading.Lock()
RUNTIME_INTRINSICS: dict[str, float] | None = None
RUNTIME_INTRINSICS_RIGHT: dict[str, Any] | None = None
RUNTIME_STEREO: dict[str, Any] | None = None
RUNTIME_YOLO_MODEL: str | None = None
RUNTIME_OBJECT_SIZES_MM: dict[str, dict[str, Any]] | None = None
REQUEST_COUNTER = 0
REQUEST_COUNTER_LOCK = threading.Lock()
IMAGE_CLIENTS: dict[str, Any] = {}
LATEST_RESULT: dict[str, Any] | None = None
RESULT_CACHE: dict[str, dict[str, Any]] = {}
RESULT_CACHE_ORDER: list[str] = []
RESULT_CACHE_LOCK = threading.Lock()
MAX_CACHED_RESULTS = 20
MODEL_CLASS_NAMES_CACHE: dict[str, list[str]] = {}
BUILTIN_OBJECT_SIZES_MM: dict[str, dict[str, float]] = {
    name: {
        "length_mm": round(float(long_side_m) * 1000.0, 1),
        "width_mm": round(float(short_side_m) * 1000.0, 1),
        "height_mm": 20.0,
    }
    for name, (long_side_m, short_side_m) in YOLO_CLASS_TOP_SIZES_M.items()
}
DEFAULT_OBJECT_SIZE_MM = {"length_mm": 161.0, "width_mm": 95.0, "height_mm": 20.0}
DEFAULT_CIGARETTE_NAMES = {
    "31019915": "熊猫(典藏中支)",
    "43010159": "白沙(和天下尊品中支)",
    "48013265": "娇子（五粮浓香中支）",
    "33013189": "利群（休闲金中支）",
    "42013109": "黄鹤楼(3mg)",
    "33013181": "利群(阳光尊细支)",
    "42013086": "黄鹤楼(逍遥6号)",
    "42013085": "黄鹤楼(雪之梦10号)",
    "34063141": "王冠(假日船长)",
    "34063140": "王冠(古雪1号)",
    "34063147": "王冠(假日·黄金海岸)",
    "42013097": "黄鹤楼(逍遥7号)",
    "42013096": "黄鹤楼(逍遥5号)",
    "34063142": "王冠(浪漫假日)",
    "42013075": "黄鹤楼(雪之梦5号)",
    "34063136": "王冠(国粹满堂彩)",
    "34063135": "王冠(假日阳光)",
    "48090225": "长城(盛世3号)",
    "48090217": "长城(经典3号)",
    "48090201": "长城(红色132)",
    "XiongMao": "熊猫烟",
    "Xizi_Liqun": "西子利群",
}
DEFAULT_CIGARETTE_IDS_BY_CLASS = {
    "0": "31019915",
    "1": "43010159",
    "2": "48013265",
    "3": "33013189",
    "4": "42013109",
    "5": "33013181",
    "6": "42013086",
    "7": "42013085",
    "8": "34063141",
    "9": "34063140",
    "10": "34063147",
    "11": "42013097",
    "12": "42013096",
    "13": "34063142",
    "14": "42013075",
    "15": "34063136",
    "16": "34063135",
    "17": "48090225",
    "18": "48090217",
    "19": "48090201",
}
DEFAULT_CIGARETTE_NAMES.update(
    {
        class_index: DEFAULT_CIGARETTE_NAMES[cigarette_id]
        for class_index, cigarette_id in DEFAULT_CIGARETTE_IDS_BY_CLASS.items()
    }
)
G1D_ADJUST_URL = "http://127.0.0.1:18084/adjust"
G1D_TARGET_ANGLE_ADJUST_URL = "http://127.0.0.1:18084/adjust_target_angle"
G1D_RIGHT_ENTRY_ADJUST_URL = "http://127.0.0.1:18084/adjust_right_entry"
G1D_ADJUST_TIMEOUT_SEC = 90.0
G1D_ADJUST_PROXY_TYPES: dict[str, type] = {
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
    "dry_run": bool,
}
OVERRIDE_TYPES: dict[str, type] = {
    "known_range_mm": float,
    "lens_glass_to_optical_center_mm": float,
    "focal_px": float,
    "fx": float,
    "fy": float,
    "cx": float,
    "cy": float,
    "dist_coeffs": list,
    "fx_right": float,
    "fy_right": float,
    "cx_right": float,
    "cy_right": float,
    "dist_coeffs_right": list,
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
    if name in ("dist_coeffs", "dist_coeffs_right"):
        return list(parse_dist_coeffs(value))
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


def _server_fx(config: ServerConfig) -> float:
    return float(config.fx if config.fx is not None else config.focal_px)


def _server_fy(config: ServerConfig) -> float:
    return float(config.fy if config.fy is not None else config.focal_px)


def _server_intrinsics(config: ServerConfig) -> dict[str, Any]:
    return {
        "focal_px": float(config.focal_px),
        "fx": _server_fx(config),
        "fy": _server_fy(config),
        "cx": float(config.cx),
        "cy": float(config.cy),
        "dist_coeffs": [float(value) for value in config.dist_coeffs],
    }


def _validate_intrinsics(values: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
    base_values = base or {
        "focal_px": CALIBRATED_LEFT_FOCAL_PX,
        "fx": CALIBRATED_LEFT_FX,
        "fy": CALIBRATED_LEFT_FY,
        "cx": CALIBRATED_LEFT_CX,
        "cy": CALIBRATED_LEFT_CY,
        "dist_coeffs": list(CALIBRATED_LEFT_DIST_COEFFS),
    }
    focal_px = float(values.get("focal_px", base_values.get("focal_px", CALIBRATED_LEFT_FOCAL_PX)))
    fx = float(values.get("fx", focal_px if "focal_px" in values else base_values.get("fx", focal_px)))
    fy = float(values.get("fy", focal_px if "focal_px" in values else base_values.get("fy", focal_px)))
    cx = float(values.get("cx", base_values.get("cx", CALIBRATED_LEFT_CX)))
    cy = float(values.get("cy", base_values.get("cy", CALIBRATED_LEFT_CY)))
    if fx <= 0.0 or fy <= 0.0 or focal_px <= 0.0:
        raise ValueError("focal_px/fx/fy must be positive")
    if "dist_coeffs" in values:
        dist_coeffs = list(parse_dist_coeffs(values.get("dist_coeffs")))
    else:
        dist_coeffs = [float(value) for value in base_values.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])]
    return {
        "focal_px": float(focal_px),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "dist_coeffs": dist_coeffs,
    }


def _runtime_intrinsics() -> dict[str, float] | None:
    with RUNTIME_CONFIG_LOCK:
        return dict(RUNTIME_INTRINSICS) if RUNTIME_INTRINSICS is not None else None


def _runtime_yolo_model_from_disk(config: ServerConfig) -> str | None:
    raw = _read_runtime_config_data(config).get("yolo_model")
    if not raw:
        return None
    try:
        resolved = _resolve_yolo_model_path(raw)
        if resolved.exists():
            return _model_path_for_config(resolved)
    except Exception:
        return None
    return None


def _runtime_yolo_model(config: ServerConfig | None = None) -> str | None:
    global RUNTIME_YOLO_MODEL
    with RUNTIME_CONFIG_LOCK:
        memory_value = str(RUNTIME_YOLO_MODEL) if RUNTIME_YOLO_MODEL else None
    if config is None:
        return memory_value
    disk_value = _runtime_yolo_model_from_disk(config)
    if disk_value and disk_value != memory_value:
        with RUNTIME_CONFIG_LOCK:
            RUNTIME_YOLO_MODEL = disk_value
        return disk_value
    return disk_value or memory_value


def _effective_yolo_model(config: ServerConfig) -> str:
    return _runtime_yolo_model(config) or str(config.yolo_model)


def _runtime_object_sizes() -> dict[str, dict[str, Any]] | None:
    with RUNTIME_CONFIG_LOCK:
        if RUNTIME_OBJECT_SIZES_MM is None:
            return None
        return {label: dict(size) for label, size in RUNTIME_OBJECT_SIZES_MM.items()}


def _effective_intrinsics(config: ServerConfig) -> dict[str, float]:
    runtime = _runtime_intrinsics()
    if runtime is not None:
        return runtime
    return _server_intrinsics(config)


def _request_intrinsics(config: ServerConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    names = {"focal_px", "fx", "fy", "cx", "cy", "dist_coeffs"}
    values = {name: overrides[name] for name in names if name in overrides}
    return _validate_intrinsics(values, base=_effective_intrinsics(config))


def _validate_intrinsics_right(values: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    fx = float(values.get("fx", base["fx"]))
    fy = float(values.get("fy", base["fy"]))
    cx = float(values.get("cx", base["cx"]))
    cy = float(values.get("cy", base["cy"]))
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("right fx/fy must be positive")
    if "dist_coeffs" in values:
        dist_coeffs = list(parse_dist_coeffs(values.get("dist_coeffs")))
    else:
        dist_coeffs = [float(value) for value in base.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])]
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "dist_coeffs": dist_coeffs}


def _server_intrinsics_right(config: ServerConfig) -> dict[str, Any]:
    left = _effective_intrinsics(config)
    return {
        "fx": float(config.fx_right) if config.fx_right is not None else float(left["fx"]),
        "fy": float(config.fy_right) if config.fy_right is not None else float(left["fy"]),
        "cx": float(config.cx_right) if config.cx_right is not None else float(left["cx"]),
        "cy": float(config.cy_right) if config.cy_right is not None else float(left["cy"]),
        "dist_coeffs": (
            [float(value) for value in config.dist_coeffs_right]
            if config.dist_coeffs_right is not None
            else [float(value) for value in left["dist_coeffs"]]
        ),
    }


def _runtime_intrinsics_right() -> dict[str, Any] | None:
    with RUNTIME_CONFIG_LOCK:
        return dict(RUNTIME_INTRINSICS_RIGHT) if RUNTIME_INTRINSICS_RIGHT is not None else None


def _effective_intrinsics_right(config: ServerConfig) -> dict[str, Any]:
    runtime = _runtime_intrinsics_right()
    if runtime is not None:
        return runtime
    return _server_intrinsics_right(config)


def _request_intrinsics_right(config: ServerConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "fx_right": "fx",
        "fy_right": "fy",
        "cx_right": "cx",
        "cy_right": "cy",
        "dist_coeffs_right": "dist_coeffs",
    }
    values = {dst: overrides[src] for src, dst in mapping.items() if src in overrides}
    return _validate_intrinsics_right(values, base=_effective_intrinsics_right(config))


def _validate_stereo(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("stereo config must be a JSON object with R and T")
    rotation = parse_float_list(data.get("R"), expected=9)
    translation = parse_float_list(data.get("T"), expected=3)
    if rotation is None or translation is None:
        raise ValueError("stereo config requires R (9 numbers) and T (3 numbers)")
    result: dict[str, Any] = {"R": list(rotation), "T": list(translation)}
    baseline = data.get("baseline_mm")
    if baseline is not None:
        result["baseline_mm"] = float(baseline)
    else:
        result["baseline_mm"] = round(float(sum(value * value for value in translation) ** 0.5), 2)
    return result


def _server_stereo(config: ServerConfig) -> dict[str, Any] | None:
    if config.stereo_R is None or config.stereo_T is None:
        return None
    translation = [float(value) for value in config.stereo_T]
    return {
        "R": [float(value) for value in config.stereo_R],
        "T": translation,
        "baseline_mm": round(float(sum(value * value for value in translation) ** 0.5), 2),
    }


def _runtime_stereo() -> dict[str, Any] | None:
    with RUNTIME_CONFIG_LOCK:
        return dict(RUNTIME_STEREO) if RUNTIME_STEREO is not None else None


def _effective_stereo(config: ServerConfig) -> dict[str, Any] | None:
    runtime = _runtime_stereo()
    if runtime is not None:
        return runtime
    return _server_stereo(config)


def _read_runtime_config_data(config: ServerConfig) -> dict[str, Any]:
    path = config.runtime_config_path
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"runtime config must be a JSON object: {path}")
    return data


def _write_runtime_config_data(config: ServerConfig, payload: dict[str, Any]) -> None:
    path = config.runtime_config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _resolve_yolo_model_path(model_path: str | Path) -> Path:
    raw_path = Path(model_path).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                REPO_ROOT / raw_path,
                Path.cwd() / raw_path,
                SCRIPT_DIR / raw_path,
            ]
        )
    candidates.append(resolve_model_path(raw_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else raw_path.resolve()


def _model_path_for_config(model_path: str | Path) -> str:
    resolved = _resolve_yolo_model_path(model_path)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except Exception:
        pass
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return str(resolved)


def _validate_yolo_model_path(model_path: Any) -> str:
    raw = str(model_path or "").strip()
    if not raw:
        raise ValueError("yolo_model is required")
    if not raw.lower().endswith(".pt"):
        raise ValueError("YOLO model must be a .pt file")
    resolved = _resolve_yolo_model_path(raw)
    if not resolved.exists():
        raise ValueError(f"YOLO model does not exist: {raw}")
    # Load once here so a broken/non-segmentation model fails before becoming default.
    get_yolo_model(resolved, task="segment")
    MODEL_CLASS_NAMES_CACHE.pop(str(resolved), None)
    _yolo_class_names(resolved)
    return _model_path_for_config(resolved)


def _available_yolo_models(config: ServerConfig) -> list[str]:
    values: list[str] = []
    for candidate in (config.yolo_model, _runtime_yolo_model(config)):
        if candidate:
            try:
                value = _model_path_for_config(candidate)
            except Exception:
                value = str(candidate)
            if value not in values:
                values.append(value)
    model_roots = [MODELS_DIR, Path.cwd() / "models", Path("models")]
    seen_roots: set[str] = set()
    for root in model_roots:
        try:
            resolved_root = root.resolve()
        except Exception:
            resolved_root = root
        root_key = str(resolved_root)
        if root_key in seen_roots or not resolved_root.exists():
            continue
        seen_roots.add(root_key)
        for path in sorted(resolved_root.glob("*.pt")):
            try:
                value = _model_path_for_config(path)
            except Exception:
                value = str(path)
            if value not in values:
                values.append(value)
    return values


def _yolo_model_config_payload(config: ServerConfig) -> dict[str, Any]:
    effective_model = _effective_yolo_model(config)
    return {
        "ok": True,
        "config_path": str(config.runtime_config_path),
        "startup_default": str(config.yolo_model),
        "runtime_default": _runtime_yolo_model(config),
        "effective_model": effective_model,
        "available_models": _available_yolo_models(config),
        "yolo_class_names": _yolo_class_names(effective_model),
    }


def _normalize_size_label(label: Any) -> str:
    return str(label or "").strip()


def _default_cigarette_name(label: str) -> str:
    return DEFAULT_CIGARETTE_NAMES.get(str(label), str(label))


def _default_cigarette_id(label: str) -> str:
    normalized = str(label)
    if normalized in DEFAULT_CIGARETTE_IDS_BY_CLASS:
        return DEFAULT_CIGARETTE_IDS_BY_CLASS[normalized]
    if normalized in DEFAULT_CIGARETTE_NAMES and normalized.isdigit() and len(normalized) >= 6:
        return normalized
    return ""


def _label_display_name(label: Any) -> str:
    normalized = _normalize_size_label(label)
    return _default_cigarette_name(normalized) if normalized else ""


def _payload_label_display_name(payload: dict[str, Any], label: Any) -> str:
    normalized = _normalize_size_label(label)
    if not normalized:
        return ""
    config_data = payload.get("object_size_config")
    if isinstance(config_data, dict):
        for key in ("object_infos", "cigarette_infos", "object_sizes", "effective_defaults"):
            values = config_data.get(key)
            if isinstance(values, dict):
                info = values.get(normalized)
                if isinstance(info, dict):
                    name = info.get("name") or info.get("display_name")
                    if name is not None and str(name).strip():
                        return str(name).strip()
    return _label_display_name(normalized)


def _object_info_name(label: str, raw: dict[str, Any], base: dict[str, Any]) -> str:
    for key in ("display_name", "cigarette_name", "name"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for key in ("display_name", "cigarette_name", "name"):
        value = base.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _default_cigarette_name(label)


def _object_info_cigarette_id(label: str, raw: dict[str, Any], base: dict[str, Any]) -> str:
    for key in ("cigarette_id", "product_id", "id"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for key in ("cigarette_id", "product_id", "id"):
        value = base.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return _default_cigarette_id(label)


def _validate_object_size(label: str, raw: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
    base_values = base or DEFAULT_OBJECT_SIZE_MM
    aliases = {
        "length_mm": ("length_mm", "long_side_mm", "long_mm", "length"),
        "width_mm": ("width_mm", "short_side_mm", "short_mm", "width"),
        "height_mm": ("height_mm", "thickness_mm", "height", "thickness"),
    }
    result: dict[str, Any] = {}
    for target, names in aliases.items():
        value = None
        for name in names:
            if name in raw:
                value = raw[name]
                break
        if value is None:
            value = base_values[target]
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{label} {target} must be a positive number")
        result[target] = round(number, 3)
    result["cigarette_id"] = _object_info_cigarette_id(label, raw, base_values)
    result["name"] = _object_info_name(label, raw, base_values)
    return result


def _validate_object_sizes(raw: Any, base: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, list):
        values: dict[str, Any] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("object_sizes list items must be JSON objects")
            label = _normalize_size_label(
                item.get("label") or item.get("class_name") or item.get("id") or item.get("yolo_label")
            )
            if not label:
                raise ValueError("object_sizes list item missing label")
            values[label] = item
    elif isinstance(raw, dict):
        values = raw
    else:
        raise ValueError("object_sizes must be a JSON object or list")

    base_values = base or {}
    sizes: dict[str, dict[str, Any]] = {}
    for label_raw, size_raw in values.items():
        label = _normalize_size_label(label_raw)
        if not label:
            continue
        if not isinstance(size_raw, dict):
            raise ValueError(f"object_sizes[{label}] must be a JSON object")
        sizes[label] = _validate_object_size(label, size_raw, base=base_values.get(label, DEFAULT_OBJECT_SIZE_MM))
    return sizes


def _startup_object_sizes(config: ServerConfig) -> dict[str, dict[str, Any]]:
    labels: list[str] = []
    for label in _yolo_class_names(_effective_yolo_model(config)):
        if label not in labels:
            labels.append(label)
    for label in BUILTIN_OBJECT_SIZES_MM:
        if label not in labels:
            labels.append(label)

    sizes: dict[str, dict[str, Any]] = {}
    for label in labels:
        sizes[label] = dict(BUILTIN_OBJECT_SIZES_MM.get(label, DEFAULT_OBJECT_SIZE_MM))
        sizes[label]["cigarette_id"] = _default_cigarette_id(label)
        sizes[label]["name"] = _default_cigarette_name(label)
    return sizes


def _effective_object_sizes(config: ServerConfig) -> dict[str, dict[str, Any]]:
    sizes = _startup_object_sizes(config)
    runtime = _runtime_object_sizes()
    if runtime:
        for label, size in runtime.items():
            sizes[label] = dict(size)
    return sizes


def _apply_object_sizes_to_pnp(config: ServerConfig) -> None:
    top_sizes: dict[str, tuple[float, float]] = {
        label: (size["length_mm"] / 1000.0, size["width_mm"] / 1000.0)
        for label, size in _effective_object_sizes(config).items()
    }
    YOLO_CLASS_TOP_SIZES_M.clear()
    YOLO_CLASS_TOP_SIZES_M.update(top_sizes)


def _object_size_config_payload(config: ServerConfig) -> dict[str, Any]:
    object_infos = _effective_object_sizes(config)
    return {
        "ok": True,
        "config_path": str(config.runtime_config_path),
        "startup_defaults": _startup_object_sizes(config),
        "runtime_defaults": _runtime_object_sizes(),
        "effective_defaults": object_infos,
        "object_sizes": object_infos,
        "object_infos": object_infos,
        "cigarette_infos": object_infos,
        "note": "name is used for display; length_mm/width_mm are used by YOLO class-aware PnP; height_mm is used by the G1-D 3D visualizer thickness",
    }


def _runtime_config_payload(config: ServerConfig) -> dict[str, Any]:
    return {
        "ok": True,
        "config_path": str(config.runtime_config_path),
        "startup_defaults": _server_intrinsics(config),
        "runtime_defaults": _runtime_intrinsics(),
        "effective_defaults": _effective_intrinsics(config),
        "right_startup_defaults": _server_intrinsics_right(config),
        "right_runtime_defaults": _runtime_intrinsics_right(),
        "right_effective_defaults": _effective_intrinsics_right(config),
        "stereo_effective": _effective_stereo(config),
        "stereo_available": _effective_stereo(config) is not None,
        "yolo_model": _yolo_model_config_payload(config),
        "object_sizes": _object_size_config_payload(config),
        "note": "pose/xyz/debug requests can still override these values with fx/fy/cx/cy query or JSON fields",
    }


def _load_runtime_intrinsics(config: ServerConfig) -> None:
    data = _read_runtime_config_data(config)
    if not data:
        _apply_object_sizes_to_pnp(config)
        return
    raw_yolo_model = data.get("yolo_model")
    yolo_model_value = None
    if raw_yolo_model is not None:
        yolo_model_value = _validate_yolo_model_path(raw_yolo_model)
        with RUNTIME_CONFIG_LOCK:
            global RUNTIME_YOLO_MODEL
            RUNTIME_YOLO_MODEL = yolo_model_value

    raw_intrinsics = data.get("intrinsics")
    if raw_intrinsics is None and any(name in data for name in ("focal_px", "fx", "fy", "cx", "cy", "dist_coeffs")):
        raw_intrinsics = data
    values = None
    if raw_intrinsics is not None:
        if not isinstance(raw_intrinsics, dict):
            raise ValueError(f"runtime intrinsics must be a JSON object: {config.runtime_config_path}")
        values = _validate_intrinsics(raw_intrinsics, base=_server_intrinsics(config))

    raw_intrinsics_right = data.get("intrinsics_right")
    values_right = None
    if raw_intrinsics_right is not None:
        if not isinstance(raw_intrinsics_right, dict):
            raise ValueError(f"runtime intrinsics_right must be a JSON object: {config.runtime_config_path}")
        values_right = _validate_intrinsics_right(raw_intrinsics_right, base=_server_intrinsics_right(config))

    raw_stereo = data.get("stereo")
    stereo_values = None
    if raw_stereo is not None:
        stereo_values = _validate_stereo(raw_stereo)

    raw_sizes = data.get("object_sizes")
    object_sizes = _validate_object_sizes(raw_sizes, base=_startup_object_sizes(config)) if raw_sizes is not None else None
    with RUNTIME_CONFIG_LOCK:
        global RUNTIME_INTRINSICS, RUNTIME_INTRINSICS_RIGHT, RUNTIME_STEREO, RUNTIME_OBJECT_SIZES_MM
        if values is not None:
            RUNTIME_INTRINSICS = values
        if values_right is not None:
            RUNTIME_INTRINSICS_RIGHT = values_right
        if stereo_values is not None:
            RUNTIME_STEREO = stereo_values
        if object_sizes is not None:
            RUNTIME_OBJECT_SIZES_MM = object_sizes
    _apply_object_sizes_to_pnp(config)


def _save_runtime_intrinsics(config: ServerConfig, intrinsics: dict[str, float]) -> None:
    payload = _read_runtime_config_data(config)
    payload["intrinsics"] = intrinsics
    with RUNTIME_CONFIG_LOCK:
        global RUNTIME_INTRINSICS
        RUNTIME_INTRINSICS = dict(intrinsics)
    _write_runtime_config_data(config, payload)


def _save_runtime_intrinsics_right(config: ServerConfig, intrinsics_right: dict[str, Any]) -> None:
    payload = _read_runtime_config_data(config)
    payload["intrinsics_right"] = intrinsics_right
    with RUNTIME_CONFIG_LOCK:
        global RUNTIME_INTRINSICS_RIGHT
        RUNTIME_INTRINSICS_RIGHT = dict(intrinsics_right)
    _write_runtime_config_data(config, payload)


def _save_runtime_stereo(config: ServerConfig, stereo: dict[str, Any]) -> None:
    payload = _read_runtime_config_data(config)
    payload["stereo"] = stereo
    with RUNTIME_CONFIG_LOCK:
        global RUNTIME_STEREO
        RUNTIME_STEREO = dict(stereo)
    _write_runtime_config_data(config, payload)


def _save_runtime_yolo_model(config: ServerConfig, yolo_model: str) -> None:
    model_value = _validate_yolo_model_path(yolo_model)
    payload = _read_runtime_config_data(config)
    payload["yolo_model"] = model_value
    with RUNTIME_CONFIG_LOCK:
        global RUNTIME_YOLO_MODEL
        RUNTIME_YOLO_MODEL = model_value
    _write_runtime_config_data(config, payload)
    _apply_object_sizes_to_pnp(config)


def _save_runtime_object_sizes(config: ServerConfig, object_sizes: dict[str, dict[str, Any]]) -> None:
    payload = _read_runtime_config_data(config)
    payload["object_sizes"] = object_sizes
    with RUNTIME_CONFIG_LOCK:
        global RUNTIME_OBJECT_SIZES_MM
        RUNTIME_OBJECT_SIZES_MM = {label: dict(size) for label, size in object_sizes.items()}
    _write_runtime_config_data(config, payload)
    _apply_object_sizes_to_pnp(config)


def _serve_intrinsics_config(handler: BaseHTTPRequestHandler, config: ServerConfig) -> None:
    if handler.command == "GET":
        _json_response(handler, 200, _runtime_config_payload(config))
        return
    overrides = _request_overrides(handler)
    names = {"focal_px", "fx", "fy", "cx", "cy", "dist_coeffs"}
    values = {name: overrides[name] for name in names if name in overrides}
    if not values:
        _json_response(
            handler,
            400,
            {"ok": False, "error": "provide at least one of focal_px/fx/fy/cx/cy/dist_coeffs"},
        )
        return
    intrinsics = _validate_intrinsics(values, base=_effective_intrinsics(config))
    _save_runtime_intrinsics(config, intrinsics)
    _json_response(handler, 200, _runtime_config_payload(config))


def _serve_intrinsics_right_config(handler: BaseHTTPRequestHandler, config: ServerConfig) -> None:
    if handler.command == "GET":
        _json_response(handler, 200, _runtime_config_payload(config))
        return
    overrides = _request_overrides(handler)
    mapping = {
        "fx_right": "fx",
        "fy_right": "fy",
        "cx_right": "cx",
        "cy_right": "cy",
        "dist_coeffs_right": "dist_coeffs",
    }
    values = {dst: overrides[src] for src, dst in mapping.items() if src in overrides}
    if not values:
        _json_response(
            handler,
            400,
            {"ok": False, "error": "provide at least one of fx_right/fy_right/cx_right/cy_right/dist_coeffs_right"},
        )
        return
    intrinsics_right = _validate_intrinsics_right(values, base=_effective_intrinsics_right(config))
    _save_runtime_intrinsics_right(config, intrinsics_right)
    _json_response(handler, 200, _runtime_config_payload(config))


def _serve_stereo_config(handler: BaseHTTPRequestHandler, config: ServerConfig) -> None:
    if handler.command == "GET":
        _json_response(handler, 200, _runtime_config_payload(config))
        return
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        _json_response(handler, 400, {"ok": False, "error": "request JSON body with R and T is required"})
        return
    data = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("request JSON body must be an object")
    stereo = _validate_stereo(data.get("stereo", data))
    _save_runtime_stereo(config, stereo)
    _json_response(handler, 200, _runtime_config_payload(config))


def _uploaded_yolo_model_path(handler: BaseHTTPRequestHandler) -> str:
    form = cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": handler.headers.get("Content-Length", ""),
        },
    )
    field = form["model_file"] if "model_file" in form else None
    if field is None or not getattr(field, "filename", ""):
        raise ValueError("multipart form requires model_file")
    filename = Path(str(field.filename)).name
    if not filename.lower().endswith(".pt"):
        raise ValueError("uploaded model must be a .pt file")
    target_dir = MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    tmp_path = target_dir / f".{target_path.stem}.upload_{int(time.time())}.pt"
    with tmp_path.open("wb") as f:
        while True:
            chunk = field.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    try:
        _validate_yolo_model_path(tmp_path)
        tmp_path.replace(target_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return _model_path_for_config(target_path)


def _serve_yolo_model_config(handler: BaseHTTPRequestHandler, config: ServerConfig) -> None:
    if handler.command == "GET":
        _json_response(handler, 200, _yolo_model_config_payload(config))
        return

    content_type = handler.headers.get("Content-Type", "")
    if content_type.lower().startswith("multipart/form-data"):
        model_path = _uploaded_yolo_model_path(handler)
    else:
        length = int(handler.headers.get("Content-Length") or 0)
        if length <= 0:
            _json_response(handler, 400, {"ok": False, "error": "request JSON body is required"})
            return
        data = json.loads(handler.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request JSON body must be an object")
        model_path = data.get("yolo_model") or data.get("model") or data.get("model_path")

    _save_runtime_yolo_model(config, str(model_path))
    _json_response(handler, 200, _yolo_model_config_payload(config))


def _serve_object_sizes_config(handler: BaseHTTPRequestHandler, config: ServerConfig) -> None:
    if handler.command == "GET":
        _json_response(handler, 200, _object_size_config_payload(config))
        return
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        _json_response(handler, 400, {"ok": False, "error": "request JSON body is required"})
        return
    data = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("request JSON body must be an object")
    raw_sizes = data.get("object_infos") or data.get("cigarette_infos") or data.get("object_sizes") or data
    object_sizes = _validate_object_sizes(raw_sizes, base=_effective_object_sizes(config))
    if not object_sizes:
        _json_response(handler, 400, {"ok": False, "error": "provide at least one object size"})
        return
    _save_runtime_object_sizes(config, object_sizes)
    _json_response(handler, 200, _object_size_config_payload(config))


def _coerce_g1d_adjust_value(name: str, value: Any) -> Any:
    target_type = G1D_ADJUST_PROXY_TYPES[name]
    if target_type is bool:
        return _parse_bool(value)
    return target_type(value)


def _g1d_adjust_values(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    parsed = urlparse(handler.path)
    values: dict[str, Any] = {}
    for key, raw_values in parse_qs(parsed.query).items():
        normalized = key.replace("-", "_")
        if normalized in G1D_ADJUST_PROXY_TYPES and raw_values:
            values[normalized] = _coerce_g1d_adjust_value(normalized, raw_values[-1])

    length = int(handler.headers.get("Content-Length") or 0)
    if length > 0:
        body = handler.rfile.read(length)
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request JSON body must be an object")
        for key, value in data.items():
            normalized = key.replace("-", "_")
            if normalized in G1D_ADJUST_PROXY_TYPES:
                values[normalized] = _coerce_g1d_adjust_value(normalized, value)
    return values


def _serve_g1d_adjust(handler: BaseHTTPRequestHandler, request_url_base: str = G1D_ADJUST_URL) -> None:
    values = _g1d_adjust_values(handler)
    query = urlencode({key: value for key, value in values.items() if value is not None})
    request_url = request_url_base if not query else f"{request_url_base}?{query}"
    try:
        with urllib.request.urlopen(request_url, timeout=G1D_ADJUST_TIMEOUT_SEC) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            status = 200 if bool(payload.get("ok")) else 500
            payload.setdefault("proxy", {})
            if isinstance(payload["proxy"], dict):
                payload["proxy"].update({"ok": True, "url": request_url})
            _json_response(handler, status, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"ok": False, "error": body or str(exc)}
        payload.setdefault("proxy", {})
        if isinstance(payload["proxy"], dict):
            payload["proxy"].update({"ok": False, "url": request_url, "http_status": exc.code})
        _json_response(handler, exc.code, payload)
    except Exception as exc:
        _json_response(
            handler,
            502,
            {
                "ok": False,
                "error": f"failed to call G1D adjust service: {exc}",
                "proxy": {"ok": False, "url": request_url},
            },
        )


def _append_arg(argv: list[str], name: str, value: Any) -> None:
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    argv.extend([flag, str(value)])


def _ensure_logging_mp_compat() -> None:
    try:
        import logging_mp
    except Exception:
        return
    if not hasattr(logging_mp, "getLogger") and hasattr(logging_mp, "get_logger"):
        logging_mp.getLogger = logging_mp.get_logger


def _get_image_client(host: str) -> Any:
    if host not in IMAGE_CLIENTS:
        _ensure_logging_mp_compat()
        from teleimager.image_client import ImageClient

        IMAGE_CLIENTS[host] = ImageClient(host=host)
    return IMAGE_CLIENTS[host]


def _yolo_class_names(model_path: str | Path) -> list[str]:
    resolved_model = _resolve_yolo_model_path(model_path)
    cache_key = str(resolved_model)
    if cache_key in MODEL_CLASS_NAMES_CACHE:
        return list(MODEL_CLASS_NAMES_CACHE[cache_key])
    try:
        model = get_yolo_model(resolved_model, task="segment")
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            def sort_key(item: Any) -> tuple[int, str]:
                try:
                    return (int(item), str(item))
                except Exception:
                    return (10_000, str(item))

            class_names = [str(names[key]) for key in sorted(names, key=sort_key)]
        elif isinstance(names, (list, tuple)):
            class_names = [str(item) for item in names]
        else:
            class_names = []
    except Exception:
        class_names = []
    class_names = [name for name in class_names if name]
    MODEL_CLASS_NAMES_CACHE[cache_key] = class_names
    return list(class_names)


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
    intrinsics = _request_intrinsics(config, overrides)
    intrinsics_right = _request_intrinsics_right(config, overrides)
    argv = [
        "--mode",
        "yolo",
        "--left-image",
        str(left_image),
        "--right-image",
        str(right_image),
        "--yolo-model",
        _effective_yolo_model(config),
        "--yolo-device",
        config.yolo_device,
        "--yolo-conf",
        str(config.yolo_conf),
        "--yolo-imgsz",
        str(config.yolo_imgsz),
        "--yolo-mask-threshold",
        str(config.yolo_mask_threshold),
        "--focal-px",
        str(intrinsics["focal_px"]),
        "--fx",
        str(intrinsics["fx"]),
        "--fy",
        str(intrinsics["fy"]),
        "--cx",
        str(intrinsics["cx"]),
        "--cy",
        str(intrinsics["cy"]),
        # Use --opt=value so comma lists that may start with '-' are not parsed as flags.
        "--dist-coeffs=" + ",".join(str(value) for value in intrinsics["dist_coeffs"]),
        "--fx-right",
        str(intrinsics_right["fx"]),
        "--fy-right",
        str(intrinsics_right["fy"]),
        "--cx-right",
        str(intrinsics_right["cx"]),
        "--cy-right",
        str(intrinsics_right["cy"]),
        "--dist-coeffs-right=" + ",".join(str(value) for value in intrinsics_right["dist_coeffs"]),
        "--out-dir",
        str(out_dir),
    ]
    stereo = _effective_stereo(config)
    if stereo is not None:
        argv.append("--stereo-r=" + ",".join(str(value) for value in stereo["R"]))
        argv.append("--stereo-t=" + ",".join(str(value) for value in stereo["T"]))
    skip = {
        "focal_px", "fx", "fy", "cx", "cy", "dist_coeffs",
        "fx_right", "fy_right", "cx_right", "cy_right", "dist_coeffs_right",
        "yolo_model", "model", "model_path",
    }
    for name, value in overrides.items():
        if name in skip:
            continue
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


def _selected_yolo_candidate(result: dict[str, Any]) -> dict[str, Any]:
    point_adjustments = result.get("point_adjustments")
    if not isinstance(point_adjustments, dict):
        return {}
    left_yolo = point_adjustments.get("left_yolo")
    return left_yolo if isinstance(left_yolo, dict) else {}


def _object_box_size_for_result(config: ServerConfig, result: dict[str, Any]) -> dict[str, Any]:
    label = _normalize_size_label(result.get("selected_yolo_label"))
    top_source = result.get("object_top_size_source")
    if isinstance(top_source, dict) and not label:
        label = _normalize_size_label(top_source.get("class_name"))

    sizes = _effective_object_sizes(config)
    if label and label in sizes:
        size = dict(sizes[label])
        source = "runtime_or_configured_label_size"
    elif isinstance(top_source, dict) and top_source.get("long_side_mm") and top_source.get("short_side_mm"):
        size = {
            "length_mm": float(top_source["long_side_mm"]),
            "width_mm": float(top_source["short_side_mm"]),
            "height_mm": DEFAULT_OBJECT_SIZE_MM["height_mm"],
        }
        source = str(top_source.get("source") or "object_top_size_source")
    else:
        size = dict(DEFAULT_OBJECT_SIZE_MM)
        source = "default_object_size"

    return {
        "label": label or None,
        "cigarette_id": str(size.get("cigarette_id") or _default_cigarette_id(label)) if label else None,
        "name": str(size.get("name") or _default_cigarette_name(label)) if label else None,
        "length_mm": round(float(size["length_mm"]), 3),
        "width_mm": round(float(size["width_mm"]), 3),
        "height_mm": round(float(size["height_mm"]), 3),
        "source": source,
    }


def _enrich_result_object_size(config: ServerConfig, result: dict[str, Any]) -> None:
    box_size = _object_box_size_for_result(config, result)
    result["object_box_size_mm"] = box_size
    top_source = result.get("object_top_size_source")
    if isinstance(top_source, dict):
        top_source["height_mm"] = box_size["height_mm"]
        top_source["box_size_mm"] = [box_size["length_mm"], box_size["width_mm"], box_size["height_mm"]]
        top_source["known_class_box_sizes_mm"] = {
            label: [size["length_mm"], size["width_mm"], size["height_mm"]]
            for label, size in _effective_object_sizes(config).items()
        }


def _g1d_visualization_data(result: dict[str, Any]) -> dict[str, Any]:
    alignment = result.get("robot_alignment")
    if not isinstance(alignment, dict):
        alignment = {}
    target = alignment.get("target")
    if not isinstance(target, dict):
        target = {}
    control = alignment.get("control_hint")
    if not isinstance(control, dict):
        control = {}
    box_axis = alignment.get("box_axis")
    if not isinstance(box_axis, dict):
        box_axis = {}
    near_alignment = result.get("near_edge_robot_alignment")
    if not isinstance(near_alignment, dict):
        near_alignment = {}
    near_target = near_alignment.get("target")
    if not isinstance(near_target, dict):
        near_target = {}
    selected_yolo = _selected_yolo_candidate(result)
    server = result.get("server")
    if not isinstance(server, dict):
        server = {}
    object_size = result.get("object_box_size_mm")
    if not isinstance(object_size, dict):
        object_size = {}

    return {
        "schema_version": 1,
        "source": "yolo_xyz_plus_external_joint_states",
        "ok": bool(result.get("ok")),
        "yolo": {
            "label": result.get("selected_yolo_label"),
            "display_name": object_size.get("name") or _label_display_name(result.get("selected_yolo_label")),
            "class_id": result.get("selected_yolo_class_id"),
            "confidence": result.get("selected_yolo_confidence"),
            "candidate_index": selected_yolo.get("selected_candidate_index"),
            "raw_yolo_index": selected_yolo.get("raw_yolo_index"),
            "points_px": selected_yolo.get("points_px"),
            "box_xyxy": selected_yolo.get("box_xyxy"),
        },
        "box": {
            "center_xyz_mm": result.get("center_xyz_mm"),
            "near_edge_midpoint_xyz_mm": result.get("near_edge_midpoint_xyz_mm"),
            "head_point_xyz_mm": result.get("box_head_point_xyz_mm"),
            "head_to_tail_unit_xyz": _nested(result, "box_head_point", "head_to_tail_unit_xyz"),
            "object_top_size_mm": result.get("object_top_size_mm"),
            "object_box_size_mm": result.get("object_box_size_mm"),
            "selected_orientation": result.get("selected_orientation"),
            "range_from_left_camera_mm": result.get("range_from_left_camera_mm"),
        },
        "metrics": {
            "turn_to_target_yaw_deg": control.get("turn_first_yaw_deg"),
            "turn_to_target_yaw_rad": control.get("turn_first_yaw_rad"),
            "box_long_axis_yaw_deg": control.get("box_parallel_yaw_deg"),
            "box_long_axis_yaw_rad": control.get("box_parallel_yaw_rad"),
            "box_axis_yaw_mod180_deg": box_axis.get("axis_yaw_mod180_deg"),
            "center_vertical_down_mm": target.get("vertical_down_mm"),
            "center_ground_forward_mm": target.get("ground_forward_mm"),
            "near_edge_ground_forward_mm": near_target.get("ground_forward_mm"),
            "right_mm": target.get("right_mm"),
            "ground_distance_mm": target.get("ground_distance_mm"),
        },
        "camera": {
            "frame": "left_camera_optical",
            "mount_parent_link": "torso_link",
            "camera_to_vertical_deg": alignment.get("camera_to_vertical_deg")
            or result.get("top_plane_camera_to_vertical_deg"),
            "basis": alignment.get("basis"),
        },
        "robot_state": {
            "source": "external_g1d_joint_states",
            "provided_by_yolo": False,
            "required_for_visual_model": True,
            "accepted_shapes": [
                {"joints": {"joint_name": "value_in_urdf_units"}},
                {"joint_states": {"name": ["joint_name"], "position": ["value_in_urdf_units"]}},
            ],
            "units": "prismatic=m, revolute_or_continuous=rad",
        },
        "server": {
            "request_id": server.get("request_id"),
            "elapsed_ms": server.get("elapsed_ms"),
        },
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
        "intrinsics_assumption": result.get("intrinsics_assumption"),
        "intrinsics_assumption_right": result.get("intrinsics_assumption_right"),
        "left": result.get("left"),
        "right": result.get("right"),
        "stereo": result.get("stereo"),
        "stereo_plane": result.get("stereo_plane"),
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
        "object_box_size_mm": result.get("object_box_size_mm"),
        "near_edge_midpoint_xyz_mm": result.get("near_edge_midpoint_xyz_mm"),
        "near_edge_midpoint": result.get("near_edge_midpoint"),
        "box_near_edge_midpoint_xyz_mm": result.get("box_near_edge_midpoint_xyz_mm"),
        "box_near_edge_midpoint": result.get("box_near_edge_midpoint"),
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
        "near_edge_robot_alignment": result.get("near_edge_robot_alignment"),
        "robot_alignment_hypotheses": result.get("robot_alignment_hypotheses"),
        "g1d_visualization": result.get("g1d_visualization") or _g1d_visualization_data(result),
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
        _apply_object_sizes_to_pnp(config)
        args = _build_pose_args(config, out_dir, left_path, right_path, overrides)
        result, exit_code = run_pose(args)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    requested_label = overrides.get("label") or overrides.get("yolo_label") or overrides.get("yolo_class_name")
    if requested_label and not result.get("requested_yolo_label"):
        result["requested_yolo_label"] = str(requested_label)
    if not isinstance(result.get("intrinsics_assumption"), dict):
        result["intrinsics_assumption"] = _effective_intrinsics(config)
    if not isinstance(result.get("intrinsics_assumption_right"), dict):
        result["intrinsics_assumption_right"] = _effective_intrinsics_right(config)
    effective_model = _effective_yolo_model(config)
    server_info = {
        "resident": True,
        "pid": os.getpid(),
        "request_id": request_id,
        "elapsed_ms": elapsed_ms,
        "model_cache_size": len(YOLO_MODEL_CACHE),
        "yolo_model": effective_model,
        "yolo_class_names": _yolo_class_names(effective_model),
    }
    result["server"] = server_info
    _enrich_result_object_size(config, result)
    result["object_size_config"] = _object_size_config_payload(config)
    result["yolo_model_config"] = _yolo_model_config_payload(config)
    result["g1d_visualization"] = _g1d_visualization_data(result)
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
        fx = float(intrinsics.get("fx", focal_px))
        fy = float(intrinsics.get("fy", focal_px))
        cx = float(intrinsics.get("cx"))
        cy = float(intrinsics.get("cy"))
    except Exception:
        return None
    u = fx * x_mm / z_mm + cx
    v = fy * y_mm / z_mm + cy
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
        ("N", "near_edge_midpoint_xyz_mm", (0, 255, 255), 42),
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
        _draw_text(image, "C=center_above  H=head_1/5_above  N=near_edge_midpoint", (12, 28), (255, 255, 255), scale=0.55)
        cv2.putText(
            image,
            "Projected from /xyz using u=fx*x/z+cx, v=fy*y/z+cy",
            (12, image.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Projected from /xyz using u=fx*x/z+cx, v=fy*y/z+cy",
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
        class_name = str(item.get("class_name") or "")
        display_name = _label_display_name(class_name) or class_name
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(item.get('candidate_index')))}</td>"
            f"<td>{html_lib.escape(str(item.get('raw_yolo_index')))}</td>"
            f"<td>{html_lib.escape(display_name)}</td>"
            f"<td>{html_lib.escape(str(item.get('confidence')))}</td>"
            f"<td>{html_lib.escape(str(item.get('score')))}</td>"
            f"<td>{html_lib.escape('是' if item.get('matches_label_filter', True) else '否')}</td>"
            f"<td>{html_lib.escape(str(item.get('mask_area_px')))}</td>"
            f"<td>{html_lib.escape(str(item.get('box_xyxy')))}</td>"
            f"<td>{html_lib.escape(str(item.get('points_px')))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>候选序号</th><th>YOLO 原序号</th><th>类别</th><th>置信度</th><th>评分</th>"
        "<th>参与当前类别</th><th>mask面积</th><th>检测框</th><th>四点像素</th></tr></thead>"
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
    near_alignment = payload.get("near_edge_robot_alignment")
    if not isinstance(near_alignment, dict):
        near_alignment = {}
    near_target = near_alignment.get("target")
    if not isinstance(near_target, dict):
        near_target = {}

    raw_label = payload.get("selected_yolo_label")
    box_size = payload.get("object_box_size_mm")
    if not isinstance(box_size, dict):
        box_size = {}
    label = box_size.get("name") or _label_display_name(raw_label) or "-"
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
        _summary_tile("近端边中点", _fmt_list(payload.get("near_edge_midpoint_xyz_mm")), "靠近机器人那条边的中点"),
        _summary_tile("近端边前向", _fmt_number(near_target.get("ground_forward_mm"), " mm", 1), "微调目标建议 200mm"),
        _summary_tile("直线距离", _fmt_number(target.get("range_from_left_camera_mm"), " mm", 1), "左目光心到中心点"),
        _summary_tile("地面前向", _fmt_number(target.get("ground_forward_mm"), " mm", 1), "Z*sin角度 - Y*cos角度"),
        _summary_tile("中心垂直距离", _fmt_number(target.get("vertical_down_mm"), " mm", 1), "沿地面垂直方向到上表面中心"),
        _summary_tile("地面平面距离", _fmt_number(target.get("ground_distance_mm"), " mm", 1), "sqrt(前向² + 左右²)"),
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


def _debug_intrinsics_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    intrinsics = payload.get("intrinsics_assumption")
    if not isinstance(intrinsics, dict):
        intrinsics = {}

    def read_float(name: str, fallback: float) -> float:
        try:
            return float(intrinsics.get(name, fallback))
        except Exception:
            return float(fallback)

    focal = read_float("focal_px", CALIBRATED_LEFT_FOCAL_PX)
    raw_dist = intrinsics.get("dist_coeffs")
    dist_coeffs = list(CALIBRATED_LEFT_DIST_COEFFS)
    if isinstance(raw_dist, (list, tuple)):
        for idx in range(min(5, len(raw_dist))):
            try:
                dist_coeffs[idx] = float(raw_dist[idx])
            except Exception:
                dist_coeffs[idx] = 0.0
    return {
        "fx": read_float("fx", focal),
        "fy": read_float("fy", focal),
        "cx": read_float("cx", CALIBRATED_LEFT_CX),
        "cy": read_float("cy", CALIBRATED_LEFT_CY),
        "dist_coeffs": dist_coeffs,
    }


def _debug_intrinsics_query(payload: dict[str, Any]) -> str:
    intrinsics = _debug_intrinsics_from_payload(payload)
    scalar = {name: f"{value:.6g}" for name, value in intrinsics.items() if name != "dist_coeffs"}
    dist = intrinsics.get("dist_coeffs") or []
    scalar["dist_coeffs"] = ",".join(f"{float(value):.6g}" for value in dist)
    return urlencode(scalar) + "&"


def _debug_intrinsics_controls_html(payload: dict[str, Any]) -> str:
    intrinsics = _debug_intrinsics_from_payload(payload)
    fields = [
        ("fx", "debugFxInput", "fx"),
        ("fy", "debugFyInput", "fy"),
        ("cx", "debugCxInput", "cx"),
        ("cy", "debugCyInput", "cy"),
    ]
    inputs = "".join(
        (
            f'<label>{label}'
            f'<input id="{html_lib.escape(input_id)}" data-intrinsic="{html_lib.escape(name)}" '
            f'type="number" step="0.01" value="{intrinsics[name]:.2f}">'
            "</label>"
        )
        for name, input_id, label in fields
    )
    dist = intrinsics.get("dist_coeffs") or [0.0, 0.0, 0.0, 0.0, 0.0]
    dist_labels = ("k1", "k2", "p1", "p2", "k3")
    dist_inputs = "".join(
        (
            f'<label>{dist_labels[idx]}'
            f'<input id="debugDist{idx}Input" data-dist-index="{idx}" '
            f'type="number" step="0.000001" value="{float(dist[idx]):.6f}">'
            "</label>"
        )
        for idx in range(5)
    )

    right_raw = payload.get("intrinsics_assumption_right")
    if not isinstance(right_raw, dict):
        right_raw = {}

    def _read_right(name: str, fallback: float) -> float:
        try:
            return float(right_raw.get(name, fallback))
        except Exception:
            return float(fallback)

    right_vals = {
        "fx": _read_right("fx", intrinsics["fx"]),
        "fy": _read_right("fy", intrinsics["fy"]),
        "cx": _read_right("cx", intrinsics["cx"]),
        "cy": _read_right("cy", intrinsics["cy"]),
    }
    right_dist_raw = right_raw.get("dist_coeffs")
    right_dist = list(dist)
    if isinstance(right_dist_raw, (list, tuple)):
        right_dist = [0.0, 0.0, 0.0, 0.0, 0.0]
        for idx in range(min(5, len(right_dist_raw))):
            try:
                right_dist[idx] = float(right_dist_raw[idx])
            except Exception:
                right_dist[idx] = 0.0
    right_scalar_inputs = "".join(
        (
            f'<label>{name}'
            f'<input id="debugRight{name}Input" data-intrinsic-right="{name}" '
            f'type="number" step="0.01" value="{right_vals[name]:.2f}">'
            "</label>"
        )
        for name in ("fx", "fy", "cx", "cy")
    )
    right_dist_inputs = "".join(
        (
            f'<label>{dist_labels[idx]}'
            f'<input id="debugRightDist{idx}Input" data-dist-right-index="{idx}" '
            f'type="number" step="0.000001" value="{float(right_dist[idx]):.6f}">'
            "</label>"
        )
        for idx in range(5)
    )
    mirrors_left = bool(right_raw.get("mirrors_left", True))
    right_section = (
        '<details class="debug-intrinsics-advanced">'
        f'<summary>右眼相机内参{"（当前与左眼相同）" if mirrors_left else "（独立标定）"}</summary>'
        '<div class="debug-intrinsics-fields">'
        f"{right_scalar_inputs}"
        f"{right_dist_inputs}"
        '<button id="debugSaveRightIntrinsics" class="button primary-alt-button" type="button">保存右眼内参</button>'
        '<span id="debugRightIntrinsicsStatus"></span>'
        '<span>右眼用于独立输出 right_camera_optical 坐标；留空字段会沿用左眼。</span>'
        "</div>"
        "</details>"
    )
    current = (
        f"当前 API 默认 / 本次计算："
        f"fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
        f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}, "
        f"dist=[{', '.join(f'{float(value):.5f}' for value in dist)}]"
    )
    return (
        '<div class="debug-intrinsics-panel">'
        '<div class="debug-intrinsics-main">'
        '<strong>相机内参</strong>'
        f'<span id="debugIntrinsicsCurrent">{html_lib.escape(current)}</span>'
        '<button id="debugSaveCalibratedIntrinsics" class="button primary-button" type="button">一键使用标定内参</button>'
        '<span id="debugIntrinsicsSaveStatus"></span>'
        "</div>"
        '<details class="debug-intrinsics-advanced">'
        '<summary>高级：手动修改标定内参</summary>'
        '<div class="debug-intrinsics-fields">'
        f"{inputs}"
        f"{dist_inputs}"
        '<button id="debugSaveDefaultIntrinsics" class="button primary-alt-button" type="button">保存手动值为 API 默认</button>'
        '<span>高级设置保存后，后续 /xyz、/pose 不带参数也会使用这里的默认值。</span>'
        "</div>"
        "</details>"
        f"{right_section}"
        "</div>"
    )


def _debug_yolo_model_controls_html(payload: dict[str, Any]) -> str:
    config_data = payload.get("yolo_model_config")
    if not isinstance(config_data, dict):
        config_data = {}
    server = payload.get("server")
    if not isinstance(server, dict):
        server = {}
    effective_model = str(config_data.get("effective_model") or server.get("yolo_model") or "")
    available = config_data.get("available_models")
    if not isinstance(available, list):
        available = [effective_model] if effective_model else []
    class_names = config_data.get("yolo_class_names") or server.get("yolo_class_names")
    class_count = len(class_names) if isinstance(class_names, list) else 0

    options: list[str] = []
    seen: set[str] = set()
    for item in available:
        value = str(item)
        if not value or value in seen:
            continue
        seen.add(value)
        selected = " selected" if value == effective_model else ""
        options.append(
            f'<option value="{html_lib.escape(value, quote=True)}"{selected}>{html_lib.escape(Path(value).name)}</option>'
        )

    return f"""
  <section class="debug-yolo-model-panel">
    <div class="debug-yolo-model-main">
      <strong>YOLO 模型</strong>
      <span>当前：{html_lib.escape(effective_model or "-")}，类别 {class_count} 个</span>
    </div>
    <div class="debug-yolo-model-controls">
      <select id="debugYoloModelSelect">{''.join(options)}</select>
      <input id="debugYoloModelPath" type="text" value="{html_lib.escape(effective_model, quote=True)}" placeholder="models/xxx.pt">
      <button id="debugSaveYoloModel" class="button primary-alt-button" type="button">切换模型</button>
      <input id="debugYoloModelFile" type="file" accept=".pt">
      <button id="debugUploadYoloModel" class="button primary-button" type="button">上传并切换</button>
      <span id="debugYoloModelStatus"></span>
    </div>
  </section>
"""


def _debug_object_size_controls_html(payload: dict[str, Any]) -> str:
    config_data = payload.get("object_size_config")
    if not isinstance(config_data, dict):
        config_data = {}
    sizes = (
        config_data.get("object_infos")
        or config_data.get("cigarette_infos")
        or config_data.get("object_sizes")
        or config_data.get("effective_defaults")
    )
    if not isinstance(sizes, dict) or not sizes:
        sizes = {"XiongMao": BUILTIN_OBJECT_SIZES_MM.get("XiongMao", DEFAULT_OBJECT_SIZE_MM)}

    selected_label = str(payload.get("selected_yolo_label") or "")
    rows: list[str] = []
    for label in sorted(sizes):
        raw_size = sizes.get(label)
        if not isinstance(raw_size, dict):
            continue
        try:
            length_mm = float(raw_size.get("length_mm", DEFAULT_OBJECT_SIZE_MM["length_mm"]))
            width_mm = float(raw_size.get("width_mm", DEFAULT_OBJECT_SIZE_MM["width_mm"]))
            height_mm = float(raw_size.get("height_mm", DEFAULT_OBJECT_SIZE_MM["height_mm"]))
        except Exception:
            continue
        cigarette_id = str(raw_size.get("cigarette_id") or raw_size.get("product_id") or _default_cigarette_id(label))
        name = str(raw_size.get("name") or raw_size.get("display_name") or _default_cigarette_name(label))
        selected_class = " selected-object-size-row" if label == selected_label else ""
        rows.append(
            (
                f'<tr class="{selected_class}" data-object-size-row data-label="{html_lib.escape(label)}">'
                f"<td><strong>{html_lib.escape(label)}</strong></td>"
                f'<td><input data-size-field="cigarette_id" type="text" value="{html_lib.escape(cigarette_id, quote=True)}"></td>'
                f'<td><input data-size-field="name" type="text" value="{html_lib.escape(name, quote=True)}"></td>'
                f'<td><input data-size-field="length_mm" type="number" min="1" step="0.1" value="{length_mm:.1f}"></td>'
                f'<td><input data-size-field="width_mm" type="number" min="1" step="0.1" value="{width_mm:.1f}"></td>'
                f'<td><input data-size-field="height_mm" type="number" min="1" step="0.1" value="{height_mm:.1f}"></td>'
                "</tr>"
            )
        )

    return (
        '<details class="debug-object-sizes-panel">'
        '<summary class="debug-object-sizes-title">'
        '<strong>烟盒信息配置</strong>'
        '<span>展开后可按 YOLO id 修改名称、长、宽、厚度</span>'
        "</summary>"
        '<div class="debug-object-sizes-note">名称用于页面显示；长/宽参与 PnP 坐标计算；高/厚度用于 18085 的 3D 展示。</div>'
        '<div class="debug-object-sizes-table-wrap">'
        '<table class="debug-object-sizes-table">'
        "<thead><tr><th>YOLO id</th><th>烟 ID</th><th>烟名称</th><th>长 mm</th><th>宽 mm</th><th>高/厚 mm</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
        '<div class="debug-object-sizes-actions">'
        '<button id="debugSaveObjectSizes" class="button primary-alt-button" type="button">保存烟盒信息</button>'
        '<span id="debugObjectSizesStatus"></span>'
        "</div>"
        "</details>"
    )


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
        near_alignment = item.get("near_edge_robot_alignment")
        if not isinstance(near_alignment, dict):
            near_alignment = {}
        near_target = near_alignment.get("target")
        if not isinstance(near_target, dict):
            near_target = {}
        selected = "是" if item.get("selected") else ""
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(selected)}</td>"
            f"<td>{html_lib.escape(_orientation_cn(name))}</td>"
            f"<td>{html_lib.escape(_fmt_list(item.get('object_top_size_mm'), ' mm'))}</td>"
            f"<td>{html_lib.escape(_fmt_number(item.get('range_from_left_camera_mm'), ' mm', 1))}</td>"
            f"<td>{html_lib.escape(_fmt_number(near_target.get('ground_forward_mm'), ' mm', 1))}</td>"
            f"<td>{html_lib.escape(_fmt_number(target.get('ground_forward_mm'), ' mm', 1))}</td>"
            f"<td>{html_lib.escape(_fmt_number(target.get('vertical_down_mm'), ' mm', 1))}</td>"
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
        "<th>近端边前向</th><th>中心前向</th><th>中心垂直距离</th><th>左右偏差</th><th>朝目标转角</th><th>烟盒长轴角</th>"
        "<th>重投影误差</th><th>左右深度差</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _visualization_row(source: str, label: str, value: str, note: str) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(source)}</td>"
        f"<td>{html_lib.escape(label)}</td>"
        f"<td><strong>{html_lib.escape(value)}</strong></td>"
        f"<td>{html_lib.escape(note)}</td>"
        "</tr>"
    )


def _g1d_visualization_table(payload: dict[str, Any]) -> str:
    data = payload.get("g1d_visualization")
    if not isinstance(data, dict):
        data = _g1d_visualization_data(payload)
    yolo = data.get("yolo") if isinstance(data.get("yolo"), dict) else {}
    box = data.get("box") if isinstance(data.get("box"), dict) else {}
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    camera = data.get("camera") if isinstance(data.get("camera"), dict) else {}
    robot_state = data.get("robot_state") if isinstance(data.get("robot_state"), dict) else {}

    rows = [
        _visualization_row(
            "YOLO",
            "选中目标",
            f"{yolo.get('display_name') or yolo.get('label') or '-'} / conf {_fmt_number(yolo.get('confidence'), '', 3)}",
            "当前用于 PnP 和相对位置计算的那个 mask",
        ),
        _visualization_row(
            "YOLO/PnP",
            "中心点坐标",
            _fmt_list(box.get("center_xyz_mm")),
            "烟盒上表面中心点，left_camera_optical 坐标",
        ),
        _visualization_row(
            "YOLO/PnP",
            "近端边中点",
            _fmt_list(box.get("near_edge_midpoint_xyz_mm")),
            "靠近机器人那条边的中点，底盘距离优先看这个",
        ),
        _visualization_row(
            "YOLO/PnP",
            "朝目标转角",
            _fmt_number(metrics.get("turn_to_target_yaw_deg"), " deg", 2),
            "机器人先面向目标中心需要转的角度",
        ),
        _visualization_row(
            "YOLO/PnP",
            "烟盒长轴角",
            _fmt_number(metrics.get("box_long_axis_yaw_deg"), " deg", 2),
            "烟盒长轴相对机器人前方的无向夹角",
        ),
        _visualization_row(
            "YOLO/PnP",
            "中心垂直距离",
            _fmt_number(metrics.get("center_vertical_down_mm"), " mm", 1),
            "沿地面垂直方向到烟盒上表面中心",
        ),
        _visualization_row(
            "YOLO/PnP",
            "中心地面前向",
            _fmt_number(metrics.get("center_ground_forward_mm"), " mm", 1),
            "中心点投影到地面前向方向的距离",
        ),
        _visualization_row(
            "YOLO/PnP",
            "近端边地面前向",
            _fmt_number(metrics.get("near_edge_ground_forward_mm"), " mm", 1),
            "调底盘前后距离更建议用这个，目标一般是 200mm",
        ),
        _visualization_row(
            "YOLO/PnP",
            "左右偏差",
            _fmt_number(metrics.get("right_mm"), " mm", 1),
            "正数表示目标在机器人右侧",
        ),
        _visualization_row(
            "相机模型",
            "left camera",
            f"{camera.get('mount_parent_link') or 'torso_link'} / {_fmt_number(camera.get('camera_to_vertical_deg'), ' deg', 1)}",
            "相机使用官方 d435_joint 挂点，光轴使用 47.6° 安装角",
        ),
        _visualization_row(
            "本体状态",
            "joint states",
            "外部读取",
            "YOLO 不产生关节角；3D 嵌入时要从本体 joint states 同步",
        ),
        _visualization_row(
            "本体状态",
            "joint 单位",
            str(robot_state.get("units") or "-"),
            "prismatic 用米，旋转关节用弧度，按 URDF joint 名称驱动模型",
        ),
    ]
    return (
        "<table class=\"visualization-data\">"
        "<thead><tr><th>来源</th><th>数据</th><th>当前值</th><th>用途</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


ORIENTATION_DISPLAY_ORDER = ("long_x_short_y", "short_x_long_y")


def _embedded_g1d_visualizer_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""
  <section class="g1d-visualizer-card">
    <div class="g1d-visualizer-title">
      <h2><a id="g1dVisualizerOpenLink" href="#" title="打开 18085 可视化页面">G1-D / 烟盒 3D 相对位置</a></h2>
      <span>使用本次 /debug 已计算结果，不重新拍照</span>
    </div>
    <iframe
      id="g1dVisualizerFrame"
      class="g1d-visualizer-frame"
      title="G1-D cigarette relative position"
      loading="eager"
    ></iframe>
    <script id="g1dCurrentPosePayload" type="application/json">{payload_json}</script>
    <script>
      (() => {{
        const frame = document.getElementById("g1dVisualizerFrame");
        const payloadEl = document.getElementById("g1dCurrentPosePayload");
        const openLink = document.getElementById("g1dVisualizerOpenLink");
        if (!frame || !payloadEl) return;
        const visualizerOrigin = `${{window.location.protocol}}//${{window.location.hostname}}:18085`;
        if (openLink) openLink.href = `${{visualizerOrigin}}/`;
        let payload = JSON.parse(payloadEl.textContent || "{{}}");
        frame.src = `${{visualizerOrigin}}/?compact=1&embedded=1&no_fetch=1&view=normal&t=${{Date.now()}}`;
        const postPayload = () => {{
          if (!frame.contentWindow) return;
          frame.contentWindow.postMessage({{ type: "g1d-visualizer-pose", pose: payload }}, visualizerOrigin);
        }};
        window.updateG1dVisualizer = (nextPayload) => {{
          payload = nextPayload || {{}};
          payloadEl.textContent = JSON.stringify(payload);
          postPayload();
          setTimeout(postPayload, 350);
        }};
        frame.addEventListener("load", () => {{
          postPayload();
          setTimeout(postPayload, 350);
          setTimeout(postPayload, 1200);
        }});
        window.addEventListener("message", (event) => {{
          if (event.origin !== visualizerOrigin) return;
          if (event.data && event.data.type === "g1d-visualizer-ready") {{
            postPayload();
          }}
        }});
      }})();
    </script>
  </section>
"""


def _debug_label_select_html(payload: dict[str, Any]) -> str:
    requested_label = str(payload.get("requested_yolo_label") or "")
    model_class_names = _nested(payload, "server", "yolo_class_names")
    if not isinstance(model_class_names, list):
        model_class_names = []
    labels = [str(item) for item in model_class_names if str(item)]
    if requested_label and requested_label not in labels:
        labels.append(requested_label)
    label_choices = [("", "自动"), *[(label, _payload_label_display_name(payload, label) or label) for label in labels]]

    options = []
    for value, text in label_choices:
        selected = " selected" if value == requested_label else ""
        options.append(
            f'<option value="{html_lib.escape(value, quote=True)}"{selected}>{html_lib.escape(text)}</option>'
        )
    return f"""
    <label class="debug-label-control" for="debugLabelSelect">
      <span>烟盒类别</span>
      <select id="debugLabelSelect" name="label">
        {''.join(options)}
      </select>
    </label>
"""


def _g1d_adjust_controls_html(payload: dict[str, Any]) -> str:
    requested_label = str(payload.get("requested_yolo_label") or "")
    selected_label_text = html_lib.escape(_payload_label_display_name(payload, requested_label) if requested_label else "自动")
    selected_label_json = json.dumps(requested_label, ensure_ascii=False).replace("</", "<\\/")
    return f"""
  <section class="g1d-adjust-panel">
    <div class="g1d-adjust-title">
      <h2>G1-D 位置微调</h2>
      <span>当前烟盒类别：<span id="g1dAdjustCurrentLabel">{selected_label_text}</span></span>
    </div>
    <div class="g1d-adjust-actions">
      <button id="g1dAdjustBtn" class="button primary-button" type="button">执行位置微调</button>
      <button id="g1dAdjustDryRunBtn" class="button" type="button">预览微调计划</button>
      <button id="g1dRightEntryAdjustBtn" class="button primary-alt-button" type="button">执行右手安全入位</button>
      <button id="g1dRightEntryDryRunBtn" class="button" type="button">预览右手入位</button>
    </div>
    <pre id="g1dAdjustStatus" class="g1d-adjust-status">等待操作</pre>
    <script>
      (() => {{
        const adjustBtn = document.getElementById("g1dAdjustBtn");
        const dryRunBtn = document.getElementById("g1dAdjustDryRunBtn");
        const rightEntryBtn = document.getElementById("g1dRightEntryAdjustBtn");
        const rightEntryDryRunBtn = document.getElementById("g1dRightEntryDryRunBtn");
        const statusEl = document.getElementById("g1dAdjustStatus");
        const initialSelectedLabel = {selected_label_json};
        const currentSelectedLabel = () => {{
          const select = document.getElementById("debugLabelSelect");
          return select ? select.value : initialSelectedLabel;
        }};
        const setStatus = (text) => {{
          if (statusEl) statusEl.textContent = text;
        }};
        const runAdjust = async (dryRun, mode) => {{
          const selectedLabel = currentSelectedLabel();
          const labelQuery = selectedLabel ? `&label=${{encodeURIComponent(selectedLabel)}}` : "";
          const endpoint = mode === "right_entry" ? "/g1d/adjust_right_entry" : "/g1d/adjust";
          const url = `${{endpoint}}?dry_run=${{dryRun ? "1" : "0"}}${{labelQuery}}&t=${{Date.now()}}`;
          const button = mode === "right_entry"
            ? (dryRun ? rightEntryDryRunBtn : rightEntryBtn)
            : (dryRun ? dryRunBtn : adjustBtn);
          if (button) button.disabled = true;
          const modeText = mode === "right_entry" ? "右手安全入位" : "位置微调";
          setStatus(dryRun ? `正在读取 YOLO 并生成${{modeText}}计划...` : `正在执行 G1-D ${{modeText}}，请等待...`);
          try {{
            const res = await fetch(url, {{ method: "POST", cache: "no-store" }});
            const payload = await res.json();
            setStatus(JSON.stringify(payload, null, 2));
            if (!dryRun && payload.ok) {{
              setStatus("微调完成，正在重新拍照刷新 YOLO...");
              const refreshLabelQuery = selectedLabel ? `label=${{encodeURIComponent(selectedLabel)}}&` : "";
              window.location.href = `/debug?${{refreshLabelQuery}}t=${{Date.now()}}`;
            }}
          }} catch (err) {{
            setStatus(`微调请求失败: ${{err}}`);
          }} finally {{
            if (button) button.disabled = false;
          }}
        }};
        if (adjustBtn) adjustBtn.addEventListener("click", () => runAdjust(false, "default"));
        if (dryRunBtn) dryRunBtn.addEventListener("click", () => runAdjust(true, "default"));
        if (rightEntryBtn) rightEntryBtn.addEventListener("click", () => runAdjust(false, "right_entry"));
        if (rightEntryDryRunBtn) rightEntryDryRunBtn.addEventListener("click", () => runAdjust(true, "right_entry"));
      }})();
    </script>
  </section>
"""


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
    requested_label_raw = str(payload.get("requested_yolo_label") or "")
    label_query = f"label={quote(requested_label_raw)}&" if requested_label_raw else ""
    intrinsics_query = _debug_intrinsics_query(payload)
    rerun_query = f"{label_query}{intrinsics_query}"
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
            f'<section><h2>左目上方点放大</h2><img src="{image_src("/latest/left_projected_zoom.jpg")}" alt="left projected above points zoom"></section>',
            f'<section><h2>左目全部候选</h2><img src="{image_src("/latest/left_candidates.jpg")}" alt="left candidates"></section>',
            f'<section><h2>右目原图</h2><img src="{image_src("/latest/right_input.jpg")}" alt="right input"></section>',
            f'<section><h2>右目四点</h2><img src="{image_src("/latest/right_points.jpg")}" alt="right points"></section>',
            f'<section><h2>右目全部候选</h2><img src="{image_src("/latest/right_candidates.jpg")}" alt="right candidates"></section>',
        ]
    )
    run_again_href = html_lib.escape(f"/debug?{rerun_query}t={cache_token}")
    pose_href = html_lib.escape(f"/pose?{rerun_query}t={cache_token}")
    xyz_href = html_lib.escape(f"/xyz?{rerun_query}t={cache_token}")
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
    .button {{ display: inline-block; padding: 7px 10px; border: 1px solid #8795a1; color: #102a43; text-decoration: none; background: #fff; font: inherit; cursor: pointer; }}
    .button:disabled {{ color: #829ab1; cursor: wait; }}
    .primary-button {{ background: #1266f1; border-color: #1266f1; color: #fff; }}
    .primary-alt-button {{ background: #0f766e; border-color: #0f766e; color: #fff; }}
    .status {{ margin: 12px 0; }}
    .debug-label-control {{ display: inline-flex; align-items: center; gap: 6px; color: #243b53; font-weight: 700; }}
    .debug-label-control select {{ padding: 7px 9px; border: 1px solid #8795a1; font: inherit; background: #fff; }}
    .debug-intrinsics-panel {{ margin: 10px 0 12px; padding: 10px; border: 1px solid #d9e2ec; border-radius: 6px; background: #f8fbff; }}
    .debug-intrinsics-main {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .debug-intrinsics-panel strong {{ margin-right: 4px; color: #102a43; }}
    .debug-intrinsics-advanced {{ margin-top: 8px; }}
    .debug-intrinsics-advanced summary {{ cursor: pointer; color: #486581; font-weight: 700; }}
    .debug-intrinsics-fields {{ display: flex; align-items: end; gap: 10px; flex-wrap: wrap; margin-top: 8px; }}
    .debug-intrinsics-panel label {{ display: grid; gap: 3px; color: #52606d; font-size: 12px; font-weight: 700; }}
    .debug-intrinsics-panel input {{ width: 84px; padding: 6px 8px; border: 1px solid #8795a1; font: inherit; background: #fff; color: #102a43; }}
    .debug-intrinsics-panel span {{ color: #627d98; font-size: 12px; }}
    .debug-yolo-model-panel {{ margin: 10px 0 16px; padding: 10px; border: 1px solid #d9e2ec; border-radius: 6px; background: #f8fbff; }}
    .debug-yolo-model-main {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }}
    .debug-yolo-model-main strong {{ color: #102a43; }}
    .debug-yolo-model-main span, .debug-yolo-model-controls span {{ color: #627d98; font-size: 12px; }}
    .debug-yolo-model-controls {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .debug-yolo-model-controls select, .debug-yolo-model-controls input[type="text"] {{ min-width: 220px; padding: 7px 9px; border: 1px solid #8795a1; font: inherit; background: #fff; color: #102a43; }}
    .debug-yolo-model-controls input[type="file"] {{ max-width: 260px; }}
    .debug-object-sizes-panel {{ margin: 10px 0 16px; padding: 10px; border: 1px solid #d9e2ec; border-radius: 6px; background: #fffdf7; }}
    .debug-object-sizes-title {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; cursor: pointer; }}
    .debug-object-sizes-title strong {{ color: #102a43; }}
    .debug-object-sizes-title span, .debug-object-sizes-actions span {{ color: #627d98; font-size: 12px; }}
    .debug-object-sizes-note {{ color: #627d98; font-size: 12px; margin: 8px 0; }}
    .debug-object-sizes-table-wrap {{ overflow-x: auto; }}
    .debug-object-sizes-table input {{ width: 94px; padding: 6px 8px; border: 1px solid #8795a1; font: inherit; background: #fff; color: #102a43; }}
    .debug-object-sizes-table input[type="text"] {{ width: 180px; }}
    .debug-object-sizes-table .selected-object-size-row td {{ background: #fff7cc; }}
    .debug-object-sizes-actions {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 8px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 14px 0 18px; }}
    .summary-tile {{ border: 1px solid #bcccdc; border-radius: 6px; padding: 10px; background: #f8fbff; min-height: 76px; }}
    .summary-label {{ color: #52606d; font-size: 13px; margin-bottom: 6px; }}
    .summary-value {{ color: #102a43; font-size: 20px; font-weight: 700; line-height: 1.2; word-break: break-word; }}
    .summary-note {{ color: #627d98; font-size: 12px; margin-top: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    section {{ border: 1px solid #d9e2ec; padding: 10px; }}
    img {{ width: 100%; max-width: 640px; height: auto; display: block; background: #f0f4f8; }}
    .g1d-visualizer-card {{ width: min(380px, 100%); margin: 16px 0 18px; border: 1px solid #bcccdc; border-radius: 6px; padding: 10px; background: #f8fbff; }}
    .g1d-visualizer-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }}
    .g1d-visualizer-title h2 {{ margin: 0; }}
    .g1d-visualizer-title a {{ color: #102a43; text-decoration: none; }}
    .g1d-visualizer-title a:hover {{ text-decoration: underline; }}
    .g1d-visualizer-title span {{ color: #627d98; font-size: 13px; }}
    .g1d-visualizer-frame {{ width: 100%; height: 560px; border: 0; border-radius: 6px; display: block; background: #0c1014; }}
    .g1d-adjust-panel {{ border: 1px solid #bcccdc; border-radius: 6px; padding: 10px; margin: 14px 0 18px; background: #f8fbff; }}
    .g1d-adjust-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
    .g1d-adjust-title h2 {{ margin: 0; }}
    .g1d-adjust-title span {{ color: #627d98; font-size: 13px; }}
    .g1d-adjust-actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }}
    .g1d-adjust-status {{ margin: 0; max-height: 260px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 5px; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    .debug-json-details {{ margin: 16px 0; }}
    .debug-json-details summary {{ cursor: pointer; color: #102a43; font-size: 20px; font-weight: 700; margin: 0 0 8px; }}
    .debug-json-details pre {{ margin-top: 10px; }}
    pre {{ overflow: auto; background: #102a43; color: #f0f4f8; padding: 12px; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>烟盒 YOLO 调试</h1>
    {_debug_label_select_html(payload)}
    <a id="debugRunAgainLink" class="button" href="{run_again_href}">重新拍照</a>
    <a id="debugPoseLink" class="button" href="{pose_href}">完整 JSON</a>
    <a id="debugXyzLink" class="button" href="{xyz_href}">坐标 JSON</a>
  </header>
  {_debug_yolo_model_controls_html(payload)}
  {_debug_intrinsics_controls_html(payload)}
  {_debug_object_size_controls_html(payload)}
  <script>
    (() => {{
      const select = document.getElementById("debugLabelSelect");
      const runAgain = document.getElementById("debugRunAgainLink");
      const pose = document.getElementById("debugPoseLink");
      const xyz = document.getElementById("debugXyzLink");
      const intrinsicInputs = Array.from(document.querySelectorAll("[data-intrinsic]"));
      const distInputs = Array.from(document.querySelectorAll("[data-dist-index]"));
      const rightInputs = Array.from(document.querySelectorAll("[data-intrinsic-right]"));
      const rightDistInputs = Array.from(document.querySelectorAll("[data-dist-right-index]"));
      const saveRightButton = document.getElementById("debugSaveRightIntrinsics");
      const rightStatus = document.getElementById("debugRightIntrinsicsStatus");
      const saveCalibratedButton = document.getElementById("debugSaveCalibratedIntrinsics");
      const saveDefaultButton = document.getElementById("debugSaveDefaultIntrinsics");
      const saveStatus = document.getElementById("debugIntrinsicsSaveStatus");
      const currentIntrinsics = document.getElementById("debugIntrinsicsCurrent");
      const yoloModelSelect = document.getElementById("debugYoloModelSelect");
      const yoloModelPath = document.getElementById("debugYoloModelPath");
      const saveYoloModelButton = document.getElementById("debugSaveYoloModel");
      const uploadYoloModelButton = document.getElementById("debugUploadYoloModel");
      const yoloModelFile = document.getElementById("debugYoloModelFile");
      const yoloModelStatus = document.getElementById("debugYoloModelStatus");
      const objectSizeRows = Array.from(document.querySelectorAll("[data-object-size-row]"));
      const saveObjectSizesButton = document.getElementById("debugSaveObjectSizes");
      const objectSizesStatus = document.getElementById("debugObjectSizesStatus");
      const calibratedIntrinsics = {{
        fx: "275.06",
        fy: "275.39",
        cx: "305.71",
        cy: "268.34",
        dist_coeffs: ["0.05998239", "-0.07112947", "-0.00037432", "0.00015172", "0.01724672"],
      }};
      const readIntrinsicsObject = () => {{
        const values = {{}};
        for (const input of intrinsicInputs) {{
          const name = input.dataset.intrinsic;
          const value = Number(input.value);
          if (name && Number.isFinite(value)) values[name] = value;
        }}
        return values;
      }};
      const readDistCoeffs = () => {{
        const coeffs = [0, 0, 0, 0, 0];
        for (const input of distInputs) {{
          const idx = Number(input.dataset.distIndex);
          const value = Number(input.value);
          if (Number.isInteger(idx) && idx >= 0 && idx < 5 && Number.isFinite(value)) {{
            coeffs[idx] = value;
          }}
        }}
        return coeffs;
      }};
      const readIntrinsicsQuery = () => {{
        const params = new URLSearchParams();
        const values = readIntrinsicsObject();
        for (const [name, value] of Object.entries(values)) {{
          params.set(name, String(value));
        }}
        if (distInputs.length) params.set("dist_coeffs", readDistCoeffs().join(","));
        const text = params.toString();
        return text ? `${{text}}&` : "";
      }};
      const setIntrinsics = (values) => {{
        for (const input of intrinsicInputs) {{
          const name = input.dataset.intrinsic;
          if (Object.prototype.hasOwnProperty.call(values, name)) input.value = values[name];
        }}
        if (Array.isArray(values.dist_coeffs)) {{
          for (const input of distInputs) {{
            const idx = Number(input.dataset.distIndex);
            if (Number.isInteger(idx) && idx >= 0 && idx < values.dist_coeffs.length) {{
              input.value = values.dist_coeffs[idx];
            }}
          }}
        }}
        updateLinks();
      }};
      const updateLinks = () => {{
        const label = select && select.value ? `label=${{encodeURIComponent(select.value)}}&` : "";
        const intrinsics = readIntrinsicsQuery();
        const adjustLabel = document.getElementById("g1dAdjustCurrentLabel");
        if (adjustLabel) adjustLabel.textContent = select && select.value
          ? (select.options[select.selectedIndex]?.textContent || select.value)
          : "自动";
        const t = Date.now();
        if (runAgain) runAgain.href = `/debug?${{label}}${{intrinsics}}t=${{t}}`;
        if (pose) pose.href = `/pose?${{label}}${{intrinsics}}t=${{t}}`;
        if (xyz) xyz.href = `/xyz?${{label}}${{intrinsics}}t=${{t}}`;
      }};
      const copyHtml = (doc, id) => {{
        const current = document.getElementById(id);
        const next = doc.getElementById(id);
        if (current && next) current.innerHTML = next.innerHTML;
      }};
      const runDebugRequest = async (event) => {{
        event.preventDefault();
        updateLinks();
        if (runAgain) runAgain.textContent = "请求中...";
        try {{
          const res = await fetch(runAgain.href, {{ cache: "no-store" }});
          const html = await res.text();
          const doc = new DOMParser().parseFromString(html, "text/html");
          for (const id of [
            "debugStatus",
            "debugKeySummary",
            "debugImageGrid",
            "debugHypotheses",
            "debugVisualizationData",
            "debugSelectedCandidate",
            "debugLeftCandidates",
            "debugRightCandidates",
            "debugJson",
          ]) {{
            copyHtml(doc, id);
          }}
          const nextPayloadEl = doc.getElementById("g1dCurrentPosePayload");
          if (nextPayloadEl && window.updateG1dVisualizer) {{
            window.updateG1dVisualizer(JSON.parse(nextPayloadEl.textContent || "{{}}"));
          }}
          window.history.replaceState(null, "", runAgain.href);
        }} catch (err) {{
          const status = document.getElementById("debugStatus");
          if (status) status.textContent = `重新拍照请求失败: ${{err}}`;
        }} finally {{
          if (runAgain) runAgain.textContent = "重新拍照";
          updateLinks();
        }}
      }};
      const saveDefaultIntrinsics = async (button) => {{
        if (!button) return;
        const values = readIntrinsicsObject();
        for (const name of ["fx", "fy", "cx", "cy"]) {{
          if (!Number.isFinite(values[name])) {{
            if (saveStatus) saveStatus.textContent = `${{name}} 不是有效数字`;
            return;
          }}
        }}
        if (distInputs.length) values.dist_coeffs = readDistCoeffs();
        button.disabled = true;
        if (saveStatus) saveStatus.textContent = "保存中...";
        try {{
          const res = await fetch("/config/intrinsics", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(values),
            cache: "no-store",
          }});
          const payload = await res.json();
          if (!res.ok || !payload.ok) throw new Error(payload.error || `HTTP ${{res.status}}`);
          const effective = payload.effective_defaults || {{}};
          if (currentIntrinsics && Number.isFinite(Number(effective.fx))) {{
            const dist = Array.isArray(effective.dist_coeffs) ? effective.dist_coeffs : [];
            currentIntrinsics.textContent = `当前 API 默认 / 本次计算：fx=${{Number(effective.fx).toFixed(2)}}, fy=${{Number(effective.fy).toFixed(2)}}, cx=${{Number(effective.cx).toFixed(2)}}, cy=${{Number(effective.cy).toFixed(2)}}, dist=[${{dist.map((v) => Number(v).toFixed(6)).join(", ")}}]`;
          }}
          if (saveStatus) saveStatus.textContent = "已设为 API 默认";
        }} catch (err) {{
          if (saveStatus) saveStatus.textContent = `保存失败: ${{err}}`;
        }} finally {{
          button.disabled = false;
        }}
      }};
      const saveRightIntrinsics = async () => {{
        if (!saveRightButton) return;
        const body = {{}};
        for (const input of rightInputs) {{
          const name = input.dataset.intrinsicRight;
          const value = Number(input.value);
          if (name && Number.isFinite(value)) body[`${{name}}_right`] = value;
        }}
        if (rightDistInputs.length) {{
          const coeffs = [0, 0, 0, 0, 0];
          for (const input of rightDistInputs) {{
            const idx = Number(input.dataset.distRightIndex);
            const value = Number(input.value);
            if (Number.isInteger(idx) && idx >= 0 && idx < 5 && Number.isFinite(value)) coeffs[idx] = value;
          }}
          body.dist_coeffs_right = coeffs;
        }}
        saveRightButton.disabled = true;
        if (rightStatus) rightStatus.textContent = "保存中...";
        try {{
          const res = await fetch("/config/intrinsics_right", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify(body),
            cache: "no-store",
          }});
          const payload = await res.json();
          if (!res.ok || !payload.ok) throw new Error(payload.error || `HTTP ${{res.status}}`);
          if (rightStatus) rightStatus.textContent = "已保存右眼内参";
        }} catch (err) {{
          if (rightStatus) rightStatus.textContent = `保存失败: ${{err}}`;
        }} finally {{
          saveRightButton.disabled = false;
        }}
      }};
      const reloadDebugPage = () => {{
        const url = new URL(window.location.href);
        url.searchParams.set("t", String(Date.now()));
        window.location.href = url.toString();
      }};
      const saveYoloModel = async () => {{
        if (!saveYoloModelButton || !yoloModelPath) return;
        const yolo_model = yoloModelPath.value.trim();
        if (!yolo_model) {{
          if (yoloModelStatus) yoloModelStatus.textContent = "请先选择或填写模型路径";
          return;
        }}
        saveYoloModelButton.disabled = true;
        if (yoloModelStatus) yoloModelStatus.textContent = "切换中...";
        try {{
          const res = await fetch("/config/yolo_model", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ yolo_model }}),
            cache: "no-store",
          }});
          const payload = await res.json();
          if (!res.ok || !payload.ok) throw new Error(payload.error || `HTTP ${{res.status}}`);
          if (yoloModelStatus) yoloModelStatus.textContent = "已切换，正在刷新类别";
          reloadDebugPage();
        }} catch (err) {{
          if (yoloModelStatus) yoloModelStatus.textContent = `切换失败: ${{err}}`;
        }} finally {{
          saveYoloModelButton.disabled = false;
        }}
      }};
      const uploadYoloModel = async () => {{
        if (!uploadYoloModelButton || !yoloModelFile) return;
        const file = yoloModelFile.files && yoloModelFile.files[0];
        if (!file) {{
          if (yoloModelStatus) yoloModelStatus.textContent = "请选择 .pt 模型文件";
          return;
        }}
        const form = new FormData();
        form.append("model_file", file);
        uploadYoloModelButton.disabled = true;
        if (yoloModelStatus) yoloModelStatus.textContent = "上传并加载中...";
        try {{
          const res = await fetch("/config/yolo_model", {{
            method: "POST",
            body: form,
            cache: "no-store",
          }});
          const payload = await res.json();
          if (!res.ok || !payload.ok) throw new Error(payload.error || `HTTP ${{res.status}}`);
          if (yoloModelStatus) yoloModelStatus.textContent = "已上传并切换，正在刷新类别";
          reloadDebugPage();
        }} catch (err) {{
          if (yoloModelStatus) yoloModelStatus.textContent = `上传失败: ${{err}}`;
        }} finally {{
          uploadYoloModelButton.disabled = false;
        }}
      }};
      const readObjectSizesObject = () => {{
        const object_sizes = {{}};
        for (const row of objectSizeRows) {{
          const label = row.dataset.label || "";
          if (!label) continue;
          const entry = {{}};
          for (const input of Array.from(row.querySelectorAll("[data-size-field]"))) {{
            const field = input.dataset.sizeField;
            if (field === "name" || field === "cigarette_id") {{
              entry[field] = input.value.trim() || label;
              continue;
            }}
            const value = Number(input.value);
            if (!field || !Number.isFinite(value) || value <= 0) {{
              throw new Error(`${{label}} 的 ${{field || "尺寸"}} 必须是正数`);
            }}
            entry[field] = value;
          }}
          object_sizes[label] = entry;
        }}
        return object_sizes;
      }};
      const saveObjectSizes = async () => {{
        if (!saveObjectSizesButton) return;
        saveObjectSizesButton.disabled = true;
        if (objectSizesStatus) objectSizesStatus.textContent = "保存中...";
        try {{
          const object_sizes = readObjectSizesObject();
          const res = await fetch("/config/object_sizes", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ object_sizes }}),
            cache: "no-store",
          }});
          const payload = await res.json();
          if (!res.ok || !payload.ok) throw new Error(payload.error || `HTTP ${{res.status}}`);
          if (objectSizesStatus) objectSizesStatus.textContent = "烟盒信息已保存，下一次 /debug、/xyz、/pose 计算生效";
        }} catch (err) {{
          if (objectSizesStatus) objectSizesStatus.textContent = `保存失败: ${{err}}`;
        }} finally {{
          saveObjectSizesButton.disabled = false;
        }}
      }};
      if (select) select.addEventListener("change", updateLinks);
      if (yoloModelSelect && yoloModelPath) {{
        yoloModelSelect.addEventListener("change", () => {{
          yoloModelPath.value = yoloModelSelect.value || "";
        }});
      }}
      for (const input of intrinsicInputs) input.addEventListener("input", updateLinks);
      for (const input of distInputs) input.addEventListener("input", updateLinks);
      if (saveYoloModelButton) saveYoloModelButton.addEventListener("click", saveYoloModel);
      if (uploadYoloModelButton) uploadYoloModelButton.addEventListener("click", uploadYoloModel);
      if (saveObjectSizesButton) saveObjectSizesButton.addEventListener("click", saveObjectSizes);
      if (saveCalibratedButton) saveCalibratedButton.addEventListener("click", async () => {{
        setIntrinsics(calibratedIntrinsics);
        await saveDefaultIntrinsics(saveCalibratedButton);
      }});
      if (saveDefaultButton) saveDefaultButton.addEventListener("click", () => saveDefaultIntrinsics(saveDefaultButton));
      if (saveRightButton) saveRightButton.addEventListener("click", saveRightIntrinsics);
      if (runAgain) runAgain.addEventListener("click", runDebugRequest);
      updateLinks();
    }})();
  </script>
  <div id="debugStatus" class="status">{status_line}</div>
  <h2>关键数据</h2>
  <div id="debugKeySummary">{_key_summary_html(payload)}</div>
  {_g1d_adjust_controls_html(payload)}
  <h2>图片显示</h2>
  <div id="debugImageGrid" class="grid">{image_cards}</div>
  {_embedded_g1d_visualizer_html(payload)}
  <h2>横竖两套假设对比</h2>
  <div id="debugHypotheses">{_alignment_hypotheses_table(payload)}</div>
  <h2>G1-D 可视化需要的数据</h2>
  <div id="debugVisualizationData">{_g1d_visualization_table(payload)}</div>
  <details class="debug-json-details">
    <summary>当前选中的左目候选 JSON</summary>
    <pre id="debugSelectedCandidate">{selected_text}</pre>
  </details>
  <h2>左目 YOLO 候选</h2>
  <div id="debugLeftCandidates">{_candidate_table(left_candidates)}</div>
  <h2>右目 YOLO 候选</h2>
  <div id="debugRightCandidates">{_candidate_table(right_candidates)}</div>
  <details class="debug-json-details">
    <summary>完整 JSON</summary>
    <pre id="debugJson">{json_text}</pre>
  </details>
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
    effective_model = _effective_yolo_model(config)
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
        "yolo_model": effective_model,
        "yolo_model_startup_default": config.yolo_model,
        "yolo_model_runtime_default": _runtime_yolo_model(config),
        "requested_device": config.yolo_device,
        "resolved_device": resolved_device,
        "intrinsics_defaults": _effective_intrinsics(config),
        "intrinsics_startup_defaults": _server_intrinsics(config),
        "intrinsics_runtime_defaults": _runtime_intrinsics(),
        "intrinsics_effective_defaults": _effective_intrinsics(config),
        "intrinsics_right_effective_defaults": _effective_intrinsics_right(config),
        "stereo_effective": _effective_stereo(config),
        "stereo_available": _effective_stereo(config) is not None,
        "object_sizes_runtime_defaults": _runtime_object_sizes(),
        "object_sizes_effective_defaults": _effective_object_sizes(config),
        "yolo_class_names": _yolo_class_names(effective_model),
        "cuda_available": cuda_available,
        "torch_cuda": torch_cuda,
        "model_cache_size": len(YOLO_MODEL_CACHE),
        "endpoints": [
            "/health",
            "/config/intrinsics",
            "/config/intrinsics_right",
            "/config/stereo",
            "/config/yolo_model",
            "/config/object_sizes",
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
            "/g1d/adjust",
            "/g1d/adjust_target_angle",
            "/g1d/adjust_right_entry",
        ],
    }


def _warmup(config: ServerConfig) -> None:
    resolved_model = _resolve_yolo_model_path(_effective_yolo_model(config))
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
                if path == "/config/intrinsics":
                    _serve_intrinsics_config(self, config)
                    return
                if path == "/config/intrinsics_right":
                    _serve_intrinsics_right_config(self, config)
                    return
                if path == "/config/stereo":
                    _serve_stereo_config(self, config)
                    return
                if path == "/config/yolo_model":
                    _serve_yolo_model_config(self, config)
                    return
                if path == "/config/object_sizes":
                    _serve_object_sizes_config(self, config)
                    return
                if path == "/g1d/adjust":
                    _serve_g1d_adjust(self)
                    return
                if path == "/g1d/adjust_target_angle":
                    _serve_g1d_adjust(self, G1D_TARGET_ANGLE_ADJUST_URL)
                    return
                if path == "/g1d/adjust_right_entry":
                    _serve_g1d_adjust(self, G1D_RIGHT_ENTRY_ADJUST_URL)
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
    parser.add_argument("--yolo-model", default="models/YanHe20class.pt")
    parser.add_argument("--yolo-device", default="cuda:0")
    # parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-mask-threshold", type=float, default=0.5)
    parser.add_argument("--focal-px", type=float, default=CALIBRATED_LEFT_FOCAL_PX)
    parser.add_argument("--fx", type=float, default=CALIBRATED_LEFT_FX, help="default left camera fx in pixels")
    parser.add_argument("--fy", type=float, default=CALIBRATED_LEFT_FY, help="default left camera fy in pixels")
    parser.add_argument("--cx", type=float, default=CALIBRATED_LEFT_CX)
    parser.add_argument("--cy", type=float, default=CALIBRATED_LEFT_CY)
    parser.add_argument(
        "--dist-coeffs",
        default=",".join(str(value) for value in CALIBRATED_LEFT_DIST_COEFFS),
        help="default left camera distortion as comma-separated OpenCV coeffs k1,k2,p1,p2,k3",
    )
    parser.add_argument("--fx-right", type=float, default=CALIBRATED_RIGHT_FX, help="default right camera fx")
    parser.add_argument("--fy-right", type=float, default=CALIBRATED_RIGHT_FY, help="default right camera fy")
    parser.add_argument("--cx-right", type=float, default=CALIBRATED_RIGHT_CX, help="default right camera cx")
    parser.add_argument("--cy-right", type=float, default=CALIBRATED_RIGHT_CY, help="default right camera cy")
    parser.add_argument(
        "--dist-coeffs-right",
        default=",".join(str(value) for value in CALIBRATED_RIGHT_DIST_COEFFS),
        help="default right camera distortion as comma-separated coeffs k1,k2,p1,p2,k3",
    )
    parser.add_argument("--stereo-r", default=None, help="stereo rotation R, 9 comma-separated values (row-major 3x3)")
    parser.add_argument("--stereo-t", default=None, help="stereo translation T, 3 comma-separated values in mm")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("config/cigarette_pose_runtime.json"),
        help="JSON file used for runtime defaults changed from the debug page",
    )
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
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        dist_coeffs=parse_dist_coeffs(args.dist_coeffs),
        fx_right=args.fx_right,
        fy_right=args.fy_right,
        cx_right=args.cx_right,
        cy_right=args.cy_right,
        dist_coeffs_right=(
            parse_dist_coeffs(args.dist_coeffs_right) if args.dist_coeffs_right is not None else None
        ),
        stereo_R=parse_float_list(args.stereo_r, expected=9),
        stereo_T=parse_float_list(args.stereo_t, expected=3),
        runtime_config_path=args.runtime_config,
        warmup=not args.no_warmup,
    )
    config.out_root.mkdir(parents=True, exist_ok=True)
    _load_runtime_intrinsics(config)
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
