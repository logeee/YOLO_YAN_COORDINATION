#!/usr/bin/env python3
"""Serve the G1-D cigarette relative-position visualizer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = 17010
DEFAULT_XYZ_URL = "http://127.0.0.1:18081/xyz"
DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
DEFAULT_DDS_INTERFACE = "eth0"
DEFAULT_DDS_LOWSTATE_TOPIC = "rt/lowstate"
DEFAULT_DDS_HISPEED_TOPIC = "rt/hispeed_state"
DEFAULT_UNITREE_SDK2PY_PATH = "/home/unitree/unitree_sdk2_python"
VIEWER_DIR = Path(__file__).resolve().parents[1] / "visualization" / "g1d_cigarette_viewer"
DEFAULT_COLUMN_RAW_MIN_MM = -185.1
DEFAULT_COLUMN_RAW_MAX_MM = 246.9
DEFAULT_COLUMN_VISUAL_MAX_MM = 420.0
COLUMN_RAW_MIN_MM = DEFAULT_COLUMN_RAW_MIN_MM
COLUMN_RAW_MAX_MM = DEFAULT_COLUMN_RAW_MAX_MM
COLUMN_VISUAL_MAX_MM = DEFAULT_COLUMN_VISUAL_MAX_MM
DEFAULT_ROBOT_STATE: dict[str, Any] = {
    "ok": True,
    "source": "visualizer_default",
    "column_extension_mm": 420.0,
    "joints": {
        "LZ_mt_Joint": 0.21,
        "LZ_it_Joint": 0.21,
    },
}
DDS_READER: "_DdsStateReader | None" = None

# --- OUR METHOD (our stereo intrinsics + our hand-eye) -----------------------
# When enabled, /api/xyz recomputes the box pose from the robot's 2D points_px
# using OUR left-camera intrinsics (mono PnP) and OUR hand-eye T_cam2base, and
# injects the result as pose["our_method"]. The robot's own 3D fields are left
# untouched so you can compare. The viewer prefers our_method when present.
OUR: dict[str, Any] = {"enabled": False}

# Intrinsics presets for the YOLO service's own mono-PnP (?fx&fy&cx&cy). The
# historical 260px preset has been retired; keep OLD_INTRINSICS as an alias for
# UI/backward compatibility so no current request sends the old calibration.
OLD_INTRINSICS: dict[str, float] = {"fx": 275.06, "fy": 275.39, "cx": 305.71, "cy": 268.34}
# NEW intrinsics are loaded from a calibration file at startup (see main() ->
# _load_left_intrinsics_file). These hard-coded numbers are the current fallback
# when the file is missing.
NEW_INTRINSICS: dict[str, float] = {"fx": 275.06, "fy": 275.39, "cx": 305.71, "cy": 268.34}
# RIGHT-camera NEW intrinsics, sent as fx_right/fy_right/cx_right/cy_right so the
# service also returns a right-eye 3D point under our calibration.
NEW_INTRINSICS_RIGHT: dict[str, float] = {
    "fx": 274.29699860633724,
    "fy": 274.5716080713627,
    "cx": 289.7163405945703,
    "cy": 274.4892508669222,
}

# NEW distortion coefficients (k1,k2,p1,p2,k3), loaded from the same calibration
# file as NEW_INTRINSICS. ONLY the distortion-corrected combos send these to the
# YOLO service; every other combo sends ZERO_DIST so the service's mono-PnP does
# NOT undistort, keeping the intrinsics/extrinsics-only comparison clean.
NEW_DIST: list[float] = [0.05998239, -0.07112947, -0.00037432, 0.00015172, 0.01724672]
NEW_DIST_RIGHT: list[float] = [
    0.06292257512401175,
    -0.07717484464783685,
    -0.000405354779537882,
    -0.00006950375556195126,
    0.019962308624586825,
]
ZERO_DIST: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]

# Default calibration files on this machine. NEW intrinsics (grid) come from the
# JSON; our_method intrinsics come from the matching OpenCV stereo YAML; NEW
# extrinsics (hand-eye) come from the handeye run that referenced that YAML.
# Left/right eyes each have their own hand-eye optical->base transform.
DEFAULT_RUNTIME_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "cigarette_pose_runtime.json"
DEFAULT_NEW_INTRINSICS_FILE = str(DEFAULT_RUNTIME_CONFIG_FILE)
DEFAULT_NEW_INTRINSICS_YAML = str(DEFAULT_RUNTIME_CONFIG_FILE)
DEFAULT_HANDEYE_LEFT_FILE = "/home/robot/yx/calib/hand_eye/handeye_data/20260625_144450/handeye_result_left.json"
DEFAULT_HANDEYE_RIGHT_FILE = "/home/robot/yx/calib/hand_eye/handeye_data/20260625_144450/handeye_result_right.json"
# URDF carrying the d435 camera joint (nominal/pre-calibration extrinsics). The
# viewer's own g1_d.urdf has no camera link, so we look in the parent project.
DEFAULT_URDF_FILE = "/home/robot/vision_arm_control/g1_d.urdf"
DEFAULT_CAMERA_TO_VERTICAL_DEG = 47.6
DEFAULT_CAMERA_OFFSET_M = (0.0576235, 0.01753, 0.42987)


def _load_intrinsics_file(path: str | Path, *, json_key: str = "left_camera",
                          yaml_key: str = "left_camera_matrix") -> dict[str, float]:
    """Read one camera's fx/fy/cx/cy from a stereo calibration file.

    Supports the ``calibration_result.json`` schema (``<json_key>.camera_matrix``)
    with the stdlib only, and the OpenCV ``stereo_calibration.yaml`` (``<yaml_key>``)
    via ``cv2.FileStorage``. Pass ``json_key="right_camera"`` /
    ``yaml_key="right_camera_matrix"`` for the right eye.
    """
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import cv2

        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            node = fs.getNode(yaml_key)
            K = node.mat() if node is not None else None
        finally:
            fs.release()
        if K is None:
            raise ValueError(f"{yaml_key} not found in {path}")
        return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2])}
    data = json.loads(path.read_text(encoding="utf-8"))
    runtime_key = "intrinsics_right" if json_key == "right_camera" else "intrinsics"
    if isinstance(data.get(runtime_key), dict):
        K = data[runtime_key]
        return {"fx": float(K["fx"]), "fy": float(K["fy"]),
                "cx": float(K["cx"]), "cy": float(K["cy"])}
    K = data[json_key]["camera_matrix"]
    return {"fx": float(K[0][0]), "fy": float(K[1][1]),
            "cx": float(K[0][2]), "cy": float(K[1][2])}


def _load_left_intrinsics_file(path: str | Path) -> dict[str, float]:
    return _load_intrinsics_file(path, json_key="left_camera", yaml_key="left_camera_matrix")


def _load_right_intrinsics_file(path: str | Path) -> dict[str, float]:
    return _load_intrinsics_file(path, json_key="right_camera", yaml_key="right_camera_matrix")


def _load_dist_coeffs_file(path: str | Path, *, json_key: str = "left_camera",
                           yaml_keys: tuple[str, ...] = (
                               "left_distortion_coefficients", "left_dist_coeffs", "left_distortion")) -> list[float]:
    """Read one camera's distortion coeffs ``[k1, k2, p1, p2, k3]`` from calibration.

    Supports ``calibration_result.json`` (``<json_key>.dist_coeffs``, stored as a
    nested ``[[...]]``) with the stdlib, and the OpenCV ``stereo_calibration.yaml``
    (first matching ``yaml_keys``) via ``cv2.FileStorage``. Always returns 5 floats
    so it maps directly onto the YOLO service's ``k1,k2,p1,p2,k3`` param.
    """
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import cv2

        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            D = None
            for key in yaml_keys:
                node = fs.getNode(key)
                if node is not None and not node.empty():
                    D = node.mat()
                    break
        finally:
            fs.release()
        if D is None:
            raise ValueError(f"distortion coeffs not found in {path}")
        flat = [float(v) for row in D.tolist() for v in (row if isinstance(row, list) else [row])]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        runtime_key = "intrinsics_right" if json_key == "right_camera" else "intrinsics"
        if isinstance(data.get(runtime_key), dict) and "dist_coeffs" in data[runtime_key]:
            D = data[runtime_key]["dist_coeffs"]
            flat = [float(v) for v in (D[0] if (isinstance(D, list) and D and isinstance(D[0], list)) else D)]
            out = (flat + [0.0, 0.0, 0.0, 0.0, 0.0])[:5]
            return out
        D = data[json_key]["dist_coeffs"]
        flat = [float(v) for v in (D[0] if (isinstance(D, list) and D and isinstance(D[0], list)) else D)]
    out = (flat + [0.0, 0.0, 0.0, 0.0, 0.0])[:5]
    return out


def _load_left_dist_coeffs_file(path: str | Path) -> list[float]:
    return _load_dist_coeffs_file(path, json_key="left_camera")


def _load_right_dist_coeffs_file(path: str | Path) -> list[float]:
    return _load_dist_coeffs_file(
        path, json_key="right_camera",
        yaml_keys=("right_distortion_coefficients", "right_dist_coeffs", "right_distortion"))


def _load_stereo_left_to_right_file(path: str | Path):
    """Load OpenCV stereo R/T that maps left optical points to right optical."""
    import numpy as np

    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import cv2

        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            R = None
            T = None
            for key in ("R", "rotation_matrix", "stereo_rotation"):
                node = fs.getNode(key)
                if node is not None and not node.empty():
                    R = node.mat()
                    break
            for key in ("T", "translation_vector", "stereo_translation"):
                node = fs.getNode(key)
                if node is not None and not node.empty():
                    T = node.mat()
                    break
        finally:
            fs.release()
        if R is None or T is None:
            raise ValueError(f"stereo R/T not found in {path}")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        stereo = data.get("stereo") if isinstance(data.get("stereo"), dict) else data
        R = stereo.get("R")
        T = stereo.get("T")
        if R is None or T is None:
            raise ValueError(f"stereo R/T not found in {path}")

    R_arr = np.asarray(R, dtype=float).reshape(3, 3)
    T_arr = np.asarray(T, dtype=float).reshape(3)
    if float(np.linalg.norm(T_arr)) > 2.0:
        T_arr = T_arr / 1000.0
    return R_arr, T_arr


def _urdf_nominal_optical_to_base(urdf_path: Path, camera_child: str = "d435_link",
                                  base_link: str = "torso_link"):
    """Build the nominal (pre-calibration) optical-frame -> base transform.

    Mirrors the colleague's URDF path: convert OpenCV optical axes to the
    ``d435_link`` robot axes ([x,y,z]->[z,-x,-y]), then apply the fixed URDF
    ``base_link -> camera_child`` joint origin (xyz + rpy). Returns a 4x4 that
    maps an optical-frame point (meters) directly into the base frame, so it is
    directly comparable to our hand-eye ``T_cam2base``.
    """
    import xml.etree.ElementTree as ET
    import numpy as np

    root = ET.parse(str(urdf_path)).getroot()
    origin = None
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is not None and child.attrib.get("link") == camera_child:
            if parent is None or parent.attrib.get("link") != base_link:
                raise ValueError(
                    f"{camera_child} parent is not {base_link!r} in URDF; nominal transform needs a direct joint."
                )
            origin = joint.find("origin")
            break
    if origin is None:
        raise ValueError(f"No URDF joint with child link {camera_child!r} found in {urdf_path}.")

    xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = Rz @ Ry @ Rx
    T_base_cam[:3, 3] = xyz

    # Optical (X right, Y down, Z forward) -> d435_link robot axes [z, -x, -y].
    M = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
    )
    return T_base_cam @ M


def _default_visual_camera_optical_to_base():
    """Fallback optical->torso transform used by the local standalone viewer."""
    import numpy as np

    theta = np.deg2rad(DEFAULT_CAMERA_TO_VERTICAL_DEG)
    x_right = np.array([0.0, -1.0, 0.0], dtype=float)
    y_down = np.array([-np.cos(theta), 0.0, -np.sin(theta)], dtype=float)
    z_forward = np.array([np.sin(theta), 0.0, -np.cos(theta)], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, 0] = x_right
    T[:3, 1] = y_down
    T[:3, 2] = z_forward
    T[:3, 3] = np.array(DEFAULT_CAMERA_OFFSET_M, dtype=float)
    return T


def _load_T_cam2base_json(handeye_path: str | Path):
    """Read the 4x4 optical->base hand-eye transform from handeye_result.json.

    Uses the stdlib only (no yolo_handeye_pick), so the grid's base-frame boxes
    work on a standalone checkout. Falls back to R_cam2base + t_cam2base_m.
    """
    import numpy as np

    data = json.loads(Path(handeye_path).read_text(encoding="utf-8"))
    T = data.get("T_cam2base")
    if T is not None:
        return np.asarray(T, dtype=float).reshape(4, 4)
    R = data.get("R_cam2base")
    t = data.get("t_cam2base_m")
    if R is None or t is None:
        raise ValueError(f"no T_cam2base (or R_cam2base + t_cam2base_m) in {handeye_path}")
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    M[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return M


def _derive_right_T_from_left_stereo(T_base_left, stereo_path: str | Path):
    """Derive right optical->base from left optical->base and stereo R/T."""
    import numpy as np

    R_left_to_right, t_left_to_right = _load_stereo_left_to_right_file(stereo_path)
    T_left_from_right = np.eye(4)
    T_left_from_right[:3, :3] = R_left_to_right.T
    T_left_from_right[:3, 3] = -R_left_to_right.T @ t_left_to_right
    return T_base_left @ T_left_from_right


def _resolve_urdf_path() -> Path | None:
    """Find a g1_d.urdf that actually contains the d435 camera joint."""
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    candidates = [
        Path(DEFAULT_URDF_FILE),
        Path("/unitree/module/unitree_eai/xr_teleoperate/assets/g1_D/g1_d.urdf"),
        Path("/unitree/ota/backup/module/unitree_eai/0.9.7.3/module/unitree_eai/file/unitree/module/unitree_eai/xr_teleoperate/assets/g1_D/g1_d.urdf"),
        VIEWER_DIR / "g1_d.urdf",
        repo_root / "g1_d.urdf",
        repo_root.parent / "g1_d.urdf",
    ]
    for cand in candidates:
        try:
            if cand.exists() and "d435_link" in cand.read_text(encoding="utf-8", errors="ignore"):
                return cand
        except OSError:
            continue
    return None


def _load_our_method(intrinsics_path: str, handeye_path: str, handeye_right_path: str = "") -> None:
    """Load the calibration the viewer needs.

    Extrinsics (left/right hand-eye ``T_cam2base`` + nominal URDF) drive the grid's
    base-frame boxes and are loaded with the stdlib, so they work even when
    ``yolo_handeye_pick`` is unavailable. The our_method PnP box is loaded
    separately and best-effort.
    """
    # NEW LEFT extrinsics: left hand-eye optical->base, straight from the JSON.
    try:
        OUR["T"] = _load_T_cam2base_json(handeye_path)
        OUR["handeye"] = handeye_path
        print(f"[extrinsics] LEFT hand-eye from {handeye_path}: "
              f"cam_offset(torso)={[round(float(v), 4) for v in OUR['T'][:3, 3]]}", flush=True)
    except (Exception, SystemExit) as exc:
        OUR["T"] = None
        print(f"[extrinsics] LEFT hand-eye unavailable ({exc})", flush=True)

    # NEW RIGHT extrinsics: right hand-eye optical->base, for the right-eye combos.
    if handeye_right_path:
        try:
            OUR["T_right"] = _load_T_cam2base_json(handeye_right_path)
            OUR["handeye_right"] = handeye_right_path
            print(f"[extrinsics] RIGHT hand-eye from {handeye_right_path}: "
                  f"cam_offset(torso)={[round(float(v), 4) for v in OUR['T_right'][:3, 3]]}", flush=True)
        except (Exception, SystemExit) as exc:
            OUR["T_right"] = None
            print(f"[extrinsics] RIGHT hand-eye unavailable ({exc})", flush=True)

    # OLD extrinsics: nominal URDF (pre-calibration) optical->base.
    try:
        urdf_path = _resolve_urdf_path()
        if urdf_path is None:
            raise FileNotFoundError("no g1_d.urdf containing a d435_link joint found")
        OUR["T_urdf"] = _urdf_nominal_optical_to_base(urdf_path)
        print(f"[extrinsics] urdf nominal from {urdf_path}: "
              f"cam_offset(torso)={[round(float(v), 4) for v in OUR['T_urdf'][:3, 3]]}", flush=True)
    except (Exception, SystemExit) as exc:
        OUR["T_urdf"] = _default_visual_camera_optical_to_base()
        print(
            "[extrinsics] urdf nominal transform unavailable "
            f"({exc}); using local viewer fallback cam_offset="
            f"{[round(float(v), 4) for v in OUR['T_urdf'][:3, 3]]}",
            flush=True,
        )

    if OUR.get("T") is None and OUR.get("T_urdf") is not None:
        OUR["T"] = OUR["T_urdf"]
        OUR["handeye"] = "local_viewer_fallback_same_as_urdf"

    if OUR.get("T_right") is None and OUR.get("T") is not None:
        stereo_candidates: list[str | Path] = []
        for cand in (intrinsics_path, DEFAULT_RUNTIME_CONFIG_FILE, DEFAULT_NEW_INTRINSICS_FILE, DEFAULT_NEW_INTRINSICS_YAML):
            if cand and cand not in stereo_candidates:
                stereo_candidates.append(cand)
        last_exc: Exception | SystemExit | None = None
        for stereo_path in stereo_candidates:
            try:
                OUR["T_right"] = _derive_right_T_from_left_stereo(OUR["T"], stereo_path)
                OUR["handeye_right"] = f"derived_from_left_stereo:{stereo_path}"
                print(f"[extrinsics] RIGHT derived from LEFT + stereo {stereo_path}: "
                      f"cam_offset(torso)={[round(float(v), 4) for v in OUR['T_right'][:3, 3]]}",
                      flush=True)
                break
            except (Exception, SystemExit) as exc:
                last_exc = exc
        else:
            print(f"[extrinsics] RIGHT derived transform unavailable ({last_exc})", flush=True)

    # Optional our_method PnP box (4th estimate); needs yolo_handeye_pick.
    try:
        scripts_dir = Path(__file__).resolve().parents[3]
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import yolo_handeye_pick as core  # our mono-PnP + hand-eye helpers

        K, dist = core.load_left_intrinsics(Path(intrinsics_path))
        if OUR.get("T") is None:
            OUR["T"] = core.load_T_cam2base(Path(handeye_path))
        OUR.update({"enabled": True, "core": core, "K": K, "dist": dist,
                    "intrinsics": intrinsics_path})
        print(f"[our_method] ENABLED  fx={K[0, 0]:.2f} fy={K[1, 1]:.2f}", flush=True)
    except (Exception, SystemExit) as exc:  # loaders raise SystemExit on bad/missing files
        OUR["enabled"] = False
        print(f"[our_method] PnP box disabled ({exc}); grid boxes still use the extrinsics above",
              flush=True)


def _optical_center_m(payload: dict[str, Any]):
    """Colleague optical center point (meters, left_camera_optical) from a payload."""
    if not isinstance(payload, dict):
        return None
    viz_box = (payload.get("g1d_visualization") or {}).get("box") or {}
    p_mm = payload.get("center_xyz_mm") or viz_box.get("center_xyz_mm")
    if not p_mm or len(p_mm) < 3:
        return None
    import numpy as np

    return np.array([float(p_mm[0]), float(p_mm[1]), float(p_mm[2])], dtype=float) / 1000.0


def _optical_center_right_m(payload: dict[str, Any]):
    """Right-eye optical center point (meters, right_camera_optical) from a payload.

    The YOLO service nests the right-eye result under ``payload["right"]`` with its
    own ``center_xyz_mm`` (mm, right_camera_optical frame).
    """
    if not isinstance(payload, dict):
        return None
    right = payload.get("right")
    if not isinstance(right, dict):
        return None
    p_mm = right.get("center_xyz_mm")
    if not p_mm or len(p_mm) < 3:
        return None
    import numpy as np

    return np.array([float(p_mm[0]), float(p_mm[1]), float(p_mm[2])], dtype=float) / 1000.0


def _stereo_center_m(payload: dict[str, Any]):
    """Stereo-triangulation center point (meters, left_camera_optical) from a payload.

    The YOLO service returns ``payload["stereo"]`` with ``available`` and its own
    ``center_xyz_mm`` (mm, left_camera_optical), computed by stereo triangulation
    rather than mono-PnP. None unless stereo is available with a valid center.
    """
    if not isinstance(payload, dict):
        return None
    stereo = payload.get("stereo")
    if not isinstance(stereo, dict) or not stereo.get("available"):
        return None
    p_mm = stereo.get("center_xyz_mm")
    if not p_mm or len(p_mm) < 3:
        return None
    import numpy as np

    return np.array([float(p_mm[0]), float(p_mm[1]), float(p_mm[2])], dtype=float) / 1000.0


def _stereo_plane_center_m(payload: dict[str, Any]):
    """Stereo+mask feature-matching center (meters, left_camera_optical) from a payload.

    The YOLO service returns ``payload["stereo_plane"]`` (method
    ``feature_epipolar_ransac``) with ``available`` and its own ``center_xyz_mm``
    (mm, left_camera_optical), computed from in-mask feature matches with epipolar
    RANSAC. None unless available with a valid center.
    """
    if not isinstance(payload, dict):
        return None
    sp = payload.get("stereo_plane")
    if not isinstance(sp, dict) or not sp.get("available"):
        return None
    p_mm = sp.get("center_xyz_mm")
    if not p_mm or len(p_mm) < 3:
        return None
    import numpy as np

    return np.array([float(p_mm[0]), float(p_mm[1]), float(p_mm[2])], dtype=float) / 1000.0


def _to_base(T, p_opt):
    import numpy as np

    return (T @ np.array([p_opt[0], p_opt[1], p_opt[2], 1.0], dtype=float))[:3]


def _inject_base_coords(
    pose: dict[str, Any],
    payload_old: dict[str, Any] | None = None,
    payload_new_dist: dict[str, Any] | None = None,
) -> None:
    """Base-frame coordinates for the intrinsics x extrinsics comparison.

    Only the YOLO 2D detection is held fixed; we vary (a) the YOLO mono-PnP
    intrinsics -> the optical 3D point, (b) the eye->base transform, and (c)
    whether the service undistorts with our distortion coeffs:
      * c_old_old_m      : OLD intrinsics point + nominal URDF extrinsics  (老内参+老外参)
      * c_old_new_m      : OLD intrinsics point + OUR hand-eye extrinsics   (老内参+新外参)
      * c_new_new_m      : NEW intrinsics point + OUR hand-eye extrinsics   (新内参+新外参)
      * c_new_old_m      : NEW intrinsics point + nominal URDF extrinsics   (新内参+老外参)
      * c_new_new_dist_m : NEW intrinsics + distortion + OUR hand-eye       (新内参+新外参+畸变校正)
    Plus the RIGHT eye (right_camera_optical -> base via right hand-eye):
      * c_right_new_m      : RIGHT NEW intrinsics + RIGHT hand-eye           (右眼 新内+新外)
      * c_right_new_dist_m : RIGHT NEW intrinsics + distortion + RIGHT hand-eye (右眼 新内+新外+畸变)
    Plus STEREO triangulation (left_camera_optical -> base via LEFT hand-eye):
      * c_stereo_m         : stereo-triangulation center + LEFT hand-eye    (双目深度)
      * c_stereo_plane_m   : stereo+mask feature-match center + LEFT hand-eye (双目深度+mask内特征匹配)
    ``pose`` carries the NEW-intrinsics payload (dist=0); ``payload_old`` the OLD
    one (dist=0); ``payload_new_dist`` the NEW intrinsics WITH distortion coeffs.
    The right-eye no-dist point rides on ``pose`` (right dist=0); the right-eye
    distortion-corrected point rides on ``payload_new_dist`` (right dist on).
    delta_ex_mm = ①->② (extrinsic effect); delta_in_mm = ②->③ (intrinsic effect);
    delta_dist_mm = ③->⑤ (left distortion effect);
    delta_right_dist_mm = ⑥->⑦ (right distortion effect).
    """
    if not isinstance(pose, dict):
        return
    T_he = OUR.get("T")          # our LEFT hand-eye, optical -> base (4x4)
    T_urdf = OUR.get("T_urdf")   # nominal URDF, optical -> base (4x4)
    T_right = OUR.get("T_right") # our RIGHT hand-eye, optical -> base (4x4)
    if T_he is None and T_urdf is None and T_right is None:
        return
    try:
        import numpy as np

        p_new = _optical_center_m(pose)
        p_old = _optical_center_m(payload_old) if payload_old is not None else None
        p_new_dist = _optical_center_m(payload_new_dist) if payload_new_dist is not None else None
        p_right = _optical_center_right_m(pose)
        p_right_dist = _optical_center_right_m(payload_new_dist) if payload_new_dist is not None else None
        p_stereo = _stereo_center_m(pose)
        p_stereo_plane = _stereo_plane_center_m(pose)

        out: dict[str, Any] = {"ok": True, "intrinsics_old": OLD_INTRINSICS, "intrinsics_new": NEW_INTRINSICS,
                               "intrinsics_new_right": NEW_INTRINSICS_RIGHT,
                               "dist_coeffs_new": [round(float(v), 6) for v in NEW_DIST],
                               "dist_coeffs_new_right": [round(float(v), 6) for v in NEW_DIST_RIGHT]}
        if p_old is not None:
            out["p_old_optical_mm"] = [round(float(v) * 1000.0, 1) for v in p_old]
        if p_new is not None:
            out["p_new_optical_mm"] = [round(float(v) * 1000.0, 1) for v in p_new]
        if p_new_dist is not None:
            out["p_new_dist_optical_mm"] = [round(float(v) * 1000.0, 1) for v in p_new_dist]
        if T_urdf is not None:
            out["cam_old_ex_m"] = [round(float(v), 4) for v in T_urdf[:3, 3]]
        if T_he is not None:
            out["cam_new_ex_m"] = [round(float(v), 4) for v in T_he[:3, 3]]
        if T_right is not None:
            out["cam_right_ex_m"] = [round(float(v), 4) for v in T_right[:3, 3]]

        c_old_old = c_old_new = c_new_new = c_new_old = c_new_new_dist = None
        c_right_new = c_right_new_dist = c_stereo = c_stereo_plane = None
        if p_old is not None and T_urdf is not None:
            c_old_old = _to_base(T_urdf, p_old)
            out["c_old_old_m"] = [round(float(v), 4) for v in c_old_old]
        if p_old is not None and T_he is not None:
            c_old_new = _to_base(T_he, p_old)
            out["c_old_new_m"] = [round(float(v), 4) for v in c_old_new]
        if p_new is not None and T_he is not None:
            c_new_new = _to_base(T_he, p_new)
            out["c_new_new_m"] = [round(float(v), 4) for v in c_new_new]
        if p_new is not None and T_urdf is not None:
            c_new_old = _to_base(T_urdf, p_new)
            out["c_new_old_m"] = [round(float(v), 4) for v in c_new_old]
        # ⑤ 新内参 + 新外参 + 畸变校正: NEW-intrinsics-undistorted point + our hand-eye.
        if p_new_dist is not None and T_he is not None:
            c_new_new_dist = _to_base(T_he, p_new_dist)
            out["c_new_new_dist_m"] = [round(float(v), 4) for v in c_new_new_dist]
        # ⑥ 右眼 新内参 + 新外参: RIGHT NEW-intrinsics point (dist=0) + RIGHT hand-eye.
        if p_right is not None and T_right is not None:
            c_right_new = _to_base(T_right, p_right)
            out["c_right_new_m"] = [round(float(v), 4) for v in c_right_new]
        # ⑦ 右眼 新内参 + 新外参 + 畸变校正: RIGHT NEW-intrinsics undistorted + RIGHT hand-eye.
        if p_right_dist is not None and T_right is not None:
            c_right_new_dist = _to_base(T_right, p_right_dist)
            out["c_right_new_dist_m"] = [round(float(v), 4) for v in c_right_new_dist]
        # ⑧ 双目深度: stereo-triangulation center (left optical) + LEFT hand-eye.
        if p_stereo is not None and T_he is not None:
            c_stereo = _to_base(T_he, p_stereo)
            out["c_stereo_m"] = [round(float(v), 4) for v in c_stereo]
            stereo = pose.get("stereo") if isinstance(pose, dict) else None
            if isinstance(stereo, dict):
                out["stereo_reproj_px"] = stereo.get("stereo_reprojection_error_px")
                out["stereo_baseline_mm"] = stereo.get("baseline_mm")
        # ⑨ 双目深度+mask内特征匹配: feature_epipolar_ransac center + LEFT hand-eye.
        if p_stereo_plane is not None and T_he is not None:
            c_stereo_plane = _to_base(T_he, p_stereo_plane)
            out["c_stereo_plane_m"] = [round(float(v), 4) for v in c_stereo_plane]
            sp = pose.get("stereo_plane") if isinstance(pose, dict) else None
            if isinstance(sp, dict):
                out["stereo_plane_epi_rms_px"] = sp.get("epipolar_rms_px")
                out["stereo_plane_rms_mm"] = sp.get("plane_rms_mm")
                out["stereo_plane_inliers"] = sp.get("num_inliers")

        if c_old_old is not None and c_old_new is not None:
            out["delta_ex_mm"] = round(float(np.linalg.norm(c_old_old - c_old_new)) * 1000.0, 1)
        if c_old_new is not None and c_new_new is not None:
            out["delta_in_mm"] = round(float(np.linalg.norm(c_old_new - c_new_new)) * 1000.0, 1)
        if c_new_new is not None and c_new_new_dist is not None:
            out["delta_dist_mm"] = round(float(np.linalg.norm(c_new_new - c_new_new_dist)) * 1000.0, 1)
        if c_right_new is not None and c_right_new_dist is not None:
            out["delta_right_dist_mm"] = round(float(np.linalg.norm(c_right_new - c_right_new_dist)) * 1000.0, 1)
        # 双目 vs ③ 左眼单目(新内+新外) 的 base 系差异。
        if c_stereo is not None and c_new_new is not None:
            out["delta_stereo_mm"] = round(float(np.linalg.norm(c_stereo - c_new_new)) * 1000.0, 1)
        # 双目+mask特征匹配 vs ③ 左眼单目 的 base 系差异。
        if c_stereo_plane is not None and c_new_new is not None:
            out["delta_stereo_plane_mm"] = round(float(np.linalg.norm(c_stereo_plane - c_new_new)) * 1000.0, 1)
        pose["base_coords"] = out
    except (Exception, SystemExit) as exc:
        pose["base_coords"] = {"ok": False, "error": str(exc)}


def _inject_our_method(pose: dict[str, Any]) -> None:
    """Recompute box pose with OUR method and attach pose['our_method']."""
    if not OUR.get("enabled") or not isinstance(pose, dict):
        return
    try:
        import numpy as np
        import cv2

        core = OUR["core"]
        viz = pose.get("g1d_visualization") or {}
        viz_yolo = viz.get("yolo") or {}
        viz_box = viz.get("box") or {}
        # points_px location varies by server: top-level (optical_api) or nested
        # under g1d_visualization.yolo (yolo_server). Accept either.
        pts = pose.get("points_px") or viz_yolo.get("points_px")
        if not pts:
            return
        points_px = np.asarray(pts, dtype=float).reshape(4, 2)
        label = (pose.get("selected_yolo_label") or pose.get("requested_yolo_label")
                 or viz_yolo.get("label"))
        size = pose.get("object_top_size_mm") or viz_box.get("object_top_size_mm")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            long_m = max(float(size[0]), float(size[1])) / 1000.0
            short_m = min(float(size[0]), float(size[1])) / 1000.0
        elif label in core.CLASS_TOP_SIZES_M:
            long_m, short_m = core.CLASS_TOP_SIZES_M[label]
        else:
            pose["our_method"] = {"ok": False, "error": f"no known size for label {label!r}"}
            return

        K, dist, T = OUR["K"], OUR["dist"], OUR["T"]
        center_cam, info = core.estimate_center_in_camera(points_px, K, dist, long_m, short_m)
        R, t = T[:3, :3], T[:3, 3]
        center_base = R @ center_cam + t
        above_base = center_base + np.array([0.0, 0.0, 0.1])

        # Full box pose from PnP. Rc maps object->camera; object axes are
        # (X=long, Y=short, Z=top-normal) when long_along_x, else X/Y swapped.
        Rc, _ = cv2.Rodrigues(info["rvec"].reshape(3, 1))
        if info["long_along_x"]:
            long_axis_cam, normal_cam = Rc[:, 0], Rc[:, 2]
        else:
            long_axis_cam, normal_cam = Rc[:, 1], Rc[:, 2]

        def _unit(v):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-12 else v

        # Build an orthonormal box basis in the base frame:
        #   x = long axis, z = top-face normal, y = z cross x (right-handed).
        # PnP of a planar quad leaves the normal sign ambiguous, so force it to
        # point UP (+Z torso). center_cam is the top-face center, so a downward
        # normal would make the viewer draw the box body above the center (it
        # looks like the center sits on the box's bottom face). Flipping keeps
        # the box hanging below its top-face center, where suction approaches.
        x_base = _unit(R @ long_axis_cam)
        z_base = _unit(R @ normal_cam)
        if z_base[2] < 0.0:
            z_base = -z_base
        y_base = _unit(np.cross(z_base, x_base))
        z_base = _unit(np.cross(x_base, y_base))
        long_axis_base = x_base
        planar = float(np.hypot(x_base[0], x_base[1]))
        yaw = float(np.arctan2(x_base[1], x_base[0])) if planar > 1e-9 else 0.0

        pose["our_method"] = {
            "ok": True,
            "label": label,
            "orientation": "long_along_x" if info["long_along_x"] else "short_along_x",
            "reproj_px": round(float(info["reproj_px"]), 2),
            "range_mm": round(float(np.linalg.norm(center_cam)) * 1000.0, 1),
            "center_cam_mm": [round(float(v) * 1000.0, 1) for v in center_cam],
            "center_base_m": [round(float(v), 4) for v in center_base],
            "above_base_m": [round(float(v), 4) for v in above_base],
            "camera_offset_m": [round(float(v), 4) for v in t],
            "box_long_axis_base": [round(float(v), 5) for v in long_axis_base],
            "box_yaw_base_rad": round(yaw, 5),
            "box_axes_base": {
                "x": [round(float(v), 5) for v in x_base],
                "y": [round(float(v), 5) for v in y_base],
                "z": [round(float(v), 5) for v in z_base],
            },
            "box_dims_m": [round(long_m, 4), round(short_m, 4)],
        }

        # Same camera-frame point, but mapped to base via the NOMINAL URDF
        # extrinsics (pre-calibration). Lets the viewer show base-frame
        # coordinates "before vs after" hand-eye calibration side by side.
        T_urdf = OUR.get("T_urdf")
        if T_urdf is not None:
            cc_h = np.array([center_cam[0], center_cam[1], center_cam[2], 1.0])
            center_base_urdf = (T_urdf @ cc_h)[:3]
            pose["our_method"]["center_base_urdf_m"] = [round(float(v), 4) for v in center_base_urdf]
            pose["our_method"]["camera_offset_urdf_m"] = [round(float(v), 4) for v in T_urdf[:3, 3]]
            pose["our_method"]["delta_urdf_vs_handeye_mm"] = round(
                float(np.linalg.norm(center_base_urdf - center_base)) * 1000.0, 1
            )
    except (Exception, SystemExit) as exc:
        pose["our_method"] = {"ok": False, "error": str(exc)}

G1D_LOWSTATE_JOINT_MAP: dict[int, str] = {
    12: "torso_Joint",
    14: "Yaw_Joint",
    15: "left_shoulder_pitch_joint",
    16: "left_shoulder_roll_joint",
    17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint",
    19: "left_wrist_roll_joint",
    20: "left_wrist_pitch_joint",
    21: "left_wrist_yaw_joint",
    22: "right_shoulder_pitch_joint",
    23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint",
    26: "right_wrist_roll_joint",
    27: "right_wrist_pitch_joint",
    28: "right_wrist_yaw_joint",
}


# Right-arm joints in the IK/URDF chain order (matches PickPlan.joint_names).
PICK_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _arm_joints_from_state(state: dict[str, Any]) -> list[float] | None:
    """Pull the 7 right-arm joints (chain order) out of a robot_state dict.

    Returns None unless all 7 are present, so a partial/placeholder state never
    silently seeds the planner with a wrong pose.
    """
    joints: dict[str, float] = {}
    js = state.get("joint_states") if isinstance(state, dict) else None
    if isinstance(js, dict) and isinstance(js.get("name"), list) and isinstance(js.get("position"), list):
        for name, pos in zip(js["name"], js["position"]):
            joints[str(name)] = pos
    elif isinstance(state, dict) and isinstance(state.get("joints"), dict):
        joints = state["joints"]
    out: list[float] = []
    for name in PICK_ARM_JOINTS:
        val = joints.get(name)
        if not isinstance(val, (int, float)):
            return None
        out.append(float(val))
    return out


def _all_joints_from_state(state: dict[str, Any]) -> dict[str, float]:
    """Flatten any joint snapshot (joint_states arrays or joints map) to a dict."""
    joints: dict[str, float] = {}
    js = state.get("joint_states") if isinstance(state, dict) else None
    if isinstance(js, dict) and isinstance(js.get("name"), list) and isinstance(js.get("position"), list):
        for name, pos in zip(js["name"], js["position"]):
            if isinstance(pos, (int, float)):
                joints[str(name)] = float(pos)
    elif isinstance(state, dict) and isinstance(state.get("joints"), dict):
        for name, pos in state["joints"].items():
            if isinstance(pos, (int, float)):
                joints[str(name)] = float(pos)
    return joints


def _ready_pose_path() -> Path:
    """File where the human-set, table-safe ready pose is stored."""
    scripts_dir = Path(__file__).resolve().parents[3]  # .../scripts
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from pick import READY_POSE_PATH  # centralized so the planner reads the same file
        return Path(READY_POSE_PATH)
    except Exception:
        return scripts_dir / "pick" / "ready_pose.json"


def _save_ready_pose(body: dict[str, Any], read_state) -> dict[str, Any]:
    """Persist the current 7 right-arm joints as the table-safe ready pose.

    Prefers the front-end's live joints (``q_current``); if absent or incomplete,
    reads the server's own live joint state (same DDS source as /api/robot_state).
    """
    qc = body.get("q_current")
    if isinstance(qc, list) and len(qc) == len(PICK_ARM_JOINTS) and all(isinstance(v, (int, float)) for v in qc):
        q = [float(v) for v in qc]
        source = "frontend_live_joints"
    else:
        state = read_state()
        live = _arm_joints_from_state(state)
        if live is None:
            raise RuntimeError(
                "无法获取机械臂关节:前端未提供 q_current,且 DDS 实时状态也不可用。"
                "请确认机械臂在线后重试。"
            )
        q = live
        source = "server_live_joints"

    all_joints = body.get("all_joints")
    all_joints = {str(k): float(v) for k, v in all_joints.items()
                  if isinstance(v, (int, float))} if isinstance(all_joints, dict) else {}

    column = body.get("column_extension_mm")
    column = float(column) if isinstance(column, (int, float)) else None

    payload: dict[str, Any] = {
        "ok": True,
        "name": str(body.get("name") or "ready_pose"),
        "joint_names": list(PICK_ARM_JOINTS),
        "q": q,
        "column_extension_mm": column,
        "all_joints": all_joints,
        "source": source,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = _ready_pose_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["path"] = str(path)
    return payload


def _load_ready_pose() -> dict[str, Any]:
    """Read back the saved ready pose; raises if none has been saved yet."""
    path = _ready_pose_path()
    if not path.exists():
        raise RuntimeError("尚未保存预备位姿(先抬臂到理想区域再点保存)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ok"] = True
    payload["path"] = str(path)
    return payload


def _run_pick_pipeline(
    *,
    execute: bool,
    body: dict[str, Any],
    viewer_dir: Path,
    arm_network_interface: str | None,
    suction_host: str | None,
) -> dict[str, Any]:
    """Run our suction-pick pipeline: perceive -> plan -> (export | execute).

    Used by the web buttons (模拟执行 / 真机执行). Returns the trajectory dict so
    the page can play it, and always writes pick_trajectory.json into the viewer.
    """
    scripts_dir = Path(__file__).resolve().parents[3]  # .../scripts
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import numpy as np
    from pick.planner import build_robot_and_ik, plan_pick, load_ready_pose, NEUTRAL_RIGHT_ARM_Q
    from pick.perception import perceive
    from pick.trajectory import build_trajectory, to_dict, export_json
    from pick import executor_iface

    def fnum(key: str, default: float) -> float:
        try:
            return float(body[key])
        except (KeyError, TypeError, ValueError):
            return default

    label = body.get("label") or None
    model, ik = build_robot_and_ik()
    box = perceive(label=label)

    qc = body.get("q_current")
    if isinstance(qc, list) and len(qc) == model.n and all(isinstance(v, (int, float)) for v in qc):
        q_current = np.asarray(qc, dtype=float)
        seed_source = body.get("_seed_origin") or "frontend_live_joints"
    elif model.n == NEUTRAL_RIGHT_ARM_Q.size:
        # No live joints (e.g. DDS off): seed from a natural arm pose, not zeros.
        q_current = NEUTRAL_RIGHT_ARM_Q.copy()
        seed_source = "neutral_fallback"
    else:
        q_current = np.zeros(model.n)
        seed_source = "zeros"

    # Table-safe ready pose saved from the web UI; if present and the right size,
    # the arm passes through it first and the pregrasp IK is seeded from it.
    q_ready = load_ready_pose(expected_n=model.n)

    plan = plan_pick(
        box,
        q_current,
        ik,
        q_ready=q_ready,
        pregrasp_dist=fnum("pregrasp_dist", 0.06),
        lift_dist=fnum("lift_dist", 0.10),
        grasp_offset=fnum("grasp_offset", 0.0),
        use_box_normal=not bool(body.get("no_box_normal")),
        enforce_above_target=not bool(body.get("no_above_target")),
        above_target_margin=fnum("above_target_margin", 0.0),
    )
    timing = dict(
        to_ready_s=fnum("to_ready_s", 2.5),
        to_pregrasp_s=fnum("to_pregrasp_s", 2.5),
        approach_s=fnum("approach_s", 2.0),
        lift_s=fnum("lift_s", 2.0),
        grasp_dwell_s=fnum("grasp_dwell_s", 0.8),
        settle_s=fnum("settle_s", 0.4),
        max_joint_speed=fnum("max_joint_speed", 0.4),
        suction=not bool(body.get("no_suction")),
    )
    traj = build_trajectory(plan, **timing)
    export_json(traj, str(viewer_dir / "pick_trajectory.json"))

    result: dict[str, Any] = {
        "ok": True,
        "executed": False,
        "seed_source": seed_source,
        "box": box.summary(),
        "diagnostics": plan.diagnostics,
        "trajectory": to_dict(traj),
    }

    if execute:
        executor = executor_iface.make_executor(network_interface=arm_network_interface or None)
        run_kwargs = dict(timing)
        if suction_host:
            run_kwargs["suction_host"] = suction_host
        executor_iface.run_plan(executor, plan, **run_kwargs)
        result["executed"] = True

    return result


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


def _post_json(url: str, body: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=float(timeout_sec)) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _pose_url_from(source_url: str) -> str:
    """Derive the POST /pose endpoint from a configured /xyz source URL."""
    parsed = urllib.parse.urlparse(source_url)
    path = parsed.path or ""
    if path.rstrip("/").endswith("/xyz"):
        path = path.rstrip("/")[: -len("/xyz")] + "/pose"
    elif not path.rstrip("/").endswith("/pose"):
        path = path.rstrip("/") + "/pose"
    return urllib.parse.urlunparse(parsed._replace(path=path, query=""))


def _fetch_xyz_or_pose(source_url: str, pose_url: str, label: str,
                       intr: dict[str, float], dist: list[float],
                       intr_right: dict[str, float], dist_right: list[float],
                       timeout_sec: float) -> dict[str, Any]:
    """Fetch one combo. Prefer GET /xyz (lightweight compact output, which now
    carries the top-level ``right`` block too); fall back to POST /pose (full
    output, also has ``right``) only if /xyz fails.

    Both eyes' intrinsics + distortion are passed so the service returns the left
    point (top level) and the right point (``right`` block) under our calibration.
    """
    query = {
        "label": label,
        **{k: str(v) for k, v in intr.items()},
        "dist_coeffs": ",".join(str(v) for v in dist),
        **{f"{k}_right": str(v) for k, v in intr_right.items()},
        "dist_coeffs_right": ",".join(str(v) for v in dist_right),
    }
    try:
        return _fetch_json(_merge_query(source_url, query), timeout_sec)
    except Exception:
        body: dict[str, Any] = {**intr, "dist_coeffs": list(dist),
                                **{f"{k}_right": v for k, v in intr_right.items()},
                                "dist_coeffs_right": list(dist_right)}
        if label:
            body["label"] = label
        return _post_json(pose_url, body, timeout_sec)


def _fetch_raw_xyz(source_url: str, label: str, timeout_sec: float) -> dict[str, Any]:
    query = {"label": label} if label else {}
    return _fetch_json(_merge_query(source_url, query), timeout_sec)


def _overlay_raw_stereo_payload(payload: dict[str, Any], raw_payload: dict[str, Any] | None) -> None:
    """Keep raw stereo outputs while mono-PnP comparison requests vary intrinsics."""
    if not isinstance(payload, dict) or not isinstance(raw_payload, dict):
        return
    overlay: dict[str, str] = {}
    for key in ("stereo", "stereo_plane"):
        value = raw_payload.get(key)
        if isinstance(value, dict) and value.get("available") and value.get("center_xyz_mm"):
            payload[key] = json.loads(json.dumps(value))
            overlay[key] = "raw_xyz"
    right = raw_payload.get("right")
    current_right = payload.get("right")
    if (not isinstance(current_right, dict) or not current_right.get("center_xyz_mm")) and isinstance(right, dict):
        payload["right"] = json.loads(json.dumps(right))
        overlay["right"] = "raw_xyz_fallback"
    if overlay:
        payload["visualizer_raw_overlay"] = overlay


def _read_member(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    if callable(value):
        return value()
    return value


def _map_column_raw_to_visual_m(raw_height_m: float) -> tuple[float, float]:
    raw_mm = float(raw_height_m) * 1000.0
    span = float(COLUMN_RAW_MAX_MM) - float(COLUMN_RAW_MIN_MM)
    visual_max_mm = max(0.0, float(COLUMN_VISUAL_MAX_MM))
    if abs(span) < 1e-9:
        visual_mm = max(0.0, min(visual_max_mm, raw_mm))
    else:
        ratio = (raw_mm - float(COLUMN_RAW_MIN_MM)) / span
        visual_mm = max(0.0, min(1.0, ratio)) * visual_max_mm
    return visual_mm / 1000.0, raw_mm


def _round_float(value: Any, ndigits: int = 6) -> float | None:
    try:
        return round(float(value), ndigits)
    except Exception:
        return None


class _DdsStateReader:
    def __init__(
        self,
        *,
        network_interface: str,
        lowstate_topic: str,
        hispeed_topic: str,
        sdk2py_path: str,
    ) -> None:
        if sdk2py_path and sdk2py_path not in sys.path and Path(sdk2py_path).exists():
            sys.path.insert(0, sdk2py_path)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self.network_interface = network_interface
        self.lowstate_topic = lowstate_topic
        self.hispeed_topic = hispeed_topic
        self.sdk2py_path = sdk2py_path
        self.lock = threading.Lock()
        self.lowstate_msg: Any = None
        self.hispeed_msg: Any = None
        self.lowstate_updated_at = 0.0
        self.hispeed_updated_at = 0.0

        ChannelFactoryInitialize(0, network_interface)
        self.lowstate_subscriber = ChannelSubscriber(lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self._on_lowstate, 10)
        self.hispeed_subscriber = ChannelSubscriber(hispeed_topic, Point32_)
        self.hispeed_subscriber.Init(self._on_hispeed, 10)

    def _on_lowstate(self, msg: Any) -> None:
        with self.lock:
            self.lowstate_msg = msg
            self.lowstate_updated_at = time.time()

    def _on_hispeed(self, msg: Any) -> None:
        with self.lock:
            self.hispeed_msg = msg
            self.hispeed_updated_at = time.time()

    def state(self, timeout_sec: float) -> dict[str, Any]:
        deadline = time.time() + max(0.05, float(timeout_sec))
        while time.time() < deadline:
            with self.lock:
                lowstate = self.lowstate_msg
                hispeed = self.hispeed_msg
                lowstate_time = self.lowstate_updated_at
                hispeed_time = self.hispeed_updated_at
            if lowstate is not None:
                return self._build_state(lowstate, hispeed, lowstate_time, hispeed_time)
            time.sleep(0.01)
        raise RuntimeError(f"DDS lowstate timeout on {self.lowstate_topic}")

    def _build_state(self, lowstate: Any, hispeed: Any, lowstate_time: float, hispeed_time: float) -> dict[str, Any]:
        motor_state = list(_read_member(lowstate, "motor_state", []) or [])
        joints: dict[str, float] = {}
        joint_states_name: list[str] = []
        joint_states_position: list[float] = []
        joint_states_velocity: list[float] = []
        joint_states_effort: list[float] = []
        raw_motor_states: list[dict[str, Any]] = []

        for index, state in enumerate(motor_state):
            q = _round_float(_read_member(state, "q"))
            dq = _round_float(_read_member(state, "dq"))
            tau_est = _round_float(_read_member(state, "tau_est"))
            raw_motor_states.append({"index": index, "q": q, "dq": dq, "tau_est": tau_est})
            joint_name = G1D_LOWSTATE_JOINT_MAP.get(index)
            if not joint_name or q is None:
                continue
            joints[joint_name] = q
            joint_states_name.append(joint_name)
            joint_states_position.append(q)
            joint_states_velocity.append(dq if dq is not None else 0.0)
            joint_states_effort.append(tau_est if tau_est is not None else 0.0)

        column_raw_m = _round_float(_read_member(hispeed, "y")) if hispeed is not None else None
        column_height_m = None
        column_raw_mm = None
        if column_raw_m is not None:
            column_height_m, column_raw_mm = _map_column_raw_to_visual_m(column_raw_m)
            per_joint = column_height_m / 2.0
            joints["LZ_mt_Joint"] = per_joint
            joints["LZ_it_Joint"] = per_joint
            for name in ("LZ_mt_Joint", "LZ_it_Joint"):
                joint_states_name.append(name)
                joint_states_position.append(per_joint)
                joint_states_velocity.append(0.0)
                joint_states_effort.append(0.0)

        imu_state = _read_member(lowstate, "imu_state")
        imu_payload: dict[str, Any] = {}
        if imu_state is not None:
            for key in ("rpy", "gyroscope", "accelerometer", "quaternion"):
                values = _read_member(imu_state, key)
                if values is not None:
                    imu_payload[key] = [_round_float(value) for value in list(values)]

        return {
            "ok": True,
            "source": "unitree_dds_lowstate",
            "dds": {
                "network_interface": self.network_interface,
                "lowstate_topic": self.lowstate_topic,
                "hispeed_topic": self.hispeed_topic,
                "lowstate_age_ms": round((time.time() - lowstate_time) * 1000.0, 1) if lowstate_time else None,
                "hispeed_age_ms": round((time.time() - hispeed_time) * 1000.0, 1) if hispeed_time else None,
                "motor_count": len(motor_state),
            },
            "column_extension_mm": round(column_height_m * 1000.0, 1) if column_height_m is not None else None,
            "column_raw_extension_mm": round(column_raw_mm, 1) if column_raw_mm is not None else None,
            "column_raw_range_mm": [float(COLUMN_RAW_MIN_MM), float(COLUMN_RAW_MAX_MM)],
            "column_visual_max_mm": float(COLUMN_VISUAL_MAX_MM),
            "joints": joints,
            "joint_states": {
                "name": joint_states_name,
                "position": joint_states_position,
                "velocity": joint_states_velocity,
                "effort": joint_states_effort,
            },
            "imu": imu_payload,
            "raw_motor_states": raw_motor_states,
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }


def _read_dds_robot_state(
    *,
    network_interface: str,
    lowstate_topic: str,
    hispeed_topic: str,
    sdk2py_path: str,
    timeout_sec: float,
) -> dict[str, Any]:
    global DDS_READER
    if DDS_READER is None:
        DDS_READER = _DdsStateReader(
            network_interface=network_interface,
            lowstate_topic=lowstate_topic,
            hispeed_topic=hispeed_topic,
            sdk2py_path=sdk2py_path,
        )
    return DDS_READER.state(timeout_sec)


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
    dds_interface: str | None,
    dds_lowstate_topic: str | None,
    dds_hispeed_topic: str | None,
    unitree_sdk2py_path: str,
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
    if dds_interface and dds_lowstate_topic:
        try:
            return _read_dds_robot_state(
                network_interface=dds_interface,
                lowstate_topic=dds_lowstate_topic,
                hispeed_topic=dds_hispeed_topic or DEFAULT_DDS_HISPEED_TOPIC,
                sdk2py_path=unitree_sdk2py_path,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            return _default_robot_state(f"DDS state unavailable: {exc}")
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
    dds_interface: str | None,
    dds_lowstate_topic: str | None,
    dds_hispeed_topic: str | None,
    unitree_sdk2py_path: str,
    joint_states_topic: str | None,
    arm_network_interface: str | None = None,
    suction_host: str | None = None,
) -> type[SimpleHTTPRequestHandler]:
    class VisualizerHandler(SimpleHTTPRequestHandler):
        server_version = "G1DCigaretteVisualizer/1.0"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.end_headers()

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in ("/api/plan", "/api/execute", "/api/save_ready_pose"):
                _json_response(self, 404, {"ok": False, "error": "unknown endpoint"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                body = {}
            if not isinstance(body, dict):
                body = {}

            if parsed.path == "/api/save_ready_pose":
                def _read() -> dict[str, Any]:
                    return _read_robot_state(
                        robot_state_url,
                        robot_state_file,
                        dds_interface,
                        dds_lowstate_topic,
                        dds_hispeed_topic,
                        unitree_sdk2py_path,
                        joint_states_topic,
                        timeout_sec,
                    )
                try:
                    result = _save_ready_pose(body, _read)
                    _json_response(self, 200, result)
                except Exception as exc:  # noqa: BLE001
                    _json_response(self, 500, {"ok": False, "error": str(exc)})
                return

            execute = parsed.path == "/api/execute"
            # Seed the plan from the real arm: prefer the front-end's live joints;
            # if absent, read the server's own live joint state (same DDS source as
            # /api/robot_state) so the simulation also starts at the real pose.
            if not isinstance(body.get("q_current"), list):
                try:
                    state = _read_robot_state(
                        robot_state_url,
                        robot_state_file,
                        dds_interface,
                        dds_lowstate_topic,
                        dds_hispeed_topic,
                        unitree_sdk2py_path,
                        joint_states_topic,
                        timeout_sec,
                    )
                    live = _arm_joints_from_state(state)
                    if live is not None:
                        body["q_current"] = live
                        body["_seed_origin"] = "server_live_joints"
                except Exception:  # noqa: BLE001
                    pass
            try:
                result = _run_pick_pipeline(
                    execute=execute,
                    body=body if isinstance(body, dict) else {},
                    viewer_dir=VIEWER_DIR,
                    arm_network_interface=arm_network_interface,
                    suction_host=suction_host,
                )
                _json_response(self, 200, result)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                payload = {"ok": False, "error": str(exc)}
                diag = getattr(exc, "diagnostics", None)
                if diag is not None:
                    payload["diagnostics"] = diag
                _json_response(self, 500, payload)

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
                        "dds_interface": dds_interface,
                        "dds_lowstate_topic": dds_lowstate_topic,
                        "dds_hispeed_topic": dds_hispeed_topic,
                        "unitree_sdk2py_path": unitree_sdk2py_path,
                        "joint_states_topic": joint_states_topic,
                    },
                )
                return
            if parsed.path == "/api/xyz":
                query = urllib.parse.parse_qs(parsed.query)
                source_url = query.get("url", [default_xyz_url])[-1]
                label = query.get("label", [""])[-1]
                # Our own PnP (4th box) is injected ONLY when the frontend opts in.
                use_our = query.get("our", ["0"])[-1].strip().lower() in ("1", "true", "yes", "on")
                # Fetch GET /xyz three times (lightweight compact output, which now
                # also carries the top-level "right" block). The service's mono-PnP
                # 3D point depends on the intrinsics AND distortion coeffs, so each
                # fetch varies them:
                #   payload          left NEW dist=0,  right NEW dist=0  -> ③ left, ⑥ right
                #   payload_old      left OLD dist=0,  right NEW dist=0  -> ① / ②
                #   payload_new_dist left NEW left-dist, right NEW right-dist -> ⑤ left, ⑦ right
                # Non-corrected combos send dist=0 so the service does NOT undistort.
                # Each fetch falls back to POST /pose if /xyz is unavailable.
                pose_url = _pose_url_from(source_url)
                ir = NEW_INTRINSICS_RIGHT

                def _combo(left_intr, left_dist, right_dist):
                    return _fetch_xyz_or_pose(source_url, pose_url, label,
                                              left_intr, left_dist, ir, right_dist, timeout_sec)

                try:
                    try:
                        payload_raw = _fetch_raw_xyz(source_url, label, timeout_sec)
                    except Exception:
                        payload_raw = None
                    payload = _combo(NEW_INTRINSICS, ZERO_DIST, ZERO_DIST)
                    try:
                        payload_old = _combo(OLD_INTRINSICS, ZERO_DIST, ZERO_DIST)
                    except Exception:
                        payload_old = None
                    try:
                        payload_new_dist = _combo(NEW_INTRINSICS, NEW_DIST, NEW_DIST_RIGHT)
                    except Exception:
                        payload_new_dist = None
                    _overlay_raw_stereo_payload(payload, payload_raw)
                    _inject_base_coords(payload, payload_old, payload_new_dist)
                    if use_our:
                        _inject_our_method(payload)
                    _json_response(self, 200 if payload.get("ok", True) else 502, {"ok": True, "url": source_url, "pose": payload})
                except Exception as exc:
                    _json_response(self, 502, {"ok": False, "url": source_url, "error": str(exc)})
                return
            if parsed.path == "/api/robot_state":
                query = urllib.parse.parse_qs(parsed.query)
                source_url = query.get("url", [robot_state_url or ""])[-1] or None
                query_dds_interface = query.get("dds_interface", [dds_interface or ""])[-1] or None
                query_dds_lowstate_topic = query.get("dds_lowstate_topic", [dds_lowstate_topic or ""])[-1] or None
                query_dds_hispeed_topic = query.get("dds_hispeed_topic", [dds_hispeed_topic or ""])[-1] or None
                topic = query.get("joint_states_topic", [joint_states_topic or ""])[-1] or None
                try:
                    payload = _read_robot_state(
                        source_url,
                        robot_state_file,
                        query_dds_interface,
                        query_dds_lowstate_topic,
                        query_dds_hispeed_topic,
                        unitree_sdk2py_path,
                        topic,
                        timeout_sec,
                    )
                    _json_response(self, 200 if payload.get("ok", True) else 502, {"ok": True, "state": payload})
                except Exception as exc:
                    _json_response(self, 502, {"ok": False, "error": str(exc)})
                return
            if parsed.path == "/api/ready_pose":
                try:
                    _json_response(self, 200, _load_ready_pose())
                except Exception as exc:  # noqa: BLE001
                    _json_response(self, 404, {"ok": False, "error": str(exc)})
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
    parser.add_argument("--dds-interface", default=DEFAULT_DDS_INTERFACE)
    parser.add_argument("--dds-lowstate-topic", default=DEFAULT_DDS_LOWSTATE_TOPIC)
    parser.add_argument("--dds-hispeed-topic", default=DEFAULT_DDS_HISPEED_TOPIC)
    parser.add_argument("--unitree-sdk2py-path", default=DEFAULT_UNITREE_SDK2PY_PATH)
    parser.add_argument("--joint-states-topic", default=DEFAULT_JOINT_STATES_TOPIC)
    parser.add_argument("--column-raw-min-mm", type=float, default=DEFAULT_COLUMN_RAW_MIN_MM)
    parser.add_argument("--column-raw-max-mm", type=float, default=DEFAULT_COLUMN_RAW_MAX_MM)
    parser.add_argument("--column-visual-max-mm", type=float, default=DEFAULT_COLUMN_VISUAL_MAX_MM)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--our-method", dest="our_method", action="store_true", default=True,
                        help="recompute box pose with OUR intrinsics + hand-eye (default on)")
    parser.add_argument("--no-our-method", dest="our_method", action="store_false",
                        help="disable our-method injection; use only the robot's own 3D")
    parser.add_argument("--new-intrinsics-file", default="",
                        help=f"calibration json/yaml for the grid's NEW intrinsics "
                             f"(default: {DEFAULT_NEW_INTRINSICS_FILE})")
    parser.add_argument("--our-intrinsics", default="",
                        help=f"our_method stereo_calibration.yaml (default: {DEFAULT_NEW_INTRINSICS_YAML})")
    parser.add_argument("--our-handeye", default="",
                        help=f"LEFT-eye handeye_result_left.json (default: {DEFAULT_HANDEYE_LEFT_FILE})")
    parser.add_argument("--our-handeye-right", default="",
                        help=f"RIGHT-eye handeye_result_right.json (default: {DEFAULT_HANDEYE_RIGHT_FILE})")
    parser.add_argument("--arm-network-interface", default="",
                        help="DDS interface for the arm executor on 真机执行 (empty = SDK default)")
    parser.add_argument("--suction-host", default="192.168.123.164",
                        help="robot host for the suction HTTP service (port 18080)")
    return parser


def main() -> int:
    global NEW_INTRINSICS, NEW_DIST, NEW_INTRINSICS_RIGHT, NEW_DIST_RIGHT
    global COLUMN_RAW_MIN_MM, COLUMN_RAW_MAX_MM, COLUMN_VISUAL_MAX_MM
    args = build_arg_parser().parse_args()
    if not VIEWER_DIR.exists():
        raise FileNotFoundError(f"viewer directory not found: {VIEWER_DIR}")

    COLUMN_RAW_MIN_MM = float(args.column_raw_min_mm)
    COLUMN_RAW_MAX_MM = float(args.column_raw_max_mm)
    COLUMN_VISUAL_MAX_MM = float(args.column_visual_max_mm)
    DEFAULT_ROBOT_STATE["column_extension_mm"] = COLUMN_VISUAL_MAX_MM
    DEFAULT_ROBOT_STATE["column_raw_range_mm"] = [COLUMN_RAW_MIN_MM, COLUMN_RAW_MAX_MM]
    DEFAULT_ROBOT_STATE["column_visual_max_mm"] = COLUMN_VISUAL_MAX_MM
    DEFAULT_ROBOT_STATE["joints"] = {
        "LZ_mt_Joint": COLUMN_VISUAL_MAX_MM / 2000.0,
        "LZ_it_Joint": COLUMN_VISUAL_MAX_MM / 2000.0,
    }

    # NEW intrinsics for the grid (?fx&fy&cx&cy) are read from a calibration file;
    # fall back to the hard-coded NEW_INTRINSICS if the file can't be read. The
    # matching distortion coeffs (only the distortion combos use them) come from the
    # same file; missing coeffs fall back to zeros (= no undistortion). Both eyes
    # are read from the same stereo calibration (left_camera / right_camera).
    new_intr_file = args.new_intrinsics_file or DEFAULT_NEW_INTRINSICS_FILE
    if new_intr_file:
        try:
            NEW_INTRINSICS = _load_left_intrinsics_file(new_intr_file)
            print(f"[intrinsics] LEFT NEW from {new_intr_file}: "
                  f"fx={NEW_INTRINSICS['fx']:.2f} fy={NEW_INTRINSICS['fy']:.2f} "
                  f"cx={NEW_INTRINSICS['cx']:.2f} cy={NEW_INTRINSICS['cy']:.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[intrinsics] LEFT NEW file unavailable ({exc}); using fallback {NEW_INTRINSICS}", flush=True)
        try:
            NEW_DIST = _load_left_dist_coeffs_file(new_intr_file)
            print(f"[dist] LEFT NEW from {new_intr_file}: "
                  f"{[round(v, 6) for v in NEW_DIST]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[dist] LEFT NEW dist unavailable ({exc}); using zeros (no undistortion)", flush=True)
        try:
            NEW_INTRINSICS_RIGHT = _load_right_intrinsics_file(new_intr_file)
            print(f"[intrinsics] RIGHT NEW from {new_intr_file}: "
                  f"fx={NEW_INTRINSICS_RIGHT['fx']:.2f} fy={NEW_INTRINSICS_RIGHT['fy']:.2f} "
                  f"cx={NEW_INTRINSICS_RIGHT['cx']:.2f} cy={NEW_INTRINSICS_RIGHT['cy']:.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[intrinsics] RIGHT NEW file unavailable ({exc}); using fallback {NEW_INTRINSICS_RIGHT}", flush=True)
        try:
            NEW_DIST_RIGHT = _load_right_dist_coeffs_file(new_intr_file)
            print(f"[dist] RIGHT NEW from {new_intr_file}: "
                  f"{[round(v, 6) for v in NEW_DIST_RIGHT]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[dist] RIGHT NEW dist unavailable ({exc}); using zeros (no undistortion)", flush=True)

    if args.our_method:
        scripts_dir = Path(__file__).resolve().parents[3]
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        # Default our_method intrinsics/hand-eye to the same calibration run as the
        # grid's NEW intrinsics, so every "new" estimate shares one calibration.
        intrinsics = args.our_intrinsics or DEFAULT_NEW_INTRINSICS_YAML
        handeye = args.our_handeye or DEFAULT_HANDEYE_LEFT_FILE
        handeye_right = args.our_handeye_right or DEFAULT_HANDEYE_RIGHT_FILE
        _load_our_method(intrinsics, handeye, handeye_right)
    server = ThreadingHTTPServer(
        (args.bind, int(args.port)),
        make_handler(
            args.xyz_url,
            args.timeout_sec,
            args.robot_state_url or None,
            args.robot_state_file,
            args.dds_interface or None,
            args.dds_lowstate_topic or None,
            args.dds_hispeed_topic or None,
            args.unitree_sdk2py_path,
            args.joint_states_topic or None,
            args.arm_network_interface or None,
            args.suction_host or None,
        ),
    )
    print(f"serving G1-D cigarette visualizer on http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
