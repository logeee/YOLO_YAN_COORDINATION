#!/usr/bin/env python3
"""Detect a red/black cuboid top face and estimate depth with PnP.

This script is perception-only. It does not command the robot.

The detector is intentionally simple and inspectable:
1. Build a foreground mask from dark/red pixels that differ from the white table.
2. Pick a compact contour with a high red-pixel fraction.
3. In that ROI, fit a quadrilateral to the dark top-face contour.
4. Run cv2.solvePnP on the known top-face rectangle.

Point order in the output is:
    top_left, top_right, bottom_right, bottom_left
where "top" means image top, not robot/world up.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    points: np.ndarray
    target_bbox: tuple[int, int, int, int]
    score: float
    red_fraction: float
    dark_fraction: float
    mask_area: float


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    values = [int(float(part.strip())) for part in text.split(",") if part.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("ROI must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def clip_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] != 4:
        raise ValueError(f"expected 4 points, got {points.shape[0]}")
    y_sorted = points[np.argsort(points[:, 1])]
    top = y_sorted[:2][np.argsort(y_sorted[:2, 0])]
    bottom = y_sorted[2:][np.argsort(y_sorted[2:, 0])]
    return np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)


def suppress_shadow_bottom_right(points: np.ndarray) -> np.ndarray:
    points = order_quad(points)
    p0, p1, p2, p3 = points
    predicted_p2 = p1 + p3 - p0
    top_len = float(np.linalg.norm(p1 - p0))
    if top_len < 1.0:
        return points

    error = float(np.linalg.norm(p2 - predicted_p2))
    right_pull = float(p2[0] - predicted_p2[0])
    # The dark top mask can absorb a right-side shadow/side panel. In that
    # failure mode points 0, 1 and 3 stay on the top face while point 2 is
    # pulled outward. Use the three stable corners to regularize it.
    if error > max(8.0, 0.18 * top_len) and right_pull > max(6.0, 0.08 * top_len):
        points[2] = predicted_p2
    return points


def contour_to_quad(contour: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    quad: np.ndarray | None = None
    for eps in (0.015, 0.02, 0.025, 0.035, 0.045, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True).reshape(-1, 2)
        if approx.shape[0] == 4:
            quad = approx.astype(np.float64)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)
    return order_quad(quad)


def quad_is_usable(points: np.ndarray, bbox_width: int, bbox_height: int) -> bool:
    points = order_quad(points)
    top_width = float(np.linalg.norm(points[1] - points[0]))
    bottom_width = float(np.linalg.norm(points[2] - points[3]))
    left_height = float(np.linalg.norm(points[3] - points[0]))
    right_height = float(np.linalg.norm(points[2] - points[1]))
    avg_width = (top_width + bottom_width) / 2.0
    avg_height = (left_height + right_height) / 2.0
    side_ratio = max(left_height, right_height) / max(min(left_height, right_height), 1.0)
    if avg_width < max(12.0, bbox_width * 0.35):
        return False
    if avg_height < max(8.0, bbox_height * 0.25):
        return False
    if avg_width > bbox_width * 1.35 or avg_height > bbox_height * 1.15:
        return False
    if side_ratio > 1.75:
        return False
    return True


def fit_contrast_quad(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> np.ndarray | None:
    """Fit the dark object's high-contrast outer edge as a quadrilateral."""
    height, width = image.shape[:2]
    x, y, w, h = bbox
    x1, y1, x2, y2 = clip_roi((x - margin, y - margin, x + w + margin, y + h + margin), width, height)
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    bx, by = x - x1, y - y1
    sx1, sy1, sx2, sy2 = clip_roi(
        (bx - 6, by - 3, bx + w + 6, by + h + 6),
        crop.shape[1],
        crop.shape[0],
    )
    search = np.zeros(gray.shape, dtype=np.uint8)
    search[sy1:sy2, sx1:sx2] = 255

    local_values = blur[search > 0]
    if local_values.size == 0:
        return None
    median = float(np.median(local_values))
    lower = int(max(20, 0.45 * median))
    upper = int(min(180, 1.25 * median))
    edges = cv2.Canny(blur, lower, upper)
    edges = cv2.bitwise_and(edges, search)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > max(40.0, w * h * 0.06)]
    if not contours:
        return None

    def score_contour(contour: np.ndarray) -> float:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        bbox_area = max(float(cw * ch), 1.0)
        area = float(cv2.contourArea(contour))
        target_cx = bx + w / 2.0
        target_cy = by + h / 2.0
        contour_cx = cx + cw / 2.0
        contour_cy = cy + ch / 2.0
        center_penalty = 0.15 * ((contour_cx - target_cx) ** 2 + (contour_cy - target_cy) ** 2)
        compactness = min(area / bbox_area, 1.0)
        return area * (0.4 + compactness) - center_penalty

    for contour in sorted(contours, key=score_contour, reverse=True):
        quad = contour_to_quad(contour)
        if not quad_is_usable(quad, w, h):
            continue
        quad[:, 0] += x1
        quad[:, 1] += y1
        return order_quad(quad)
    return None


def build_target_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    chroma = cv2.absdiff(a_chan, 128) + cv2.absdiff(b_chan, 128)

    dark = (hsv[:, :, 2] < 135) & (hsv[:, :, 1] > 20)
    red1 = (hsv[:, :, 0] < 20) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 215)
    red2 = (hsv[:, :, 0] > 160) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 215)
    color_diff = (chroma > 20) & (l_chan < 180)
    mask = (dark | red1 | red2 | color_diff).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return mask, hsv, lab


def contour_color_stats(hsv: np.ndarray, contour: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    local = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
    cv2.drawContours(local, [shifted], -1, 255, -1)
    hsv_roi = hsv[y : y + h, x : x + w]
    inside = local > 0
    if inside.sum() == 0:
        return 0.0, 0.0
    hue = hsv_roi[:, :, 0]
    sat = hsv_roi[:, :, 1]
    val = hsv_roi[:, :, 2]
    red = (((hue < 22) | (hue > 158)) & (sat > 35) & (val < 220) & inside).sum() / inside.sum()
    dark = ((val < 115) & (sat > 15) & inside).sum() / inside.sum()
    return float(red), float(dark)


def find_target_bbox(
    image: np.ndarray,
    search_roi: tuple[int, int, int, int] | None,
    min_red_fraction: float,
) -> tuple[tuple[int, int, int, int], float, float, float, float]:
    height, width = image.shape[:2]
    mask, hsv, _lab = build_target_mask(image)
    search = np.zeros_like(mask)
    if search_roi is None:
        # Ignore the far top and bottom edges. In the G1 head camera, the
        # target sits on the table in the middle/lower-middle of the frame,
        # while the bottom edge often contains robot/body artifacts.
        search[int(height * 0.30) : int(height * 0.82), :] = 255
    else:
        x1, y1, x2, y2 = clip_roi(search_roi, width, height)
        search[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, search)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, tuple[int, int, int, int], float, float, float] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 80:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 8 or w > width * 0.35 or h > height * 0.35:
            continue
        red_fraction, dark_fraction = contour_color_stats(hsv, contour, (x, y, w, h))
        if red_fraction < min_red_fraction:
            continue
        aspect = w / max(h, 1)
        aspect_score = 1.0 if 0.8 <= aspect <= 4.0 else 0.65
        compact_score = min(area / max(w * h, 1), 1.0)
        score = area * aspect_score * (0.4 + 4.0 * red_fraction + 1.2 * dark_fraction) * (0.6 + compact_score)
        if best is None or score > best[0]:
            best = (float(score), (x, y, w, h), red_fraction, dark_fraction, area)
    if best is None:
        raise RuntimeError("could not find a red/black cuboid candidate")
    score, bbox, red_fraction, dark_fraction, area = best
    return bbox, score, red_fraction, dark_fraction, area


def fit_top_quad(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, w, h = bbox

    contrast_quad = fit_contrast_quad(image, bbox, margin)
    if contrast_quad is not None:
        return contrast_quad

    x1, y1, x2, y2 = clip_roi((x - margin, y - margin, x + w + margin, y + h + margin), width, height)
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    sat = hsv[:, :, 1]

    bx, by = x - x1, y - y1
    if h > 1.35 * max(w, 1):
        # A tall bbox usually means the mask swallowed the box side face. The
        # pose target is only the top surface, so fit the upper connected
        # target component before falling back to the darker inner print.
        target_mask, _target_hsv, _target_lab = build_target_mask(crop)
        ux1, uy1, ux2, uy2 = clip_roi(
            (bx - 6, by - 3, bx + w + 6, by + int(h * 0.50)),
            crop.shape[1],
            crop.shape[0],
        )
        upper_search = np.zeros(target_mask.shape, dtype=np.uint8)
        upper_search[uy1:uy2, ux1:ux2] = 255
        upper_mask = cv2.bitwise_and(target_mask, upper_search)
        upper_mask = cv2.morphologyEx(upper_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        upper_mask = cv2.morphologyEx(upper_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > max(45.0, w * h * 0.12)]
        if contours:
            contour = max(contours, key=cv2.contourArea)
            quad = contour_to_quad(contour)
            if quad_is_usable(quad, w, h):
                quad[:, 0] += x1
                quad[:, 1] += y1
                return order_quad(quad)

    bx1, by1, bx2, by2 = clip_roi(
        (
            bx - max(2, int(w * 0.05)),
            by - max(2, int(h * 0.08)),
            bx + w + max(2, int(w * 0.05)),
            by + int(h * 0.78),
        ),
        crop.shape[1],
        crop.shape[0],
    )
    box_value = value[max(0, by) : min(crop.shape[0], by + h), max(0, bx) : min(crop.shape[1], bx + w)]
    if box_value.size == 0:
        raise RuntimeError("target bbox is outside image crop")
    dark_threshold = int(np.clip(np.percentile(box_value, 42) + 12, 70, 145))
    box_sat = sat[max(0, by) : min(crop.shape[0], by + h), max(0, bx) : min(crop.shape[1], bx + w)]
    search = np.zeros(value.shape, dtype=np.uint8)
    search[by1:by2, bx1:bx2] = 255

    # First try a strict top-face mask. Table shadows are dark, but they are
    # usually much less saturated than the printed dark top surface. Keeping
    # only the darker, saturated connected component prevents left-camera
    # shadows from widening the fitted quadrilateral.
    strict_value_threshold = int(
        min(dark_threshold, np.clip(np.percentile(box_value, 30) + 20, 85, 120))
    )
    strict_sat_threshold = int(np.clip(np.percentile(box_sat, 20) - 4, 18, 24))
    strict_surface = ((value <= strict_value_threshold) & (sat >= strict_sat_threshold)).astype(np.uint8) * 255
    strict_mask = cv2.bitwise_and(strict_surface, search)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(strict_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > max(35.0, w * h * 0.08)]
    if contours:
        def strict_score(contour: np.ndarray) -> float:
            area = float(cv2.contourArea(contour))
            _cx, cy, _cw, ch = cv2.boundingRect(contour)
            center_y = cy + ch / 2.0
            lower_penalty = max(0.0, center_y - (by + h * 0.62))
            return area - 8.0 * lower_penalty * lower_penalty

        contour = max(contours, key=strict_score)
        quad = contour_to_quad(contour)
        if quad_is_usable(quad, w, h):
            quad[:, 0] += x1
            quad[:, 1] += y1
            return suppress_shadow_bottom_right(quad)

    # The actual top face is the low-value dark surface. The brighter printed
    # side panels are intentionally excluded, even if they are colorful.
    dark_surface = ((value <= dark_threshold) & ((sat > 8) | (value < 95))).astype(np.uint8) * 255
    top_mask = cv2.bitwise_and(dark_surface, search)
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    top_mask = cv2.dilate(top_mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 35]
    if not contours:
        raise RuntimeError("target found, but top-face mask has no usable contour")

    # Prefer upper dark components and fit a hull around their union. Text and
    # glare can split the dark top into islands, so using only one contour is
    # less stable.
    def contour_score(contour: np.ndarray) -> float:
        area = float(cv2.contourArea(contour))
        _cx, cy, _cw, _ch = cv2.boundingRect(contour)
        cy = cy + _ch / 2
        return area - 1.5 * max(0.0, cy - by2) ** 2

    contours = sorted(contours, key=contour_score, reverse=True)
    selected: list[np.ndarray] = []
    selected_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if selected and area < max(25.0, selected_area * 0.08):
            continue
        selected.append(contour)
        selected_area += area
        if len(selected) >= 4:
            break
    contour = cv2.convexHull(np.vstack(selected))
    quad = contour_to_quad(contour)
    quad[:, 0] += x1
    quad[:, 1] += y1
    return suppress_shadow_bottom_right(quad)


def detect_top_quad(
    image: np.ndarray,
    search_roi: tuple[int, int, int, int] | None = None,
    min_red_fraction: float = 0.12,
    margin: int = 20,
) -> Detection:
    bbox, score, red_fraction, dark_fraction, area = find_target_bbox(image, search_roi, min_red_fraction)
    points = fit_top_quad(image, bbox, margin)
    return Detection(points, bbox, score, red_fraction, dark_fraction, area)


def object_points(width_m: float, height_m: float) -> np.ndarray:
    return np.asarray(
        [
            [-width_m / 2.0, -height_m / 2.0, 0.0],
            [width_m / 2.0, -height_m / 2.0, 0.0],
            [width_m / 2.0, height_m / 2.0, 0.0],
            [-width_m / 2.0, height_m / 2.0, 0.0],
        ],
        dtype=np.float64,
    )


def solve_depth(
    points: np.ndarray,
    width_m: float,
    height_m: float,
    focal_px: float,
    cx: float,
    cy: float,
    fy_px: float | None = None,
    dist_coeffs: Any | None = None,
) -> dict[str, Any]:
    obj = object_points(width_m, height_m)
    fx_px = float(focal_px)
    fy_px = float(fy_px if fy_px is not None else focal_px)
    k = np.asarray([[fx_px, 0.0, cx], [0.0, fy_px, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if dist_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    ok, rvec, tvec = cv2.solvePnP(obj, points.astype(np.float64), k, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    reproj = float(np.linalg.norm(projected.reshape(-1, 2) - points, axis=1).mean())
    corner_xyz = (rotation @ obj.T + tvec).T
    return {
        "center_xyz_m": tvec.reshape(3).tolist(),
        "center_depth_m": float(tvec.reshape(3)[2]),
        "corner_depth_range_m": [float(corner_xyz[:, 2].min()), float(corner_xyz[:, 2].max())],
        "rotation_matrix": rotation.tolist(),
        "corner_xyz_m": corner_xyz.tolist(),
        "object_points_m": obj.tolist(),
        "mean_reprojection_px": reproj,
    }


def triangulate_stereo_quad(
    left_points: np.ndarray,
    right_points: np.ndarray,
    k_left: np.ndarray,
    dist_left: Any,
    k_right: np.ndarray,
    dist_right: Any,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    """Triangulate matched left/right quad corners into the LEFT camera frame.

    left_points/right_points are 4x2 pixel coordinates in the *same* corner
    order. rotation (3x3) and translation (3,) are the stereo extrinsics from
    cv2.stereoCalibrate, i.e. X_right = rotation @ X_left + translation, with
    translation in millimetres. Returned 3D points are in the left camera
    OpenCV frame (+X right, +Y down, +Z forward), in millimetres.
    """
    left = np.asarray(left_points, dtype=np.float64).reshape(-1, 1, 2)
    right = np.asarray(right_points, dtype=np.float64).reshape(-1, 1, 2)
    k_left = np.asarray(k_left, dtype=np.float64).reshape(3, 3)
    k_right = np.asarray(k_right, dtype=np.float64).reshape(3, 3)
    dist_l = np.asarray(dist_left, dtype=np.float64).reshape(-1, 1) if dist_left is not None else None
    dist_r = np.asarray(dist_right, dtype=np.float64).reshape(-1, 1) if dist_right is not None else None
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)

    # Undistort to normalized image coordinates so triangulation works in the
    # metric left-camera frame with P_left = [I|0], P_right = [R|T].
    norm_left = cv2.undistortPoints(left, k_left, dist_l).reshape(-1, 2)
    norm_right = cv2.undistortPoints(right, k_right, dist_r).reshape(-1, 2)
    p_left = np.hstack([np.eye(3), np.zeros((3, 1))])
    p_right = np.hstack([rotation, translation])
    homog = cv2.triangulatePoints(p_left, p_right, norm_left.T, norm_right.T)
    corners = (homog[:3] / homog[3]).T  # 4x3, left camera frame, mm

    # Reproject to both views to obtain a stereo consistency (reprojection) error.
    rvec_left = np.zeros((3, 1), dtype=np.float64)
    tvec_left = np.zeros((3, 1), dtype=np.float64)
    proj_left, _ = cv2.projectPoints(corners, rvec_left, tvec_left, k_left, dist_l)
    rvec_right, _ = cv2.Rodrigues(rotation)
    proj_right, _ = cv2.projectPoints(corners, rvec_right, translation, k_right, dist_r)
    err_left = np.linalg.norm(proj_left.reshape(-1, 2) - left.reshape(-1, 2), axis=1)
    err_right = np.linalg.norm(proj_right.reshape(-1, 2) - right.reshape(-1, 2), axis=1)
    reproj_px = float(np.concatenate([err_left, err_right]).mean())

    center = corners.mean(axis=0)

    # Fit the top-face plane normal via SVD of the centered corners; orient it
    # back toward the camera (negative depth direction).
    centered = corners - center
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2]
    if float(np.dot(normal, center)) > 0.0:
        normal = -normal
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # Measured top-face edge lengths (corner order TL,TR,BR,BL).
    edge_tl_tr = float(np.linalg.norm(corners[1] - corners[0]))
    edge_tr_br = float(np.linalg.norm(corners[2] - corners[1]))
    edge_br_bl = float(np.linalg.norm(corners[3] - corners[2]))
    edge_bl_tl = float(np.linalg.norm(corners[0] - corners[3]))
    side_a = 0.5 * (edge_tl_tr + edge_br_bl)
    side_b = 0.5 * (edge_tr_br + edge_bl_tl)
    long_mm = max(side_a, side_b)
    short_mm = min(side_a, side_b)

    return {
        "corner_xyz_mm": [[round(float(v), 1) for v in row] for row in corners],
        "center_xyz_mm": [round(float(v), 1) for v in center],
        "center_depth_mm": round(float(center[2]), 1),
        "top_plane_normal_xyz": [round(float(v), 6) for v in normal],
        "measured_top_size_mm": [round(long_mm, 1), round(short_mm, 1)],
        "corner_depth_range_mm": [round(float(corners[:, 2].min()), 1), round(float(corners[:, 2].max()), 1)],
        "stereo_reprojection_error_px": round(reproj_px, 3),
        "baseline_mm": round(float(np.linalg.norm(translation)), 2),
    }


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _ransac_plane(
    points: np.ndarray,
    thresh_mm: float = 2.0,
    iters: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Robustly fit a plane to a 3D point cloud.

    Returns (unit_normal, point_on_plane, inlier_mask, inlier_rms_mm) or None.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(int(iters)):
        idx = rng.choice(n, 3, replace=False)
        trio = points[idx]
        normal = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = float(np.dot(normal, trio[0]))
        dist = np.abs(points @ normal - offset)
        mask = dist < thresh_mm
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < 3:
        return None
    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid)
    normal = vh[2] / (np.linalg.norm(vh[2]) + 1e-12)
    offset = float(np.dot(normal, centroid))
    rms = float(np.sqrt(np.mean((inliers @ normal - offset) ** 2)))
    return normal, centroid, best_mask, rms


def stereo_feature_plane(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_quad: np.ndarray,
    k_left: np.ndarray,
    dist_left: Any,
    k_right: np.ndarray,
    dist_right: Any,
    rotation: np.ndarray,
    translation: np.ndarray,
    max_features: int = 300,
    mask_erode_px: int = 4,
    epipolar_max_px: float = 1.5,
    fb_max_px: float = 1.0,
    ransac_thresh_mm: float = 2.0,
    min_inliers: int = 8,
) -> dict[str, Any] | None:
    """Estimate the top-face plane pose from sparse stereo feature matches.

    Independent of the corner-based triangulation: detects texture features
    inside the left top-face polygon, matches them to the right image with LK
    optical flow, rejects outliers via forward-backward and epipolar checks,
    triangulates the survivors, and RANSAC-fits a plane. The center is the
    intersection of the quad-centroid viewing ray with that plane, so depth no
    longer depends on individual corner localization. All 3D output is in the
    left camera optical frame (mm). Returns None when there is too little
    texture or too few consistent matches (caller then keeps other methods).
    """
    if left_image is None or right_image is None or left_quad is None:
        return None
    k_left = np.asarray(k_left, dtype=np.float64).reshape(3, 3)
    k_right = np.asarray(k_right, dtype=np.float64).reshape(3, 3)
    dist_l = np.asarray(dist_left, dtype=np.float64).reshape(-1, 1) if dist_left is not None else None
    dist_r = np.asarray(dist_right, dtype=np.float64).reshape(-1, 1) if dist_right is not None else None
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    quad = np.asarray(left_quad, dtype=np.float64).reshape(-1, 2)

    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if left_image.ndim == 3 else left_image
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if right_image.ndim == 3 else right_image

    # Gate features to the detected top face only (mask, not bbox), eroded to
    # keep clear of the jittery boundary.
    mask = np.zeros(gray_left.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    if mask_erode_px > 0:
        mask = cv2.erode(mask, np.ones((mask_erode_px, mask_erode_px), np.uint8), iterations=1)
    if int(mask.sum()) <= 0:
        return None

    feats = cv2.goodFeaturesToTrack(
        gray_left, maxCorners=int(max_features), qualityLevel=0.01, minDistance=4, mask=mask
    )
    if feats is None or len(feats) < min_inliers:
        return None
    feats = feats.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    cv2.cornerSubPix(gray_left, feats, (5, 5), (-1, -1), criteria)

    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=criteria)
    fwd, st_fwd, _ = cv2.calcOpticalFlowPyrLK(gray_left, gray_right, feats, None, **lk_params)
    if fwd is None:
        return None
    back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray_right, gray_left, fwd, None, **lk_params)
    if back is None:
        return None

    pts_left = feats.reshape(-1, 2)
    pts_right = fwd.reshape(-1, 2)
    fb_err = np.linalg.norm(pts_left - back.reshape(-1, 2), axis=1)
    status = (st_fwd.reshape(-1) == 1) & (st_back.reshape(-1) == 1) & (fb_err < fb_max_px)

    # Epipolar consistency via the fundamental matrix derived from calibration.
    essential = _skew(translation) @ rotation
    fmat = np.linalg.inv(k_right).T @ essential @ np.linalg.inv(k_left)
    ones = np.ones((len(pts_left), 1))
    hl = np.hstack([pts_left, ones])
    hr = np.hstack([pts_right, ones])
    lines_r = (fmat @ hl.T).T  # epiline in right image for each left point
    denom_r = np.sqrt(lines_r[:, 0] ** 2 + lines_r[:, 1] ** 2) + 1e-12
    epi_r = np.abs(np.sum(lines_r * hr, axis=1)) / denom_r
    lines_l = (fmat.T @ hr.T).T
    denom_l = np.sqrt(lines_l[:, 0] ** 2 + lines_l[:, 1] ** 2) + 1e-12
    epi_l = np.abs(np.sum(lines_l * hl, axis=1)) / denom_l
    epi_err = 0.5 * (epi_r + epi_l)
    status = status & (epi_err < epipolar_max_px)

    if int(status.sum()) < min_inliers:
        return None
    match_left = pts_left[status]
    match_right = pts_right[status]

    norm_left = cv2.undistortPoints(match_left.reshape(-1, 1, 2), k_left, dist_l).reshape(-1, 2)
    norm_right = cv2.undistortPoints(match_right.reshape(-1, 1, 2), k_right, dist_r).reshape(-1, 2)
    p_left = np.hstack([np.eye(3), np.zeros((3, 1))])
    p_right = np.hstack([rotation, translation.reshape(3, 1)])
    homog = cv2.triangulatePoints(p_left, p_right, norm_left.T, norm_right.T)
    cloud = (homog[:3] / homog[3]).T  # Nx3 in left frame, mm

    # Keep points in front of the camera and within a sane depth window.
    valid = cloud[:, 2] > 1.0
    cloud = cloud[valid]
    if len(cloud) < min_inliers:
        return None

    fit = _ransac_plane(cloud, thresh_mm=ransac_thresh_mm, iters=200)
    if fit is None:
        return None
    normal, plane_pt, inlier_mask, rms = fit
    if int(inlier_mask.sum()) < min_inliers:
        return None
    inliers = cloud[inlier_mask]

    # Orient normal toward the camera (top face seen from the front).
    if float(np.dot(normal, plane_pt)) > 0.0:
        normal = -normal

    # Center = intersection of the quad-centroid viewing ray with the plane.
    centroid_px = quad.mean(axis=0).reshape(1, 1, 2)
    cen_norm = cv2.undistortPoints(centroid_px.astype(np.float64), k_left, dist_l).reshape(2)
    ray = np.asarray([cen_norm[0], cen_norm[1], 1.0], dtype=np.float64)
    plane_d = float(np.dot(normal, plane_pt))
    denom = float(np.dot(normal, ray))
    if abs(denom) < 1e-9:
        center = inliers.mean(axis=0)
    else:
        center = (plane_d / denom) * ray

    # In-plane long axis via PCA of the inlier cloud.
    centered = inliers - inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    long_axis = vh[0] / (np.linalg.norm(vh[0]) + 1e-12)

    return {
        "center_xyz_mm": [round(float(v), 1) for v in center],
        "center_depth_mm": round(float(center[2]), 1),
        "top_plane_normal_xyz": [round(float(v), 6) for v in normal],
        "long_axis_unit_xyz": [round(float(v), 6) for v in long_axis],
        "num_features": int(len(pts_left)),
        "num_matches": int(status.sum()),
        "num_inliers": int(inlier_mask.sum()),
        "inlier_ratio": round(float(inlier_mask.sum()) / float(len(pts_left)), 3),
        "plane_rms_mm": round(float(rms), 3),
        "epipolar_rms_px": round(float(np.sqrt(np.mean(epi_err[status] ** 2))), 3),
        "baseline_mm": round(float(np.linalg.norm(translation)), 2),
        "inlier_depth_range_mm": [round(float(inliers[:, 2].min()), 1), round(float(inliers[:, 2].max()), 1)],
    }


def draw_debug(image: np.ndarray, detection: Detection, output_path: Path) -> None:
    output = image.copy()
    x, y, w, h = detection.target_bbox
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
    pts = detection.points.astype(np.int32)
    cv2.polylines(output, [pts], True, (0, 255, 0), 2)
    for idx, point in enumerate(pts):
        cv2.circle(output, tuple(point), 5, (0, 255, 255), -1)
        cv2.putText(
            output,
            str(idx),
            tuple(point + np.array([6, -6])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)


def capture_head_images(host: str, wait_sec: float) -> tuple[np.ndarray, np.ndarray]:
    from teleimager.image_client import ImageClient

    client = ImageClient(host=host)
    deadline = time.monotonic() + wait_sec
    try:
        while time.monotonic() < deadline:
            frame, _fps = client.get_head_frame()
            if frame is not None:
                height, width = frame.shape[:2]
                if width % 2 != 0:
                    raise RuntimeError(f"expected side-by-side binocular image with even width, got {frame.shape}")
                return frame[:, : width // 2].copy(), frame[:, width // 2 :].copy()
            time.sleep(0.05)
    finally:
        client.close()
    raise TimeoutError(f"no head camera frame after {wait_sec:.1f}s")


def load_images(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.capture:
        return capture_head_images(args.host, args.wait_sec)
    if args.left_image is None or args.right_image is None:
        raise SystemExit("provide --left-image/--right-image or use --capture")
    left = cv2.imread(str(args.left_image), cv2.IMREAD_COLOR)
    right = cv2.imread(str(args.right_image), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(args.left_image)
    if right is None:
        raise FileNotFoundError(args.right_image)
    return left, right


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-image", type=Path)
    parser.add_argument("--right-image", type=Path)
    parser.add_argument("--capture", action="store_true", help="capture from teleimager head camera")
    parser.add_argument("--host", default="127.0.0.1", help="teleimager host for --capture")
    parser.add_argument("--wait-sec", type=float, default=5.0)
    parser.add_argument("--object-width-m", type=float, default=0.149)
    parser.add_argument("--object-height-m", type=float, default=0.093)
    parser.add_argument("--focal-px", type=float, default=260.0, help="fallback head camera focal length in pixels")
    parser.add_argument("--fx", type=float, help="head camera fx in pixels; defaults to --focal-px")
    parser.add_argument("--fy", type=float, help="head camera fy in pixels; defaults to --focal-px")
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--left-roi", type=parse_roi, default=None, help="optional x1,y1,x2,y2")
    parser.add_argument("--right-roi", type=parse_roi, default=None, help="optional x1,y1,x2,y2")
    parser.add_argument("--min-red-fraction", type=float, default=0.12)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/auto_pnp_cuboid"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    left, right = load_images(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_dir / "left_input.jpg"), left)
    cv2.imwrite(str(args.out_dir / "right_input.jpg"), right)

    detections = {
        "left": detect_top_quad(left, args.left_roi, args.min_red_fraction),
        "right": detect_top_quad(right, args.right_roi, args.min_red_fraction),
    }
    depths = {
        side: solve_depth(
            det.points,
            args.object_width_m,
            args.object_height_m,
            args.fx if args.fx is not None else args.focal_px,
            args.cx,
            args.cy,
            fy_px=args.fy if args.fy is not None else args.focal_px,
        )
        for side, det in detections.items()
    }
    draw_debug(left, detections["left"], args.out_dir / "left_auto_quad.jpg")
    draw_debug(right, detections["right"], args.out_dir / "right_auto_quad.jpg")

    result = {
        "object_top_size_m": [args.object_width_m, args.object_height_m],
        "intrinsics_assumption": {
            "focal_px": args.focal_px,
            "fx": args.fx if args.fx is not None else args.focal_px,
            "fy": args.fy if args.fy is not None else args.focal_px,
            "cx": args.cx,
            "cy": args.cy,
        },
        "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "views": {},
        "estimated_camera_mid_top_center_depth_m": float(
            (depths["left"]["center_depth_m"] + depths["right"]["center_depth_m"]) / 2.0
        ),
        "debug_images": {
            "left": str(args.out_dir / "left_auto_quad.jpg"),
            "right": str(args.out_dir / "right_auto_quad.jpg"),
        },
    }
    for side, det in detections.items():
        result["views"][side] = {
            "points_px": det.points.tolist(),
            "target_bbox_xywh": list(det.target_bbox),
            "mask_score": det.score,
            "red_fraction": det.red_fraction,
            "dark_fraction": det.dark_fraction,
            "mask_area_px": det.mask_area,
            **depths[side],
        }

    result_path = args.out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
