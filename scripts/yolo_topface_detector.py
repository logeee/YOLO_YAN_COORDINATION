#!/usr/bin/env python3
"""YOLO segmentation detector for cigarette-box top-face candidates.

This module is intentionally independent from PnP/coordinate conversion. It
turns one image into one or more top-face candidates. The pose layer decides
which candidate is used for XYZ.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from auto_pnp_cuboid_depth import Detection, contour_to_quad, order_quad


SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_SELECT_METHODS = (
    "confidence",
    "score",
    "largest",
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "center",
    "index",
)
YOLO_MODEL_CACHE: dict[tuple[str, str], Any] = {}
TORCHVISION_NMS_PATCHED = False


def _points_list(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points.tolist()]


def resolve_model_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path
    repo_relative = SCRIPT_DIR.parent / path
    if repo_relative.exists():
        return repo_relative
    script_relative = SCRIPT_DIR / path
    if script_relative.exists():
        return script_relative
    return path


def resolve_yolo_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_yolo_model(resolved_model: Path, task: str = "segment") -> Any:
    cache_key = (str(resolved_model.resolve()), task)
    if cache_key not in YOLO_MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "YOLO mode requires ultralytics. Install it in the robot Python env first."
            ) from exc
        YOLO_MODEL_CACHE[cache_key] = YOLO(str(resolved_model), task=task)
    return YOLO_MODEL_CACHE[cache_key]


def _torch_nms_fallback(boxes: Any, scores: Any, iou_threshold: float) -> Any:
    import torch

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep: list[Any] = []

    while order.numel() > 0:
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[index], x1[rest])
        yy1 = torch.maximum(y1[index], y1[rest])
        xx2 = torch.minimum(x2[index], x2[rest])
        yy2 = torch.minimum(y2[index], y2[rest])
        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        union = areas[index] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-12)
        order = rest[iou <= float(iou_threshold)]

    return torch.stack(keep).to(dtype=torch.long)


def _ensure_torchvision_nms() -> None:
    global TORCHVISION_NMS_PATCHED
    if TORCHVISION_NMS_PATCHED:
        return
    try:
        import torch
        import torchvision

        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cpu")
        scores = torch.tensor([1.0], device="cpu")
        torchvision.ops.nms(boxes, scores, 0.5)
    except Exception:
        import torchvision

        torchvision.ops.nms = _torch_nms_fallback
    TORCHVISION_NMS_PATCHED = True


def _bbox_xyxy_to_xywh(xyxy: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    x1_i = max(0, min(width - 1, int(math.floor(x1))))
    y1_i = max(0, min(height - 1, int(math.floor(y1))))
    x2_i = max(x1_i + 1, min(width, int(math.ceil(x2))))
    y2_i = max(y1_i + 1, min(height, int(math.ceil(y2))))
    return x1_i, y1_i, x2_i - x1_i, y2_i - y1_i


def _quad_from_yolo_mask(mask: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[np.ndarray, float]:
    if mask.shape[:2] != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    clean = (mask > 0).astype(np.uint8) * 255
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 20.0]
    if not contours:
        raise RuntimeError("YOLO mask has no usable contour")
    contour = max(contours, key=cv2.contourArea)
    return order_quad(contour_to_quad(contour)), float(cv2.contourArea(contour))


def _sort_candidate_records(
    records: list[dict[str, Any]],
    select: str,
    image_shape: tuple[int, int, int],
) -> list[dict[str, Any]]:
    height, width = image_shape[:2]
    center_x = float(width) / 2.0
    center_y = float(height) / 2.0

    def center_distance_sq(record: dict[str, Any]) -> float:
        dx = float(record["center_x"]) - center_x
        dy = float(record["center_y"]) - center_y
        return dx * dx + dy * dy

    if select == "confidence":
        return sorted(records, key=lambda item: (-float(item["confidence"]), int(item["raw_yolo_index"])))
    if select == "score":
        return sorted(records, key=lambda item: (-float(item["score"]), int(item["raw_yolo_index"])))
    if select == "largest":
        return sorted(records, key=lambda item: (-float(item["mask_area_px"]), int(item["raw_yolo_index"])))
    if select == "leftmost":
        return sorted(records, key=lambda item: (float(item["center_x"]), int(item["raw_yolo_index"])))
    if select == "rightmost":
        return sorted(records, key=lambda item: (-float(item["center_x"]), int(item["raw_yolo_index"])))
    if select == "topmost":
        return sorted(records, key=lambda item: (float(item["center_y"]), int(item["raw_yolo_index"])))
    if select == "bottommost":
        return sorted(records, key=lambda item: (-float(item["center_y"]), int(item["raw_yolo_index"])))
    if select == "center":
        return sorted(records, key=lambda item: (center_distance_sq(item), int(item["raw_yolo_index"])))
    if select == "index":
        return sorted(records, key=lambda item: int(item["raw_yolo_index"]))
    raise ValueError(f"unsupported YOLO select method: {select}")


def _normalize_label(text: Any) -> str:
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum())


def _parse_label_filter(label_filter: str | None) -> set[str]:
    if label_filter is None:
        return set()
    labels = {_normalize_label(item) for item in str(label_filter).split(",") if _normalize_label(item)}
    return labels


def _class_matches_filter(class_id: int, class_name: str, labels: set[str]) -> bool:
    if not labels:
        return True
    class_id_text = _normalize_label(class_id)
    class_name_text = _normalize_label(class_name)
    for label in labels:
        if label == class_id_text or label == class_name_text:
            return True
        if label and label in class_name_text:
            return True
    return False


def detect_yolo_points_from_image(
    image: np.ndarray,
    model_path: str | Path,
    conf: float = 0.15,
    imgsz: int = 640,
    device: str = "auto",
    mask_threshold: float = 0.5,
    select: str = "score",
    select_index: int = 0,
    label_filter: str | None = None,
) -> tuple[Detection, dict[str, Any]]:
    """Return selected top-face detection plus metadata for all candidates.

    The returned Detection is the single candidate selected for downstream PnP.
    The metadata dict contains `candidates`, an ordered list of all YOLO masks
    that passed the threshold.
    """
    resolved_model = resolve_model_path(model_path)
    if not resolved_model.exists():
        raise RuntimeError(f"YOLO model not found: {resolved_model}")

    model = get_yolo_model(resolved_model, task="segment")
    _ensure_torchvision_nms()
    resolved_device = resolve_yolo_device(device)
    predict_kwargs: dict[str, Any] = {
        "conf": float(conf),
        "imgsz": int(imgsz),
        "retina_masks": True,
        "verbose": False,
        "device": resolved_device,
    }
    result = model.predict(image, **predict_kwargs)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError(f"YOLO found no detections above conf={conf}")

    names = result.names or getattr(model, "names", {}) or {}
    requested_labels = _parse_label_filter(label_filter)
    candidate_records: list[dict[str, Any]] = []
    filtered_out_records: list[dict[str, Any]] = []
    for idx, box in enumerate(result.boxes):
        confidence = float(box.conf.item())
        if confidence < float(conf):
            continue
        cls_id = int(box.cls.item())
        class_name = str(names.get(cls_id, cls_id))
        xyxy = box.xyxy.reshape(-1).detach().cpu().numpy().astype(float)
        bbox = _bbox_xyxy_to_xywh(xyxy, image.shape)
        x1, y1, x2, y2 = [float(value) for value in xyxy.tolist()]
        if result.masks is None:
            raise RuntimeError("YOLO result has no segmentation mask")
        mask = result.masks.data[idx].detach().cpu().numpy()
        mask = (mask >= float(mask_threshold)).astype(np.uint8) * 255
        points, mask_area = _quad_from_yolo_mask(mask, image.shape)

        detection = Detection(points, bbox, confidence, 0.0, 0.0, mask_area)
        score = confidence + min(float(mask_area), 20000.0) * 1e-7
        info = {
            "model": str(resolved_model),
            "task": "segment",
            "source": "yolo_segmentation_mask",
            "device": resolved_device,
            "raw_yolo_index": int(idx),
            "confidence": round(confidence, 4),
            "score": round(float(score), 6),
            "class_id": cls_id,
            "class_name": class_name,
            "box_xyxy": [round(float(value), 1) for value in xyxy.tolist()],
            "box_center_px": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
            "mask_area_px": round(float(mask_area), 1),
            "points_px": _points_list(points),
        }
        record = {
            "raw_yolo_index": int(idx),
            "confidence": float(confidence),
            "score": float(score),
            "mask_area_px": float(mask_area),
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
            "detection": detection,
            "info": info,
        }
        if _class_matches_filter(cls_id, class_name, requested_labels):
            candidate_records.append(record)
        else:
            filtered_out_records.append(record)

    if not candidate_records:
        available = sorted(
            {
                str(record["info"].get("class_name"))
                for record in filtered_out_records
                if isinstance(record.get("info"), dict)
            }
        )
        label_note = f" matching label_filter={label_filter!r}" if requested_labels else ""
        available_note = f"; available labels above conf: {available}" if available else ""
        raise RuntimeError(f"YOLO found no detections above conf={conf}{label_note}{available_note}")
    ordered_records = _sort_candidate_records(candidate_records, select, image.shape)
    for candidate_index, record in enumerate(ordered_records):
        record["info"]["candidate_index"] = int(candidate_index)
        record["info"]["selection_method"] = select
    if select_index < 0 or select_index >= len(ordered_records):
        raise RuntimeError(
            f"YOLO candidate index {select_index} out of range for {len(ordered_records)} candidates "
            f"after --yolo-select {select}"
        )

    selected_record = ordered_records[int(select_index)]
    selected_info = dict(selected_record["info"])
    selected_info.update(
        {
            "selected": True,
            "selected_candidate_index": int(select_index),
            "selection_method": select,
            "candidate_count": len(ordered_records),
            "label_filter": str(label_filter) if label_filter else None,
            "filtered_out_candidate_count": len(filtered_out_records),
            "candidates": [dict(record["info"]) for record in ordered_records],
            "all_candidate_labels": sorted(
                {
                    str(record["info"].get("class_name"))
                    for record in [*candidate_records, *filtered_out_records]
                    if isinstance(record.get("info"), dict)
                }
            ),
        }
    )
    return selected_record["detection"], selected_info


# Backward-compatible aliases for older service code/imports.
_YOLO_MODEL_CACHE = YOLO_MODEL_CACHE
_get_yolo_model = get_yolo_model
_resolve_model_path = resolve_model_path
_resolve_yolo_device = resolve_yolo_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only YOLO top-face detection on one image.")
    parser.add_argument("--image", type=Path, required=True, help="input image path")
    parser.add_argument("--model", default="models/Liqun_Xiongmao.pt", help="Ultralytics segmentation model")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--select", choices=YOLO_SELECT_METHODS, default="confidence")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--label", "--class-name", dest="label_filter", help="only select this YOLO class name/id")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    detection, info = detect_yolo_points_from_image(
        image,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        mask_threshold=args.mask_threshold,
        select=args.select,
        select_index=args.index,
        label_filter=args.label_filter,
    )
    candidates = info.pop("candidates", [])
    result = {
        "ok": True,
        "detector": "yolo_topface_detector",
        "selected": info,
        "candidates": candidates,
        "selected_points_px": _points_list(detection.points),
        "selected_bbox_xywh": list(detection.target_bbox),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
