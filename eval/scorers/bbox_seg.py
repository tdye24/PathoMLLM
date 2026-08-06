"""Bounding-box detection evaluation with AP at IoU 0.5."""

from __future__ import annotations

import ast
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_BOX = re.compile(
    rf"[\[(]\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*[\])]"
)
_REMOTE_PREFIXES = ("s3://", "obs://")


def _valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return box


def _boxes_from_object(value: Any) -> list[list[float]]:
    """Extract boxes from JSON-like values without treating arbitrary numbers as boxes."""
    box = _valid_box(value)
    if box is not None:
        return [box]
    if isinstance(value, dict):
        for key in ("bbox", "box", "bboxes", "boxes", "ground_truth"):
            if key in value:
                return _boxes_from_object(value[key])
        return []
    if isinstance(value, (list, tuple)):
        boxes = []
        for item in value:
            boxes.extend(_boxes_from_object(item))
        return boxes
    return []


def parse_boxes(value: Any) -> list[list[float]]:
    """Parse SmartPath-style ``<bbox>[x1,y1,x2,y2]</bbox>`` and JSON boxes."""
    if not isinstance(value, str):
        return _boxes_from_object(value)

    text = value.strip()
    if not text:
        return []
    candidates = [text]
    candidates.extend(re.findall(r"<bbox(?:es)?>\s*(.*?)\s*</bbox(?:es)?>", text, re.I | re.S))
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                boxes = _boxes_from_object(loader(candidate))
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if boxes:
                return boxes
    return [[float(x) for x in match] for match in _BOX.findall(text) if _valid_box(match)]


def _is_explicit_empty(value: Any) -> bool:
    """Return whether an annotation explicitly represents an empty box collection."""
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if isinstance(value, dict):
        for key in ("bbox", "box", "bboxes", "boxes", "ground_truth"):
            if key in value:
                return _is_explicit_empty(value[key])
        return False
    if isinstance(value, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                return _is_explicit_empty(loader(value.strip()))
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
    return False


def _area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _image_size(sample: dict[str, Any]) -> tuple[int, int]:
    """Read the first image size, using moxing for ModelArts object storage."""
    images = sample.get("images")
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list) or not images:
        raise ValueError(f"Sample {sample.get('id')} needs an image to restore norm1000 boxes")

    path = str(images[0]).strip()
    if path.startswith(_REMOTE_PREFIXES):
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        try:
            import moxing as mox  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(f"moxing is required to read remote image {path}") from exc
        with mox.file.File(path, "rb") as stream:
            data = stream.read()
        if not data:
            raise OSError(f"empty read from remote image: {path}")
        with Image.open(io.BytesIO(data)) as image:
            return image.size

    with Image.open(Path(path)) as image:
        return image.size


def norm1000_to_pixels(box: list[float], width: int, height: int) -> list[float]:
    """Restore a Qwen3.5 norm1000 box to coordinates in the original image."""
    x1, y1, x2, y2 = box
    return [x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000]


def box_metrics(reference: list[float], prediction: list[float]) -> dict[str, float]:
    """Compute overlap metrics by treating each bounding box as a binary rectangle mask."""
    ix = max(0.0, min(reference[2], prediction[2]) - max(reference[0], prediction[0]))
    iy = max(0.0, min(reference[3], prediction[3]) - max(reference[1], prediction[1]))
    intersection = ix * iy
    ref_area, pred_area = _area(reference), _area(prediction)
    union = ref_area + pred_area - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "dice": 2 * intersection / (ref_area + pred_area) if ref_area + pred_area else 0.0,
        "precision": intersection / pred_area if pred_area else 0.0,
        "recall": intersection / ref_area if ref_area else 0.0,
    }


def calculate_ap50(gt_boxes: list[list[float]], pred_boxes: list[list[float]]) -> float:
    """Reproduce SmartPath-R1's per-image, generation-order AP50 calculation."""
    if not gt_boxes or not pred_boxes:
        return 0.0

    true_positives = []
    false_positives = []
    gt_matched = [False] * len(gt_boxes)
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = box_metrics(gt_box, pred_box)["iou"]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        is_true_positive = best_iou >= 0.5
        true_positives.append(int(is_true_positive))
        false_positives.append(int(not is_true_positive))
        if is_true_positive:
            gt_matched[best_gt_idx] = True

    cumulative_tp = 0
    cumulative_fp = 0
    recalls = [0.0]
    precisions = [0.0]
    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / len(gt_boxes))
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp + 1e-8))
    recalls.append(1.0)
    precisions.append(0.0)

    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    return sum(
        (recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]
        for idx in range(len(recalls) - 1)
    )


def score(gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a single-class detection dataset using AP50."""
    pred_by_id = {str(item["id"]): item for item in pred if "id" in item}
    details = []
    ap50_scores = []
    parsed_predictions = 0
    empty_ground_truth = 0
    missing_predictions = 0

    for sample in gt:
        sample_id = str(sample["id"])
        ground_truth = sample.get("ground_truth", sample.get("boxes", sample.get("bbox")))
        references = parse_boxes(ground_truth)
        if not references and not _is_explicit_empty(ground_truth):
            raise ValueError(f"Sample {sample_id} has no valid ground-truth box")
        empty_ground_truth += not references
        prediction_record = pred_by_id.get(sample_id)
        if prediction_record is None:
            missing_predictions += 1
            continue
        prediction_value = prediction_record.get("prediction", "") if prediction_record else ""
        norm_predictions = parse_boxes(prediction_value)
        prediction_is_explicit_empty = _is_explicit_empty(prediction_value)
        prediction_is_valid = bool(norm_predictions) or prediction_is_explicit_empty
        parsed_predictions += prediction_is_valid
        if norm_predictions:
            width, height = _image_size(sample)
            predictions = [norm1000_to_pixels(box, width, height) for box in norm_predictions]
        else:
            predictions = []
        sample_ap50 = calculate_ap50(references, predictions)
        ap50_scores.append(sample_ap50)
        details.append(
            {
                "id": sample_id,
                "ground_truth_boxes": references,
                "prediction_boxes_norm1000": norm_predictions,
                "prediction_boxes": predictions,
                "ap50": sample_ap50,
            }
        )

    n_samples = len(gt)
    n_scored = len(ap50_scores)
    return {
        "scores": {"ap50": sum(ap50_scores) / n_scored if n_scored else 0.0},
        "counts": {
            "n_samples": n_samples,
            "n_scored": n_scored,
            "n_missing_predictions": missing_predictions,
            "n_parsed_predictions": parsed_predictions,
            "n_missing_or_invalid": n_samples - missing_predictions - parsed_predictions,
            "n_empty_ground_truth": empty_ground_truth,
        },
        "details": details,
    }
