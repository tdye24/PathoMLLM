"""Bounding-box evaluation for detection/box-supervised segmentation datasets."""

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


def score(gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, Any]:
    """Score boxes; for multiple boxes, greedily match pairs by descending IoU."""
    pred_by_id = {str(item["id"]): item for item in pred if "id" in item}
    metric_names = ("iou", "dice", "precision", "recall")
    totals = dict.fromkeys(metric_names, 0.0)
    details = []
    parsed_predictions = 0
    empty_ground_truth = 0

    for sample in gt:
        sample_id = str(sample["id"])
        ground_truth = sample.get("ground_truth", sample.get("boxes", sample.get("bbox")))
        references = parse_boxes(ground_truth)
        if not references and not _is_explicit_empty(ground_truth):
            raise ValueError(f"Sample {sample_id} has no valid ground-truth box")
        empty_ground_truth += not references
        prediction_record = pred_by_id.get(sample_id)
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

        pairs = sorted(
            (
                (box_metrics(ref, candidate)["iou"], ref_idx, pred_idx)
                for ref_idx, ref in enumerate(references)
                for pred_idx, candidate in enumerate(predictions)
            ),
            reverse=True,
        )
        used_refs: set[int] = set()
        used_preds: set[int] = set()
        sums = dict.fromkeys(metric_names, 0.0)
        for _, ref_idx, pred_idx in pairs:
            if ref_idx in used_refs or pred_idx in used_preds:
                continue
            used_refs.add(ref_idx)
            used_preds.add(pred_idx)
            metrics = box_metrics(references[ref_idx], predictions[pred_idx])
            for name in metric_names:
                sums[name] += metrics[name]

        # Empty GT + empty prediction is a correct empty-mask prediction. Any unmatched
        # GT or prediction is a zero-overlap object (macro object average).
        if not references and not predictions and prediction_is_explicit_empty:
            sample_scores = dict.fromkeys(metric_names, 1.0)
        else:
            denominator = max(len(references), len(predictions), 1)
            sample_scores = {name: sums[name] / denominator for name in metric_names}
        for name in metric_names:
            totals[name] += sample_scores[name]
        details.append(
            {
                "id": sample_id,
                "ground_truth_boxes": references,
                "prediction_boxes_norm1000": norm_predictions,
                "prediction_boxes": predictions,
                "scores": sample_scores,
            }
        )

    n_samples = len(gt)
    return {
        "scores": {name: totals[name] / n_samples if n_samples else 0.0 for name in metric_names},
        "counts": {
            "n_samples": n_samples,
            "n_parsed_predictions": parsed_predictions,
            "n_missing_or_invalid": n_samples - parsed_predictions,
            "n_empty_ground_truth": empty_ground_truth,
        },
        "details": details,
    }
