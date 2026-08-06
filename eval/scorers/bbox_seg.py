"""Bounding-box evaluation for detection/box-supervised segmentation datasets."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_BOX = re.compile(
    rf"[\[(]\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*[\])]"
)


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


def _area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


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

    for sample in gt:
        sample_id = str(sample["id"])
        references = parse_boxes(sample.get("ground_truth", sample.get("boxes", sample.get("bbox"))))
        if not references:
            raise ValueError(f"Sample {sample_id} has no valid ground-truth box")
        prediction_record = pred_by_id.get(sample_id)
        predictions = parse_boxes(prediction_record.get("prediction", "")) if prediction_record else []
        parsed_predictions += bool(predictions)

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

        # Unmatched GT and predictions are zero-overlap objects (macro object average).
        denominator = max(len(references), len(predictions), 1)
        sample_scores = {name: sums[name] / denominator for name in metric_names}
        for name in metric_names:
            totals[name] += sample_scores[name]
        details.append(
            {
                "id": sample_id,
                "ground_truth_boxes": references,
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
        },
        "details": details,
    }
