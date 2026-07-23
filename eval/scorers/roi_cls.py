"""ROI classification VQA: overall + per-task (same layout as bcnb)."""

from typing import Any

from . import bcnb


def score(gt: list[dict], pred: list[dict]) -> dict[str, Any]:
    return bcnb.score(gt, pred)
