"""One scorer module per benchmark."""

from typing import Any, Callable

from . import bcnb, mcq, pathmmu, roi_cls

ScorerFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]

SCORERS: dict[str, ScorerFn] = {
    "mcq": mcq.score,
    "bcnb": bcnb.score,
    "roi_cls": roi_cls.score,
    "pathmmu": pathmmu.score,
}


def score(scorer: str, gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, Any]:
    fn = SCORERS.get(scorer)
    if fn is None:
        raise KeyError(f"Unknown scorer {scorer!r}. Available: {sorted(SCORERS)}")
    return fn(gt, pred)
