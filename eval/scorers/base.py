"""Join GT samples with predictions into metric-ready records."""

from typing import Any

from eval.metrics.registry import compute_metrics

CLS_METRICS = ["acc", "bacc", "f1"]


def join_records(gt: list[dict], pred: list[dict]) -> list[dict]:
    pred_by_id = {str(p["id"]): p for p in pred if "id" in p}
    out = []
    for s in gt:
        sid = str(s["id"])
        if "ground_truth" not in s:
            raise KeyError(f"Sample {sid} missing 'ground_truth'")
        p = pred_by_id.get(sid)
        out.append(
            {
                "id": sid,
                "status": "missing_prediction" if p is None else p.get("status", "unknown"),
                "ground_truth": str(s["ground_truth"]),
                "prediction": "" if p is None else p.get("prediction", ""),
                "raw_prediction": "" if p is None else p.get("raw_prediction", ""),
            }
        )
    return out


def score_overall(gt: list[dict], pred: list[dict]) -> dict[str, Any]:
    raw = compute_metrics(join_records(gt, pred), CLS_METRICS, "mcq")
    return {"scores": raw["metrics"], "counts": raw["counts"]}


def format_cls_scores(scores: dict[str, float]) -> str:
    return " ".join(f"{k}={scores[k]:.4f}" for k in CLS_METRICS if k in scores)
