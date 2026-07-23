"""PathMMU-style: overall + per-subset cls metrics (acc, bacc, f1)."""

import argparse
import json
from pathlib import Path
from typing import Any

from eval.metrics.registry import compute_metrics

from .base import CLS_METRICS, format_cls_scores, join_records

METRICS = CLS_METRICS


def score(gt: list[dict], pred: list[dict]) -> dict[str, Any]:
    if not all("subset" in s for s in gt):
        raise ValueError("PathMMU GT JSON must have 'subset' on every sample")

    records = join_records(gt, pred)
    raw = compute_metrics(records, METRICS, "mcq")
    subset_by_id = {str(s["id"]): s["subset"] for s in gt}

    by_subset = {}
    for subset in sorted(set(subset_by_id.values())):
        sub = compute_metrics([r for r in records if subset_by_id[r["id"]] == subset], METRICS, "mcq")
        by_subset[subset] = {"scores": sub["metrics"], "counts": sub["counts"]}

    return {"scores": raw["metrics"], "counts": raw["counts"], "by_subset": by_subset}


def main() -> None:
    p = argparse.ArgumentParser(description="PathMMU overall + per-subset cls metrics")
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    result = score(gt, pred)

    c, s = result["counts"], result["scores"]
    print(f"overall: {format_cls_scores(s)} ({c['n_correct']}/{c['n_scored']})")
    for subset, block in result["by_subset"].items():
        c, s = block["counts"], block["scores"]
        print(f"  {subset}: {format_cls_scores(s)} ({c['n_correct']}/{c['n_scored']})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
