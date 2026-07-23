#!/usr/bin/env python
"""Score predictions — dispatches to dataset-specific scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.manifest import get_dataset, load_manifest, sample_entries
from eval.scorers import score as run_scorer

from eval.scorers.base import join_records  # noqa: F401


def score_dataset(
    manifest_path: str,
    dataset_name: str,
    pred_json: str,
    *,
    gt_samples: list[dict] | None = None,
    output_json: str | None = None,
) -> dict[str, Any]:
    m = load_manifest(manifest_path)
    entry = get_dataset(m, dataset_name)
    gt = gt_samples or sample_entries(entry, m)
    preds = json.loads(Path(pred_json).read_text(encoding="utf-8"))

    scored = run_scorer(entry["scorer"], gt, preds)
    result = {
        "dataset": dataset_name,
        "scorer": entry["scorer"],
        "pred_json": str(Path(pred_json).resolve()),
        "gt_json": entry["path"],
        **scored,
    }
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred_json", required=True)
    p.add_argument("--output_json", default=None)
    args = p.parse_args()
    result = score_dataset(args.manifest, args.dataset, args.pred_json, output_json=args.output_json)
    print(json.dumps({k: result["scores"][k] for k in ("acc", "bacc", "f1") if k in result["scores"]} | {"counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
