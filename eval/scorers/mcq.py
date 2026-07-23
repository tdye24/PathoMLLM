"""Generic MCQ scorer — overall acc / bacc / f1 only."""

import argparse
import json
from pathlib import Path
from typing import Any

from .base import format_cls_scores, score_overall


def score(gt: list[dict], pred: list[dict]) -> dict[str, Any]:
    return score_overall(gt, pred)


def main() -> None:
    p = argparse.ArgumentParser(description="MCQ cls metrics")
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    result = score(gt, pred)

    c, s = result["counts"], result["scores"]
    print(f"{format_cls_scores(s)} ({c['n_correct']}/{c['n_scored']})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
