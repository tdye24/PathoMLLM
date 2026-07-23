#!/usr/bin/env python
"""Collect *_score.json under a run dir into summary.csv."""

import argparse
import csv
import json
import re
from pathlib import Path


def aggregate(results_dir: Path, output: Path | None = None) -> Path:
    rows = []
    metrics: set[str] = set()
    for path in sorted(results_dir.rglob("*_score.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scores = data.get("scores", {})
        metrics.update(scores)
        ckpt = next((p for p in reversed(path.parts) if p.startswith("checkpoint-") or p == "final_model"), "")
        step = int(m.group(1)) if (m := re.match(r"checkpoint-(\d+)$", ckpt)) else ""
        rows.append(
            {
                "run_name": path.parent.parent.name if ckpt else "",
                "checkpoint": ckpt,
                "step": step,
                "dataset": data.get("dataset", ""),
                **scores,
                "score_file": str(path),
            }
        )

    if not rows:
        raise SystemExit(f"No *_score.json under {results_dir}")

    cols = ["run_name", "checkpoint", "step", "dataset"] + sorted(metrics) + ["score_file"]
    out = output or results_dir / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    out = aggregate(Path(args.results_dir), Path(args.output) if args.output else None)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
