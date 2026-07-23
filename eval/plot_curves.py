#!/usr/bin/env python
"""Plot eval metric curves vs checkpoint step (single run or compare runs)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_DATASET_METRIC: dict[str, str] = {
    "bcnb": "acc",
    "roi_cls": "acc",
    "roi_cls_vqa": "acc",
    "pathmmu": "acc",
    "mcq": "acc",
    "tiny_mcq": "acc",
}

DEFAULT_METRICS = ("acc", "bacc", "f1")
_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class Point:
    step: int
    label: str
    values: dict[str, float]


@dataclass(frozen=True)
class Series:
    run_label: str
    points: tuple[Point, ...]


def parse_run_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Empty run label in {spec!r}")
    return label, Path(path)


def checkpoint_info(name: str) -> tuple[int, str]:
    if name == "final_model":
        return 10**9, "final"
    if m := _CHECKPOINT.match(name):
        step = int(m.group(1))
        return step, str(step)
    return 0, name


def _scores_from_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data.get("scores") or {}
    return {m: float(scores[m]) for m in DEFAULT_METRICS if m in scores}


def load_series_from_run_dir(run_label: str, run_dir: Path, dataset: str) -> Series:
    pattern = f"{dataset}_score.json"
    rows: list[tuple[int, str, dict[str, float]]] = []

    for score_path in sorted(run_dir.rglob(pattern)):
        ckpt = next(
            (p for p in reversed(score_path.parts) if p.startswith("checkpoint-") or p == "final_model"),
            "",
        )
        if not ckpt:
            continue
        data = json.loads(score_path.read_text(encoding="utf-8"))
        if data.get("dataset") and data["dataset"] != dataset:
            continue
        sort_key, x_label = checkpoint_info(ckpt)
        rows.append((sort_key, x_label, _scores_from_json(score_path)))

    if not rows:
        raise FileNotFoundError(f"No {pattern} under {run_dir}")

    rows.sort(key=lambda r: r[0])
    points = tuple(Point(step=k, label=lab, values=vals) for k, lab, vals in rows)
    return Series(run_label=run_label, points=points)


def load_series_from_summary(summary_csv: Path, run_label: str, dataset: str) -> Series:
    rows: list[tuple[int, str, dict[str, float]]] = []
    with summary_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != dataset:
                continue
            if run_label and row.get("run_name") and row["run_name"] != run_label:
                continue
            ckpt = row.get("checkpoint") or ""
            sort_key, x_label = checkpoint_info(ckpt)
            if row.get("step"):
                try:
                    sort_key = int(row["step"])
                    x_label = str(sort_key)
                except ValueError:
                    pass
            values = {m: float(row[m]) for m in DEFAULT_METRICS if m in row and row[m] != ""}
            if not values:
                continue
            rows.append((sort_key, x_label, values))

    if not rows:
        raise ValueError(f"No rows for dataset={dataset!r} in {summary_csv}")

    rows.sort(key=lambda r: r[0])
    label = run_label or (summary_csv.parent.name if summary_csv.parent.name else "run")
    points = tuple(Point(step=k, label=lab, values=vals) for k, lab, vals in rows)
    return Series(run_label=label, points=points)


def load_all_series(
    runs: Iterable[tuple[str, Path]],
    datasets: Iterable[str],
) -> dict[str, list[Series]]:
    out: dict[str, list[Series]] = {ds: [] for ds in datasets}
    for label, run_dir in runs:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        for ds in datasets:
            out[ds].append(load_series_from_run_dir(label, run_dir, ds))
    return out


def _x_positions(all_series: list[Series]) -> tuple[list[int], list[str]]:
    steps: dict[int, str] = {}
    for series in all_series:
        for pt in series.points:
            steps[pt.step] = pt.label
    ordered = sorted(steps)
    return ordered, [steps[s] for s in ordered]


def plot_dataset(
    dataset: str,
    series_list: list[Series],
    output_path: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    x_steps, x_labels = _x_positions(series_list)
    x_index = {s: i for i, s in enumerate(x_steps)}

    fig, ax = plt.subplots(figsize=(9, 5))
    run_markers = ("o", "s", "^", "D", "v", "P", "X")
    metric_styles = {
        "acc": ("-", "C0"),
        "bacc": ("--", "C1"),
        "f1": ("-.", "C2"),
    }

    for run_i, series in enumerate(series_list):
        xs = [x_index[pt.step] for pt in series.points]
        for metric in metrics:
            if not all(metric in pt.values for pt in series.points):
                continue
            ys = [pt.values[metric] for pt in series.points]
            linestyle, color = metric_styles.get(metric, ("-", None))
            label = f"{series.run_label} {metric}" if len(series_list) > 1 else metric
            ax.plot(
                xs,
                ys,
                linestyle=linestyle,
                color=color,
                marker=run_markers[run_i % len(run_markers)],
                linewidth=2,
                markersize=6,
                label=label,
            )

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title or dataset)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(
    series_by_dataset: dict[str, list[Series]],
    output_dir: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    fmt: str = "png",
) -> list[Path]:
    written: list[Path] = []
    for dataset, series_list in series_by_dataset.items():
        if not series_list:
            continue
        out = output_dir / f"{dataset}_curves.{fmt}"
        plot_dataset(dataset, series_list, out, metrics=metrics)
        written.append(out)
    return written


def plot_combined(
    series_by_dataset: dict[str, list[Series]],
    output_path: Path,
    *,
    metric: str = "acc",
) -> Path:
    """One figure with subplots per dataset (single run recommended)."""
    import matplotlib.pyplot as plt

    datasets = [ds for ds, sl in series_by_dataset.items() if sl]
    if not datasets:
        raise ValueError("no datasets to plot")

    n = len(datasets)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        series_list = series_by_dataset[dataset]
        x_steps, x_labels = _x_positions(series_list)
        x_index = {s: i for i, s in enumerate(x_steps)}
        markers = ("o", "s", "^", "D")

        for run_i, series in enumerate(series_list):
            xs = [x_index[pt.step] for pt in series.points]
            ys = [pt.values.get(metric, float("nan")) for pt in series.points]
            ax.plot(
                xs,
                ys,
                marker=markers[run_i % len(markers)],
                linewidth=2,
                label=series.run_label,
            )

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(dataset)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        if run_i > 0 or len(series_list) > 1:
            ax.legend(fontsize=8)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"{metric} vs checkpoint", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def discover_datasets(run_dir: Path) -> list[str]:
    names: set[str] = set()
    for path in run_dir.rglob("*_score.json"):
        name = path.name.removesuffix("_score.json")
        if name:
            names.add(name)
    return sorted(names)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot eval metrics vs checkpoint step.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--run_dir",
        type=Path,
        help="Single eval run dir (contains checkpoint-*/<dataset>_score.json)",
    )
    src.add_argument(
        "--summary_csv",
        type=Path,
        help="summary.csv from eval.aggregate (alternative to --run_dir)",
    )
    p.add_argument(
        "--runs",
        nargs="*",
        type=parse_run_arg,
        metavar="LABEL=DIR",
        help="Compare multiple runs, e.g. exp1=eval/results/run_a exp2=eval/results/run_b",
    )
    p.add_argument("--output_dir", type=Path, default=Path("eval/plots"))
    p.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to plot (default: auto-discover from score files)",
    )
    p.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        choices=DEFAULT_METRICS,
        help="Metrics to draw on each dataset chart (default: acc bacc f1)",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Also write one combined figure (subplots per dataset, acc only)",
    )
    p.add_argument(
        "--combined_metric",
        default="acc",
        choices=DEFAULT_METRICS,
        help="Metric for --combined figure",
    )
    args = p.parse_args()

    metrics = tuple(args.metrics)

    if args.runs:
        runs = list(args.runs)
        datasets = args.datasets or sorted(
            {ds for _, rd in runs for ds in discover_datasets(rd)}
        )
        series_by_dataset = load_all_series(runs, datasets)
    elif args.run_dir:
        run_dir = args.run_dir
        label = run_dir.name
        datasets = args.datasets or discover_datasets(run_dir)
        if not datasets:
            raise SystemExit(f"No *_score.json under {run_dir}")
        series_by_dataset = {ds: [load_series_from_run_dir(label, run_dir, ds)] for ds in datasets}
    else:
        summary = args.summary_csv
        if not summary.is_file():
            raise SystemExit(f"summary not found: {summary}")
        datasets = args.datasets
        if not datasets:
            with summary.open(encoding="utf-8") as f:
                datasets = sorted({row["dataset"] for row in csv.DictReader(f) if row.get("dataset")})
        series_by_dataset = {
            ds: [load_series_from_summary(summary, "", ds)] for ds in datasets
        }

    written = plot_all(series_by_dataset, args.output_dir, metrics=metrics, fmt=args.format)
    for path in written:
        print(f"Wrote {path}")

    if args.combined:
        combined = args.output_dir / f"all_datasets_{args.combined_metric}.{args.format}"
        plot_combined(series_by_dataset, combined, metric=args.combined_metric)
        print(f"Wrote {combined}")


if __name__ == "__main__":
    main()
