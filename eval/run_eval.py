#!/usr/bin/env python
"""Run inference + scoring across checkpoints and datasets."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from eval.aggregate import aggregate
from eval.manifest import EVAL_DIR, load_manifest, load_run_config, sample_by_dataset
from eval.score import score_dataset


def _load_module(script: Path):
    name = f"_inf_{script.stem}_{abs(hash(script)) & 0xFFFF:x}"
    spec = importlib.util.spec_from_file_location(name, script)
    if not spec or not spec.loader:
        raise ImportError(script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _inference_argv(run_cfg: dict, checkpoint: str, output_json: str, enable_thinking: bool) -> list[str]:
    argv = ["--model_id", str(run_cfg["model_id"]), run_cfg["checkpoint_arg"], checkpoint, "--output_json", output_json]
    if enable_thinking:
        argv.append("--enable_thinking")
    for key, val in (run_cfg.get("extra_args") or {}).items():
        flag = key if str(key).startswith("--") else f"--{key}"
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        else:
            argv.extend([flag, str(val)])
    return argv


def run_inference(run_cfg: dict, checkpoint: str, output_json: str, samples: list[dict], enable_thinking: bool) -> None:
    mod = _load_module(Path(run_cfg["inference_script"]))
    args = mod.parse_args(_inference_argv(run_cfg, checkpoint, output_json, enable_thinking))
    mod.run_inference(args, samples=samples)


def _print_score_block(result: dict[str, Any], indent: str = "  ") -> None:
    c, s = result["counts"], result["scores"]
    parts = [f"{k}={s[k]:.4f}" for k in ("acc", "bacc", "f1") if k in s]
    line = indent + " ".join(parts) if parts else f"{indent}scores={s}"
    if "n_correct" in c:
        line += f" ({c['n_correct']}/{c['n_scored']} correct)"
    print(line)
    if c.get("per_class"):
        print(f"{indent}  per_class: {c['per_class']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run_config", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--skip_inference", action="store_true")
    p.add_argument("--skip_scoring", action="store_true")
    p.add_argument("--enable_thinking", action="store_true")
    args = p.parse_args()

    m = load_manifest(args.manifest)
    run_cfg = load_run_config(args.run_config)
    thinking = bool(
        run_cfg.get("enable_thinking")
        or (run_cfg.get("extra_args") or {}).get("enable_thinking")
        or args.enable_thinking
    )
    samples_by_ds = sample_by_dataset(m)

    print(f"seed={m.get('seed', 42)} enable_thinking={thinking}")
    for e in m["datasets"]:
        total = len(json.loads(Path(e["path"]).read_text(encoding="utf-8")))
        print(f"  {e['name']}: {len(samples_by_ds[e['name']])}/{total}")

    run_dir = Path(args.output_dir or m.get("output_dir") or EVAL_DIR / "results") / run_cfg["run_name"]

    for checkpoint in run_cfg["checkpoints"]:
        ckpt = Path(checkpoint).name
        ckpt_dir = run_dir / ckpt
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for e in m["datasets"]:
            name = e["name"]
            samples = samples_by_ds[name]
            pred = ckpt_dir / f"{name}_pred.json"
            score_path = ckpt_dir / f"{name}_score.json"

            if not args.skip_inference:
                print(f"\n>>> {name} @ {ckpt} ({len(samples)} samples)")
                run_inference(run_cfg, checkpoint, str(pred), samples, thinking)

            if not args.skip_scoring:
                result = score_dataset(
                    args.manifest,
                    name,
                    str(pred),
                    gt_samples=samples,
                    output_json=str(score_path),
                )
                print("  overall:", end="")
                _print_score_block(result)
                for key in ("by_task", "by_subset", "by_broad_category", "by_project"):
                    if key not in result:
                        continue
                    group = key.removeprefix("by_")
                    for label, block in result[key].items():
                        print(f"    {group}={label}:", end="")
                        _print_score_block(block, indent=" ")

    if not args.skip_scoring:
        print(f"\nWrote {aggregate(run_dir)}")
        print(f"Plot curves: python -m eval.plot_curves --run_dir {run_dir}")


if __name__ == "__main__":
    main()
