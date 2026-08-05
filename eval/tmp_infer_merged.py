#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Temporary one-shot inference for a fully merged Qwen3.5 model (no LoRA).

Fill the three paths below, or pass --model_id / --input_json / --output_json.

Input JSON: a list of samples, each with at least:
  {
    "id": "sample_001",
    "messages": [{"role": "user", "content": "<image>\\nYour question..."}],
    "images": ["s3://.../xxx.png"],
    "ground_truth": "optional"
  }

Usage:
  cd PathoMLLM
  python eval/tmp_infer_merged.py
  # or:
  python eval/tmp_infer_merged.py \\
    --model_id /path/to/Qwen3.5-9B-v1 \\
    --input_json /path/to/my_tmp.json \\
    --output_json /path/to/my_tmp_pred.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

# =============================================================================
# Fill these in (CLI flags override when provided)
# =============================================================================
MODEL_ID = ""  # e.g. /home/ma-user/work/yetiandi/PathoMLLM/model/Qwen3.5-9B-v1
INPUT_JSON = ""  # e.g. /path/to/my_tmp.json
OUTPUT_JSON = ""  # e.g. /path/to/my_tmp_pred.json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

# Import after path setup so pathomllm_plugin bootstraps via batch_inference.
from eval.batch_inference import (  # noqa: E402
    build_engine,
    extract_answer,
    filter_inference_messages,
    resolve_chat_template_kwargs,
    resolve_images,
    setup_logging,
    validate_image_tags,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temp merged-model inference (no LoRA).")
    p.add_argument("--model_id", type=str, default=MODEL_ID or None)
    p.add_argument("--input_json", type=str, default=INPUT_JSON or None)
    p.add_argument("--output_json", type=str, default=OUTPUT_JSON or None)
    p.add_argument("--limit_samples", type=int, default=None)
    p.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    args = p.parse_args(argv)
    missing = [k for k in ("model_id", "input_json", "output_json") if not getattr(args, k)]
    if missing:
        p.error(
            "Set paths at the top of this file or via CLI; missing: "
            + ", ".join(f"--{k}" for k in missing)
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()

    from swift.infer_engine import InferRequest

    engine, request_config = build_engine(args.model_id, adapter_dir=None, attn_impl=args.attn_implementation)

    with open(args.input_json, encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError(f"input_json must be a JSON list, got {type(samples).__name__}")
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    printed_traceback = False

    with tqdm(total=len(samples), desc="Inference", unit="sample") as pbar:
        for idx, sample in enumerate(samples):
            sample_id = sample.get("id", f"sample_{idx}")
            try:
                if "messages" not in sample:
                    raise ValueError("Sample missing 'messages'.")
                image_paths = resolve_images(sample)
                validate_image_tags(sample["messages"], image_paths, str(sample_id))
                infer_messages = filter_inference_messages(sample["messages"])
                chat_template_kwargs = resolve_chat_template_kwargs(sample)

                resp_list = engine.infer(
                    [
                        InferRequest(
                            messages=infer_messages,
                            images=image_paths,
                            chat_template_kwargs=chat_template_kwargs,
                        )
                    ],
                    request_config=request_config,
                )
                raw_response = resp_list[0].choices[0].message.content.strip()
                prediction = extract_answer(raw_response)

                row: dict[str, Any] = {
                    "id": sample_id,
                    "status": "success",
                    "prediction": prediction,
                    "raw_response": raw_response,
                }
                if "ground_truth" in sample:
                    row["ground_truth"] = sample["ground_truth"]
                results.append(row)

                print(f"\n[{idx + 1}/{len(samples)}] {sample_id}: {prediction}")
            except Exception as exc:
                tb = traceback.format_exc()
                row = {
                    "id": sample_id,
                    "status": "error",
                    "prediction": "",
                    "raw_response": "",
                    "error_msg": str(exc),
                    "traceback": tb,
                }
                if "ground_truth" in sample:
                    row["ground_truth"] = sample["ground_truth"]
                results.append(row)
                print(f"\n[{idx + 1}/{len(samples)}] {sample_id} ERROR: {exc}")
                if not printed_traceback:
                    print(tb)
                    printed_traceback = True

            pbar.update(1)
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("status") == "success")
    print(f"\nSaved {len(results)} -> {out_path} (success={n_ok}, error={len(results) - n_ok})")


if __name__ == "__main__":
    main()
