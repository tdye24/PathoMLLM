#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PathoMLLM batch inference — Qwen3.5 native vision + LoRA adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train"
PLUGIN = TRAIN_DIR / "pathomllm_plugin.py"


def _bootstrap_plugin() -> None:
    """Load pathomllm_plugin (FSDP shim + s3:// image patch) before swift imports."""
    if not PLUGIN.is_file():
        raise FileNotFoundError(f"pathomllm_plugin not found: {PLUGIN}")
    if str(TRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(TRAIN_DIR))
    spec = importlib.util.spec_from_file_location("pathomllm_plugin", PLUGIN)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin: {PLUGIN}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pathomllm_plugin"] = mod
    spec.loader.exec_module(mod)


_bootstrap_plugin()

from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGE_MARKER = "<image>"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def extract_answer(text: str) -> str:
    stripped = text.strip()
    if "</think>" in stripped:
        return stripped.split("</think>", 1)[-1].strip()
    return stripped


def count_image_markers(messages: List[Dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            count += content.count(IMAGE_MARKER)
    return count


def filter_inference_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    filtered = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg.get("role") in {"user", "system"}
    ]
    if not filtered:
        raise ValueError("No user/system messages left after removing assistant turns.")
    return filtered


def resolve_images(sample: Dict[str, Any]) -> List[str]:
    images = sample.get("images")
    if images is None:
        raise ValueError("Sample missing 'images' field.")
    if isinstance(images, str):
        images = [images]
    return [str(path) for path in images]


def resolve_chat_template_kwargs(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if sample.get("chat_template_kwargs"):
        return dict(sample["chat_template_kwargs"])
    max_tokens = sample.get("max_tokens")
    if max_tokens is None:
        return None
    if isinstance(max_tokens, list):
        max_tokens = min(int(v) for v in max_tokens)
    else:
        max_tokens = int(max_tokens)
    return {"max_pixels": max_tokens * 32 * 32}


def validate_image_tags(messages: List[Dict[str, Any]], images: List[str], sample_id: str) -> None:
    tag_count = count_image_markers(messages)
    if tag_count != len(images):
        raise ValueError(
            f"Sample {sample_id}: image placeholder count ({tag_count}) != len(images) ({len(images)})."
        )


def build_engine(model_id: str, adapter_dir: str, attn_impl: str):
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
    os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "128")
    os.environ.setdefault("FPS_MAX_FRAMES", "12")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    from swift.infer_engine import RequestConfig, TransformersEngine

    engine = TransformersEngine(
        model_id,
        adapters=[adapter_dir],
        attn_impl=attn_impl,
        torch_dtype="bfloat16",
    )
    request_config = RequestConfig(max_tokens=2048, temperature=0.0)
    return engine, request_config


def run_inference(args: argparse.Namespace, samples=None) -> None:
    setup_logging()

    from swift.infer_engine import InferRequest

    engine, request_config = build_engine(args.model_id, args.adapter_dir, args.attn_implementation)

    if samples is None:
        with open(args.input_json, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if getattr(args, "limit_samples", None) is not None:
            samples = samples[: args.limit_samples]

    results: List[Dict[str, Any]] = []
    printed_traceback = False

    with tqdm(total=len(samples), desc="Inference Progress", unit="sample") as pbar:
        for idx, sample in enumerate(samples):
            sample_id = sample.get("id", f"sample_{idx}")
            try:
                if "messages" not in sample:
                    raise ValueError("Sample missing 'messages'.")

                image_paths = resolve_images(sample)
                validate_image_tags(sample["messages"], image_paths, str(sample_id))
                infer_messages = filter_inference_messages(sample["messages"])
                chat_template_kwargs = resolve_chat_template_kwargs(sample)

                infer_request = InferRequest(
                    messages=infer_messages,
                    images=image_paths,
                    chat_template_kwargs=chat_template_kwargs,
                )
                resp_list = engine.infer([infer_request], request_config=request_config)
                raw_prediction = resp_list[0].choices[0].message.content.strip()
                prediction = extract_answer(raw_prediction)

                print(f"\n{'=' * 60}")
                print(f"[{idx + 1}/{len(samples)}] ID: {sample_id}")
                print(f"Prediction: {prediction}")
                print(f"{'=' * 60}\n")

                results.append(
                    {
                        "id": sample_id,
                        "status": "success",
                        "prediction": prediction,
                        "raw_prediction": raw_prediction,
                    }
                )
            except Exception as exc:
                tb = traceback.format_exc()
                results.append(
                    {
                        "id": sample_id,
                        "status": "error",
                        "prediction": "",
                        "error_msg": str(exc),
                        "traceback": tb,
                    }
                )
                print(f"\n{'=' * 60}")
                print(f"[{idx + 1}/{len(samples)}] ID: {sample_id} - ERROR")
                print(f"Error: {exc}")
                if not printed_traceback:
                    print(tb)
                    printed_traceback = True
                print(f"{'=' * 60}\n")

            pbar.update(1)
            success_count = sum(1 for item in results if item.get("status") == "success")
            pbar.set_postfix(success=success_count, error=len(results) - success_count)

            with open(args.output_json, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)

    success_count = sum(1 for item in results if item.get("status") == "success")
    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} samples to {args.output_json}")
    print(f"Success: {success_count} | Error: {len(results) - success_count}")
    print(f"{'=' * 60}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PathoMLLM batch inference (Qwen3.5 native vision + LoRA)."
    )
    parser.add_argument("--model_id", type=str, required=True, help="Base Qwen3.5 model path")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="ms-swift LoRA checkpoint dir (checkpoint-* or final adapter dir)",
    )
    parser.add_argument("--input_json", type=str, default="eval/data/bcnb.json")
    parser.add_argument("--output_json", type=str, default="eval/results/pred.json")
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_inference(parse_args())
