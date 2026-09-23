#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a Qwen3.5 base checkpoint.

The merged output is a normal Hugging Face model directory that can be served
directly by vLLM or loaded with Transformers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a Qwen3.5 checkpoint.")
    parser.add_argument("--base-model", required=True, help="Base model path or Hugging Face model id.")
    parser.add_argument("--lora-path", required=True, help="LoRA adapter checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for the merged model.")
    parser.add_argument(
        "--model-kind",
        choices=("auto", "vision", "text"),
        default="auto",
        help="Use 'vision' for Qwen VL/image-text models, 'text' for language-only models.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="bf16",
        help="Torch dtype used while loading the base model.",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument(
        "--safe-merge",
        action="store_true",
        help="Check adapter weights for NaNs before merging. Slower, but useful for debugging.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable custom model code.")
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_base_model(args: argparse.Namespace):
    common_kwargs = {
        "dtype": resolve_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.model_kind == "vision":
        return AutoModelForImageTextToText.from_pretrained(args.base_model, **common_kwargs)
    if args.model_kind == "text":
        return AutoModelForCausalLM.from_pretrained(args.base_model, **common_kwargs)

    try:
        return AutoModelForImageTextToText.from_pretrained(args.base_model, **common_kwargs)
    except Exception as vision_error:
        print(f"[merge_lora] ImageTextToText load failed, falling back to CausalLM: {vision_error}")
        return AutoModelForCausalLM.from_pretrained(args.base_model, **common_kwargs)


def save_processor_or_tokenizer(base_model: str, output_dir: Path, trust_remote_code: bool) -> None:
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        processor.save_pretrained(output_dir)
        print("[merge_lora] Saved processor files.")
        return
    except Exception as processor_error:
        print(f"[merge_lora] Processor save failed, falling back to tokenizer: {processor_error}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)
    print("[merge_lora] Saved tokenizer files.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge_lora] Loading base model: {args.base_model}")
    base_model = load_base_model(args)

    print(f"[merge_lora] Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("[merge_lora] Merging adapter into base weights.")
    model = model.merge_and_unload(safe_merge=args.safe_merge)

    print(f"[merge_lora] Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    save_processor_or_tokenizer(args.base_model, output_dir, args.trust_remote_code)

    print("[merge_lora] Done.")


if __name__ == "__main__":
    main()
