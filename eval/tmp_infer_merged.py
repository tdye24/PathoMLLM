#!/usr/bin/env python
"""Single-file inference for a merged Qwen3.5 vision model."""

import argparse
import json
import os
from pathlib import Path

MODEL_ID = ""
INPUT_JSON = ""
OUTPUT_JSON = ""


def prepare_swift():
    """Apply compatibility settings needed by ms-swift."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
    try:
        from torch.distributed.fsdp import FSDPModule  # noqa: F401
    except ImportError:
        from torch.distributed._composable.fsdp import FSDPModule
        import torch.distributed.fsdp as fsdp

        fsdp.FSDPModule = FSDPModule

    from PIL import Image, PngImagePlugin

    Image.MAX_IMAGE_PIXELS = None
    PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024**2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID or None, required=not MODEL_ID)
    parser.add_argument("--input_json", default=INPUT_JSON or None, required=not INPUT_JSON)
    parser.add_argument("--output_json", default=OUTPUT_JSON or None, required=not OUTPUT_JSON)
    parser.add_argument("--limit_samples", type=int)
    parser.add_argument("--attn_implementation", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"))
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_swift()

    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from tqdm import tqdm

    engine = TransformersEngine(args.model_id, attn_impl=args.attn_implementation, torch_dtype="bfloat16")
    config = RequestConfig(max_tokens=2048, temperature=0.0)
    samples = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    samples = samples[: args.limit_samples] if args.limit_samples is not None else samples
    output, results = Path(args.output_json), []
    output.parent.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(tqdm(samples, desc="Inference")):
        sample_id = sample.get("id", f"sample_{index}")
        try:
            images = sample["images"]
            images = [images] if isinstance(images, str) else [str(x) for x in images]
            messages = [
                {"role": x["role"], "content": x["content"]}
                for x in sample["messages"]
                if x.get("role") in {"system", "user"}
            ]
            if sum(str(x["content"]).count("<image>") for x in messages) != len(images):
                raise ValueError("the number of <image> tags does not match images")
            kwargs = sample.get("chat_template_kwargs")
            if not kwargs and sample.get("max_tokens") is not None:
                tokens = sample["max_tokens"]
                tokens = min(map(int, tokens)) if isinstance(tokens, list) else int(tokens)
                kwargs = {"max_pixels": tokens * 32**2}

            request = InferRequest(messages=messages, images=images, chat_template_kwargs=kwargs)
            raw = engine.infer([request], request_config=config)[0].choices[0].message.content.strip()
            prediction = raw.split("</think>", 1)[-1].strip()
            row = {"id": sample_id, "status": "success", "prediction": prediction, "raw_response": raw}
        except Exception as error:  # Keep processing and save failures in the output file.
            row = {"id": sample_id, "status": "error", "prediction": "", "error_msg": str(error)}
        if "ground_truth" in sample:
            row["ground_truth"] = sample["ground_truth"]
        results.append(row)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(results)} results to {output}")


if __name__ == "__main__":
    main()
