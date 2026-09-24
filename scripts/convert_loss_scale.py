#!/usr/bin/env python3
"""Convert Swift JSONL to assistant-message loss_scale format.

This script is intended for datasets whose assistant content may contain:
  <think>...</think>
  <answer>...</answer>

ms-swift applies loss_scale at the assistant-message level, so a single
assistant message containing both tags is split into separate assistant
messages with different loss weights. Samples without a think block are kept
and only the answer block is weighted.

Example:
  python scripts/convert_loss_scale.py \
    --input data/msr_swift.jsonl \
    --output data/msr_swift_loss_scale.jsonl \
    --think_loss_scale 0.5 \
    --answer_loss_scale 1.0 \
    --overwrite

Input assistant message:
  {"role": "assistant",
   "content": "<think>reasoning</think>\n\n<answer>A. xxx</answer>"}

Output assistant messages:
  {"role": "assistant",
   "content": "<think>reasoning</think>\n\n",
   "loss_scale": 0.5}
  {"role": "assistant",
   "content": "<answer>A. xxx</answer>",
   "loss_scale": 1.0}

If a sample has no positive reasoning path and contains only an answer block,
the script keeps that sample and only applies answer_loss_scale:
  {"role": "assistant",
   "content": "<answer>B. xxx</answer>",
   "loss_scale": 1.0}

When training with non-binary loss weights such as 0.5 or 0.3, pass the
following arguments to ms-swift:
  --loss_scale default
  --is_binary_loss_scale false
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


TAG_RE = re.compile(r"(?is)<(think|answer)\b[^>]*>.*?</\1>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split assistant <think>/<answer> blocks and add ms-swift loss_scale values."
    )
    parser.add_argument("--input", required=True, help="Input Swift JSONL.")
    parser.add_argument("--output", required=True, help="Output Swift JSONL with loss_scale fields.")
    parser.add_argument("--think_loss_scale", type=float, default=0.5)
    parser.add_argument("--answer_loss_scale", type=float, default=1.0)
    parser.add_argument(
        "--untagged_loss_scale",
        type=float,
        default=1.0,
        help="Loss scale for assistant messages without <think> or <answer> tags.",
    )
    parser.add_argument(
        "--drop_empty_think",
        action="store_true",
        default=True,
        help="Drop empty <think></think> blocks.",
    )
    parser.add_argument(
        "--keep_empty_think",
        action="store_false",
        dest="drop_empty_think",
        help="Keep empty <think></think> blocks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print JSON with this indent. Default writes compact JSONL.",
    )
    return parser.parse_args()


def tag_body_is_empty(block: str) -> bool:
    body = re.sub(r"(?is)^<think\b[^>]*>|</think>$", "", block).strip()
    return not body


def scaled_message(
    message: dict[str, Any],
    content: str,
    loss_scale: float,
    strip_content: bool = True,
) -> dict[str, Any]:
    converted = dict(message)
    converted["content"] = content.strip() if strip_content else content
    converted["loss_scale"] = loss_scale
    return converted


def split_assistant_message(
    message: dict[str, Any],
    think_loss_scale: float,
    answer_loss_scale: float,
    untagged_loss_scale: float,
    drop_empty_think: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    content = str(message.get("content", ""))
    pieces: list[dict[str, Any]] = []
    stats = {
        "assistant_messages": 1,
        "think_blocks": 0,
        "answer_blocks": 0,
        "untagged_assistant_messages": 0,
        "dropped_empty_think": 0,
    }

    matches = list(TAG_RE.finditer(content))
    if not matches:
        stats["untagged_assistant_messages"] += 1
        return [scaled_message(message, content, untagged_loss_scale)], stats

    for index, match in enumerate(matches):
        tag = match.group(1).lower()
        block = match.group(0).strip()
        next_match = matches[index + 1] if index + 1 < len(matches) else None
        separator = ""
        if next_match is not None:
            between = content[match.end(): next_match.start()]
            if between and between.strip() == "":
                separator = between
        if tag == "think":
            if drop_empty_think and tag_body_is_empty(block):
                stats["dropped_empty_think"] += 1
                continue
            stats["think_blocks"] += 1
            pieces.append(scaled_message(message, block + separator, think_loss_scale, strip_content=False))
        elif tag == "answer":
            stats["answer_blocks"] += 1
            pieces.append(scaled_message(message, block + separator, answer_loss_scale, strip_content=False))

    if not pieces:
        stats["untagged_assistant_messages"] += 1
        return [scaled_message(message, content, untagged_loss_scale)], stats
    return pieces, stats


def convert_record(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, int]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record does not contain a messages list")

    converted_messages: list[dict[str, Any]] = []
    stats = {
        "records": 1,
        "assistant_messages": 0,
        "think_blocks": 0,
        "answer_blocks": 0,
        "untagged_assistant_messages": 0,
        "dropped_empty_think": 0,
        "answer_only_records": 0,
        "think_answer_records": 0,
    }
    record_think_blocks = 0
    record_answer_blocks = 0

    for message in messages:
        if not isinstance(message, dict):
            converted_messages.append(message)
            continue
        if message.get("role") != "assistant":
            converted_messages.append(message)
            continue

        split_messages, message_stats = split_assistant_message(
            message,
            args.think_loss_scale,
            args.answer_loss_scale,
            args.untagged_loss_scale,
            args.drop_empty_think,
        )
        converted_messages.extend(split_messages)
        for key in (
            "assistant_messages",
            "think_blocks",
            "answer_blocks",
            "untagged_assistant_messages",
            "dropped_empty_think",
        ):
            stats[key] += message_stats[key]
        record_think_blocks += message_stats["think_blocks"]
        record_answer_blocks += message_stats["answer_blocks"]

    if record_answer_blocks and not record_think_blocks:
        stats["answer_only_records"] += 1
    if record_answer_blocks and record_think_blocks:
        stats["think_answer_records"] += 1

    converted = dict(record)
    converted["messages"] = converted_messages
    return converted, stats


def add_stats(total: dict[str, int], current: dict[str, int]) -> None:
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_path}; pass --overwrite to replace it")

    total: dict[str, int] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"expected JSON object, got {type(record).__name__}")
                converted, stats = convert_record(record, args)
            except Exception as exc:
                raise ValueError(f"failed at {input_path}:{line_no}: {exc}") from exc
            add_stats(total, stats)
            dst.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":"), indent=args.indent))
            dst.write("\n")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "think_loss_scale": args.think_loss_scale,
        "answer_loss_scale": args.answer_loss_scale,
        "untagged_loss_scale": args.untagged_loss_scale,
        **total,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
