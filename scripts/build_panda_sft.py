#!/usr/bin/env python3
"""Normalize PANDA VQA JSONL into Qwen/ms-swift SFT chat format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"^\s*<answer>\s*(.*?)\s*</answer>\s*$", re.I | re.S)
IMAGE_TAG_RE = re.compile(r"^(?:\s*<image>\s*)+")


def clean(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number}: JSON value must be an object")
            records.append(obj)
    return records


def normalize_answer(content: Any) -> str:
    text = clean(content)
    match = ANSWER_RE.fullmatch(text)
    if match:
        text = clean(match.group(1))
    if not text:
        raise ValueError("assistant answer is empty")
    return f"<answer>{text}</answer>"


def normalize_user(content: Any, image_count: int) -> str:
    text = clean(content)
    text = IMAGE_TAG_RE.sub("", text).strip()
    prefix = "\n".join("<image>" for _ in range(image_count))
    return f"{prefix}\n{text}" if text else prefix


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    sample_id = record.get("id")
    if not sample_id:
        raise ValueError("missing id")

    images = record.get("images")
    if not isinstance(images, list) or not images or not all(isinstance(x, str) and x for x in images):
        raise ValueError(f"{sample_id}: images must be a non-empty string list")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{sample_id}: messages must contain at least user and assistant turns")

    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{sample_id}: every message must be an object")
        role = message.get("role")
        if role == "user":
            content = normalize_user(message.get("content"), len(images))
        elif role == "assistant":
            content = normalize_answer(message.get("content"))
        else:
            raise ValueError(f"{sample_id}: unsupported role {role!r}")
        normalized_messages.append({"role": role, "content": content})

    if not any(m["role"] == "assistant" for m in normalized_messages):
        raise ValueError(f"{sample_id}: no assistant message")

    return {
        "id": str(sample_id),
        "messages": normalized_messages,
        "images": images,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    raw_records = read_jsonl(args.input)
    if args.limit is not None:
        raw_records = raw_records[: args.limit]

    normalized = []
    skipped = 0
    for index, record in enumerate(raw_records, 1):
        try:
            normalized.append(normalize_record(record))
        except ValueError as exc:
            skipped += 1
            print(f"skip record {index}: {exc}")

    write_jsonl(args.output, normalized)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "written": len(normalized), "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
