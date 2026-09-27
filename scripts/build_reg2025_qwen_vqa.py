#!/usr/bin/env python3
"""Generate REG2025 multi-turn VQA conversations with local Qwen."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None  # type: ignore[assignment]


GENERATOR_SYSTEM = """You are a pathology VQA dataset constructor.
Use only facts explicitly stated in the pathology report.
Return only valid JSON."""

VALIDATOR_SYSTEM = """You are a strict pathology VQA reviewer.
Fix or remove generated QA turns that are not directly supported by the report.
Return only valid JSON."""

GENERATOR_PROMPT = """Create a multi-turn VQA conversation for one whole-slide image from this REG2025 pathology report.

REG2025 report format:
- Most reports follow: "Organ, procedure; diagnosis content".
- The header before the semicolon contains the organ and specimen procedure.
- The body after the semicolon contains diagnosis content, sometimes as numbered items.
- Some reports contain a "Note)" section.
- Diagnosis content may include histologic type, grade, scores, or quantitative fields.

Common fields:
- Prostate: Gleason score, patterns, grade group, pattern percentage, tumor volume.
- Breast: Nottingham grade, tubule formation, nuclear grade, mitoses, necrosis, microcalcification.
- Bladder: invasive/non-invasive urothelial carcinoma, carcinoma in situ, no tumor, muscle proper status.
- Cervix: LSIL, HSIL, CIN grade, squamous cell carcinoma, AIS.
- Colon/rectum/stomach: adenocarcinoma differentiation, adenoma, dysplasia, inflammation, lymphoma.
- Lung: adenocarcinoma, squamous cell carcinoma, small cell carcinoma, granulomatous inflammation, no malignancy.

Report:
{report}

Requirements:
1. Generate 3 to {max_turns} QA turns.
2. Include organ, procedure, and main diagnosis if available.
3. Ask organ-specific grading, score, quantitative, note, or status questions only when explicitly stated.
4. Do not ask about unmentioned fields.
5. Keep answers short and faithful to the report.

Return exactly:
{{
  "turns": [
    {{"question": "__question__", "answer": "__answer__"}}
  ]
}}"""

VALIDATOR_PROMPT = """Validate these REG2025 VQA turns against the source report.

Source report:
{report}

Generated turns:
{turns_json}

Rules:
1. Keep only turns explicitly supported by the report.
2. Remove turns about unmentioned fields.
3. Correct any changed number, score, percentage, grade, diagnosis, organ, procedure, or note.
4. Keep 3 to {max_turns} turns when possible.

Return exactly:
{{
  "status": "pass" | "fixed" | "fail",
  "issues": ["short issue strings"],
  "turns": [
    {{"question": "__question__", "answer": "__answer__"}}
  ]
}}"""


def clean(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_json(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("input must be a JSON list")
    return data


def json_from_response(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        text = match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("model response JSON is not an object")
    return obj


def turns_from(obj: dict[str, Any], max_turns: int) -> list[dict[str, str]]:
    turns = []
    for item in obj.get("turns", []):
        if not isinstance(item, dict):
            continue
        question = clean(item.get("question"))
        answer = clean(item.get("answer"))
        if question and answer:
            turns.append({"question": question, "answer": answer})
        if len(turns) >= max_turns:
            break
    return turns


def ask_qwen(client: Any, args: argparse.Namespace, system: str, prompt: str) -> str:
    kwargs: dict[str, Any] = {
        "model": args.api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if not args.no_extra_body:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return client.chat.completions.create(**kwargs).choices[0].message.content or ""


def build_sample(case_id: str, image: str, turns: list[dict[str, str]]) -> dict[str, Any]:
    messages = []
    for i, turn in enumerate(turns):
        question = turn["question"]
        if i == 0:
            question = "<image>\n" + question
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": f"<answer>{turn['answer']}</answer>"})
    return {
        "id": f"REG2025_{Path(case_id).stem}",
        "case_id": case_id,
        "source": "REG2025",
        "task_type": "multiturn_vqa",
        "messages": messages,
        "images": [image],
    }


def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def image_path(case_id: str, image_root: str) -> str:
    return str(Path(image_root) / case_id) if image_root else case_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="papers/train.json")
    parser.add_argument("--output", default="papers/reg2025_qwen_multiturn_vqa.jsonl")
    parser.add_argument("--debug_output", default="papers/reg2025_qwen_multiturn_debug.jsonl")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min_turns", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--api_base_url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:9005/v1"))
    parser.add_argument("--api_model", default=os.getenv("API_MODEL", "ascendc-kernel"))
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--no_extra_body", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = read_json(args.input)
    first_prompt = GENERATOR_PROMPT.format(report=clean(records[0]["report"]), max_turns=args.max_turns)
    if args.dry_run:
        print(first_prompt)
        return 0
    if OpenAI is None:
        raise ModuleNotFoundError("Install openai or run with the same Python environment used for scripts/msr.py.")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)
    generated = 0
    for record in records:
        if args.limit is not None and generated >= args.limit:
            break
        case_id = str(record["id"])
        report = clean(record["report"])

        raw = ask_qwen(client, args, GENERATOR_SYSTEM, GENERATOR_PROMPT.format(report=report, max_turns=args.max_turns))
        generated_obj = json_from_response(raw)
        validator_prompt = VALIDATOR_PROMPT.format(
            report=report,
            turns_json=json.dumps(generated_obj.get("turns", []), ensure_ascii=False, indent=2),
            max_turns=args.max_turns,
        )
        reviewed_raw = ask_qwen(client, args, VALIDATOR_SYSTEM, validator_prompt)
        reviewed_obj = json_from_response(reviewed_raw)
        turns = turns_from(reviewed_obj, args.max_turns)

        status = "ok" if len(turns) >= args.min_turns else "skipped"
        if status == "ok":
            append_jsonl(args.output, build_sample(case_id, image_path(case_id, args.image_root), turns))
            generated += 1

        append_jsonl(
            args.debug_output,
            {
                "case_id": case_id,
                "status": status,
                "report": report,
                "generated_response": raw,
                "reviewed_response": reviewed_raw,
                "turns": turns,
            },
        )
        print(f"{generated} {case_id} {status} turns={len(turns)}", flush=True)

    print(json.dumps({"output": args.output, "debug_output": args.debug_output, "generated": generated}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
