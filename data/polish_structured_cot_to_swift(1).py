#!/usr/bin/env python3
"""Polish structured MCTS CoT paths and export Swift chat JSONL.

The input is a JSONL file whose records contain:
  id, question, image_path, reasoning_chain, answer

The output follows the sampled Swift CoT format:
  {"id": "...", "messages": [..., {"role": "assistant",
   "content": "<think>...\n\nThe answer is C.</think><answer>C. ...</answer>"}],
   "images": ["..."]}

The Qwen API is only used to polish the reasoning text. The answer letter is
always taken from the input record and appended by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

from openai import APIConnectionError, APITimeoutError, OpenAI


DEFAULT_INPUT = "tools/post_progress_cot/CCRCC_pos.jsonl"
DEFAULT_OUTPUT = "tools/post_progress_cot/CCRCC_pos_qwen38_swift_cot.jsonl"
DEFAULT_API_BASE_URL = "http://127.0.0.1:2573/v1"
DEFAULT_API_MODEL = "Qwen/Qwen3.8-27B"

ANSWER_TAIL_RE = re.compile(
    r"(?:\n\s*)*The\s+answer\s+is\s+\[?\s*([A-Za-z])\s*\]?\.\s*$",
    re.IGNORECASE,
)
ANSWER_ANY_RE = re.compile(
    r"The\s+answer\s+is\s+\[?\s*([A-Za-z])\s*\]?\.",
    re.IGNORECASE,
)
OPTION_RE = re.compile(
    r"(?ms)^\s*([A-Z])\.\s*(.*?)(?=^\s*[A-Z]\.\s+|\Z)"
)
GENERIC_ROOT_PATTERNS = (
    "let's inspect the pathology image systematically",
    "derive the diagnosis from visible morphological evidence",
)


SYSTEM_PROMPT = """You are an expert pathology reasoning editor.
Rewrite structured pathology reasoning into a smooth, concise chain of thought.
Preserve the meaning of every provided visual finding.
Do not add new visual findings, stains, immunomarkers, organs, clinical history, labels, or molecular findings.
Do not change the final answer.
Do not mention node names, scoring, MCTS, or that text was rewritten.
Return only valid JSON in the form {"cot": "..."}."""

# Initial writing requirements kept for rollback:
# 1. Treat the detailed observation nodes as careful microscopic observations.
# 2. Treat the finish-node evidence as the decisive evidence that ties the observations to the final diagnosis.
# 3. Merge duplicated wording naturally instead of repeating the same morphology twice.
# 4. Keep the reasoning image-grounded and concise, usually 2 to 4 sentences.
# 5. Do not include <think>, </think>, <answer>, option labels as a final line, or "The answer is ...".
# 6. Do not output Markdown.

USER_PROMPT_TEMPLATE = """Polish the pathology reasoning for a multiple-choice VQA sample.

Question and choices:
{question}

Final answer fixed by the dataset:
{answer}. {answer_text}

Detailed observation nodes:
{observations}

Key evidence from the finish node:
{key_evidence}

Writing requirements:
1. Treat the detailed observation nodes as careful microscopic observations.
2. Treat the finish-node evidence as the decisive evidence that ties the observations to the final diagnosis.
3. Merge duplicated wording naturally instead of repeating the same morphology twice.
4. Make the reasoning coherent and logical, but do not add new evidence.
5. Do not include <think>, </think>, <answer>, option labels as a final line, or "The answer is ...".
6. Do not output Markdown.

Return only:
{{"cot": "polished reasoning text"}}"""


def iter_json_values(text: str, path: Path, line_no: int) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length:
            if text[index].isspace() or text[index] in {",", ";", "\ufeff", "\x00"}:
                index += 1
                continue
            if text.startswith("/n", index) or text.startswith("\\n", index):
                index += 2
                continue
            if text.startswith("/r", index) or text.startswith("\\r", index):
                index += 2
                continue
            break
        if index >= length:
            break
        if text[index] not in "{[":
            next_obj = min(
                [pos for pos in (text.find("{", index + 1), text.find("[", index + 1)) if pos != -1],
                default=-1,
            )
            if next_obj == -1:
                context = text[max(0, index - 80): min(length, index + 80)]
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: expected JSON object at char {index}; "
                    f"context={context!r}"
                )
            index = next_obj
        try:
            obj, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            next_obj = min(
                [pos for pos in (text.find("{", index + 1), text.find("[", index + 1)) if pos != -1],
                default=-1,
            )
            if next_obj == -1:
                context = text[max(0, index - 80): min(length, index + 80)]
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}; context={context!r}"
                ) from exc
            index = next_obj
            continue
        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Expected object in JSON array at {path}:{line_no}, "
                        f"got {type(item).__name__}"
                    )
                yield item
        elif isinstance(obj, dict):
            yield obj
        else:
            raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj).__name__}")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            for obj in iter_json_values(line, path, line_no):
                yield line_no, obj


def clean_spaces(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_answer_tail(text: str) -> tuple[str, str | None]:
    text = clean_spaces(text)
    match = ANSWER_TAIL_RE.search(text)
    if not match:
        return text, None
    return text[: match.start()].strip(), match.group(1).upper()


def is_generic_root(text: str) -> bool:
    lower = clean_spaces(text).lower()
    return any(pattern in lower for pattern in GENERIC_ROOT_PATTERNS)


def parse_choices(question: str) -> dict[str, str]:
    choices = {}
    for match in OPTION_RE.finditer(str(question or "")):
        letter = match.group(1).upper()
        option_text = clean_spaces(match.group(2))
        if option_text:
            choices[letter] = option_text
    return choices


def require_field(record: dict, field: str, source: str):
    value = record.get(field)
    if value is None:
        raise ValueError(f"Missing required field {field!r} at {source}")
    return value


def structured_path(record: dict, source: str, skip_root: bool = True) -> tuple[list[str], str]:
    chain = require_field(record, "reasoning_chain", source)
    if not isinstance(chain, list):
        raise ValueError(f"reasoning_chain must be a list at {source}")

    observations = []
    key_evidence = ""
    for index, node in enumerate(chain):
        if not isinstance(node, dict):
            continue
        raw_text = node.get("step_text")
        if not raw_text:
            continue
        text, answer = strip_answer_tail(str(raw_text))
        if skip_root and index == 0 and is_generic_root(text):
            continue
        if answer is not None:
            if text:
                key_evidence = text
            continue
        if text:
            observations.append(text)

    if not key_evidence and observations:
        key_evidence = observations[-1]
    if not observations and key_evidence:
        observations = [key_evidence]
    return observations, key_evidence


def numbered_lines(items: list[str]) -> str:
    if not items:
        return "None"
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))


def build_prompt(record: dict, source: str) -> tuple[str, str, str, list[str], str]:
    question = str(require_field(record, "question", source))
    answer = str(require_field(record, "answer", source)).strip().upper()
    choices = parse_choices(question)
    answer_text = choices.get(answer, "")
    observations, key_evidence = structured_path(record, source)
    prompt = USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        answer=answer,
        answer_text=answer_text or "(option text not parsed)",
        observations=numbered_lines(observations),
        key_evidence=key_evidence or "None",
    )
    return prompt, answer, answer_text, observations, key_evidence


def extract_json_object(text: str) -> dict:
    text = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start: end + 1])

    last_error = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise ValueError("model response did not contain a JSON object")


def sanitize_cot(text: str) -> str:
    text = clean_spaces(text)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    text = ANSWER_ANY_RE.sub("", text)
    text = re.sub(r"^\s*[A-D]\s*(?:[.)]\s*[^.\n]+)?\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = clean_spaces(text)
    return text


def fallback_cot(observations: list[str], key_evidence: str) -> str:
    parts = []
    for item in observations:
        if item and item not in parts:
            parts.append(item)
    if key_evidence and key_evidence not in parts:
        parts.append(key_evidence)
    return "\n\n".join(parts).strip()


def call_model(client: OpenAI, args: argparse.Namespace, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    extra_body = None if args.no_extra_body else {"chat_template_kwargs": {"enable_thinking": False}}
    last_error = None
    for attempt in range(args.api_retries + 1):
        try:
            kwargs = {
                "model": args.api_model,
                "messages": messages,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except (APITimeoutError, APIConnectionError) as exc:
            last_error = exc
            if attempt >= args.api_retries:
                break
            time.sleep(args.api_retry_delay * (attempt + 1))
    raise last_error


def polish_record(client: OpenAI, args: argparse.Namespace, record: dict, source: str) -> tuple[str, dict]:
    prompt, answer, answer_text, observations, key_evidence = build_prompt(record, source)
    debug = {
        "id": record.get("id"),
        "source": source,
        "status": "ok",
        "raw_response": None,
        "fallback": False,
    }
    if args.dry_run:
        print(prompt)
        raise SystemExit(0)

    try:
        raw_response = call_model(client, args, prompt)
        debug["raw_response"] = raw_response
        obj = extract_json_object(raw_response)
        cot = sanitize_cot(str(obj.get("cot", "")))
        if not cot:
            raise ValueError("empty cot after sanitize")
    except Exception as exc:
        if not args.fallback_on_error:
            raise
        cot = sanitize_cot(fallback_cot(observations, key_evidence))
        debug["status"] = f"fallback:{type(exc).__name__}:{exc}"
        debug["fallback"] = True

    if not cot:
        raise ValueError(f"Empty polished CoT at {source}")
    debug["cot"] = cot
    debug["answer"] = answer
    debug["answer_text"] = answer_text
    return cot, debug


def answer_payload(answer: str, answer_text: str) -> str:
    if answer_text:
        return f"{answer}. {answer_text}"
    return answer


def convert_record(record: dict, cot: str, source: str) -> dict:
    sample_id = require_field(record, "id", source)
    question = str(require_field(record, "question", source)).strip()
    image_path = require_field(record, "image_path", source)
    answer = str(require_field(record, "answer", source)).strip().upper()
    answer_text = parse_choices(question).get(answer, "")
    assistant_content = (
        f"<think>{cot.strip()}\n\n"
        f"The answer is {answer}.</think>"
        f"<answer>{answer_payload(answer, answer_text)}</answer>"
    )
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": "<image>\n" + question},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = obj.get("id")
            if sample_id is not None:
                ids.add(str(sample_id))
    return ids


def should_keep_index(index: int, num_chunks: int, chunk_idx: int) -> bool:
    if num_chunks <= 1:
        return True
    return index % num_chunks == chunk_idx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Polish structured MCTS CoT paths with Qwen API and export Swift JSONL."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input structured CoT JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output Swift JSONL.")
    parser.add_argument("--debug_output", default=None, help="Optional JSONL with raw model responses and status.")
    parser.add_argument("--limit", type=int, default=None, help="Only generate N records after chunking/resume.")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    parser.add_argument("--dry_run", action="store_true", help="Print the first prompt and exit.")

    parser.add_argument("--api_base_url", default=os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL))
    parser.add_argument("--api_model", default=os.getenv("API_MODEL", DEFAULT_API_MODEL))
    parser.add_argument("--api_key_env", default=os.getenv("API_KEY_ENV", "DASHSCOPE_API_KEY"))
    parser.add_argument("--api_key", default=None, help="Defaults to --api_key_env, OPENAI_API_KEY, or dummy.")
    parser.add_argument("--api_timeout", type=float, default=float(os.getenv("API_TIMEOUT_SECONDS")) if os.getenv("API_TIMEOUT_SECONDS") else None)
    parser.add_argument("--api_retries", type=int, default=int(os.getenv("API_RETRIES", "3")))
    parser.add_argument("--api_retry_delay", type=float, default=float(os.getenv("API_RETRY_DELAY", "5")))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each API call.")
    parser.add_argument("--no_extra_body", action="store_true", help="Do not send Qwen enable_thinking=False extra_body.")
    parser.add_argument("--fallback_on_error", action="store_true", default=True)
    parser.add_argument("--no_fallback_on_error", action="store_false", dest="fallback_on_error")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_chunks < 1:
        raise ValueError("--num_chunks must be >= 1")
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError("--chunk_idx must satisfy 0 <= chunk_idx < num_chunks")

    input_path = Path(args.input)
    output_path = Path(args.output)
    debug_path = Path(args.debug_output) if args.debug_output else None

    api_key = args.api_key or os.getenv(args.api_key_env) or os.getenv("OPENAI_API_KEY") or "dummy"
    client_kwargs = {"api_key": api_key, "base_url": args.api_base_url}
    if args.api_timeout is not None:
        client_kwargs["timeout"] = args.api_timeout
    client = OpenAI(**client_kwargs)

    done_ids = existing_ids(output_path) if args.resume else set()
    generated = 0
    skipped_existing = 0
    skipped_chunk = 0
    fallback_count = 0

    for index, (line_no, record) in enumerate(iter_jsonl(input_path)):
        if not should_keep_index(index, args.num_chunks, args.chunk_idx):
            skipped_chunk += 1
            continue
        sample_id = str(record.get("id", index))
        if sample_id in done_ids:
            skipped_existing += 1
            continue
        if args.limit is not None and generated >= args.limit:
            break

        source = f"{input_path}:{line_no}"
        cot, debug = polish_record(client, args, record, source)
        converted = convert_record(record, cot, source)
        append_jsonl(output_path, converted)
        if debug_path is not None:
            append_jsonl(debug_path, debug)
        fallback_count += int(bool(debug.get("fallback")))
        generated += 1
        print(
            f"{generated} id={sample_id} status={debug['status']} answer={debug.get('answer')}",
            flush=True,
        )
        if args.sleep:
            time.sleep(args.sleep)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "debug_output": str(debug_path) if debug_path else None,
        "api_base_url": args.api_base_url,
        "api_model": args.api_model,
        "generated_count": generated,
        "fallback_count": fallback_count,
        "skipped_existing_count": skipped_existing,
        "skipped_chunk_count": skipped_chunk,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
