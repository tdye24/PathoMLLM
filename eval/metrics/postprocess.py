"""Post-process predictions and ground_truth before metric computation."""

from __future__ import annotations

import re

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
_LETTER = re.compile(r"\b([A-Z])\b")
_PAREN = re.compile(r"\(([A-Z])\)", re.I)
_THINK = "</think>"


def postprocess_prediction(text: str, mode: str) -> str | None:
    body = text.strip()
    if _THINK in body:
        body = body.split(_THINK, 1)[-1].strip()
    if m := _ANSWER.search(body):
        body = m.group(1).strip()

    if mode == "mcq":
        if len(body) == 1 and body.isalpha():
            return body.upper()
        if m := _LETTER.search(body):
            return m.group(1).upper()
        if m := _PAREN.search(body):
            return m.group(1).upper()
        return None

    if mode in {"free_text", "caption"}:
        body = re.sub(r"[^\w\s]", " ", body.lower())
        return re.sub(r"\s+", " ", body).strip()

    raise ValueError(mode)


def postprocess_reference(text: str, mode: str) -> str:
    if mode == "mcq":
        letter = str(text).strip().upper()
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"MCQ ground_truth must be one letter, got {text!r}")
        return letter
    if mode in {"free_text", "caption"}:
        return postprocess_prediction(str(text), mode) or ""
    raise ValueError(mode)
