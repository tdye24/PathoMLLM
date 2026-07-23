"""Metric registry for eval scoring."""

from collections import Counter
from typing import Any, Callable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .postprocess import postprocess_prediction, postprocess_reference

try:
    from sacrebleu import BLEU
except ImportError:  # pragma: no cover
    BLEU = None

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None

MCQ_SKLEARN_METRICS: dict[str, tuple[Callable[..., float], str]] = {
    "macro_f1": (f1_score, "macro"),
    "micro_f1": (f1_score, "micro"),
    "macro_precision": (precision_score, "macro"),
    "micro_precision": (precision_score, "micro"),
    "macro_recall": (recall_score, "macro"),
    "micro_recall": (recall_score, "micro"),
}


def sklearn_class_metric(
    y_true: list[str],
    y_pred: list[str],
    fn: Callable[..., float],
    average: str,
) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    return float(fn(y_true, y_pred, average=average, labels=labels, zero_division=0))


def corpus_bleu(preds: list[str], refs: list[str], max_ngram_order: int) -> float:
    if BLEU is None:
        raise ImportError("sacrebleu is required for BLEU metrics: pip install sacrebleu")
    if not preds:
        return 0.0
    return BLEU(max_ngram_order=max_ngram_order).corpus_score(preds, [refs]).score / 100.0


def rouge_l(preds: list[str], refs: list[str]) -> float:
    if rouge_scorer is None:
        raise ImportError("rouge-score is required for ROUGE metrics: pip install rouge-score")
    if not preds:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores)


def mcq_per_class(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    out: dict[str, dict[str, int]] = {}
    for label in labels:
        support = sum(1 for t in y_true if t == label)
        pred_n = sum(1 for p in y_pred if p == label)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        out[label] = {"support": support, "pred": pred_n, "correct": correct}
    return out


def compute_metrics(
    records: list[dict[str, Any]],
    metrics: list[str],
    postprocess: str,
) -> dict[str, Any]:
    """Return ``{metrics: {acc, ...}, counts: {n, n_correct, support, ...}}``."""
    y_true: list[str] = []
    y_pred: list[str] = []
    text_true: list[str] = []
    text_pred: list[str] = []
    n_success = 0
    parse_failures = 0

    for rec in records:
        if rec.get("status") != "success":
            continue
        n_success += 1
        ref = postprocess_reference(str(rec["ground_truth"]), postprocess)
        pred_raw = rec.get("prediction") or rec.get("raw_prediction") or ""
        pred = postprocess_prediction(str(pred_raw), postprocess)
        if pred is None:
            parse_failures += 1
            if postprocess == "mcq":
                y_true.append(ref)
                y_pred.append("__PARSE_FAIL__")
            continue
        if postprocess == "mcq":
            y_true.append(ref)
            y_pred.append(pred)
        else:
            text_true.append(ref)
            text_pred.append(pred)

    n = len(records)
    counts: dict[str, Any] = {
        "n": n,
        "n_success": n_success,
        "n_scored": len(y_true) + len(text_true),
        "parse_failures": parse_failures,
    }
    rates: dict[str, float] = {}

    if postprocess == "mcq" and y_true:
        n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        counts["n_correct"] = n_correct
        counts["support"] = dict(Counter(y_true))
        counts["pred"] = dict(Counter(y_pred))
        counts["per_class"] = mcq_per_class(y_true, y_pred)

        if "acc" in metrics:
            rates["acc"] = float(accuracy_score(y_true, y_pred))
        if "bacc" in metrics:
            rates["bacc"] = sklearn_class_metric(y_true, y_pred, recall_score, "macro")
        if "f1" in metrics:
            rates["f1"] = sklearn_class_metric(y_true, y_pred, f1_score, "macro")
        for name, (fn, average) in MCQ_SKLEARN_METRICS.items():
            if name in metrics:
                rates[name] = sklearn_class_metric(y_true, y_pred, fn, average)

    if postprocess in {"free_text", "caption"} and text_true:
        n_exact = sum(1 for p, r in zip(text_pred, text_true) if p == r)
        counts["n_exact_match"] = n_exact
        if "bleu1" in metrics:
            rates["bleu1"] = corpus_bleu(text_pred, text_true, max_ngram_order=1)
        if "bleu4" in metrics:
            rates["bleu4"] = corpus_bleu(text_pred, text_true, max_ngram_order=4)
        if "rouge_l" in metrics:
            rates["rouge_l"] = rouge_l(text_pred, text_true)
        if "exact_match" in metrics:
            rates["exact_match"] = n_exact / len(text_true)

    for m in metrics:
        rates.setdefault(m, 0.0)

    return {"metrics": rates, "counts": counts}
