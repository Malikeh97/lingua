"""
Metric functions for evaluation.

Uses HuggingFace evaluate library for standard metrics.
"""

from typing import Callable, Dict, List, Tuple, Any

import evaluate


MetricFn = Callable[[List[str], List[Dict[str, Any]]], Dict[str, float]]
"""Signature: (predictions, example_misc) -> score"""


# Lazy-loaded metrics to avoid loading at import time
_SQUAD_METRIC = None
_ROUGE_METRIC = None
_BLEU_METRIC = None


def _get_squad_metric():
    global _SQUAD_METRIC
    if _SQUAD_METRIC is None:
        _SQUAD_METRIC = evaluate.load("squad")
    return _SQUAD_METRIC


def _get_rouge_metric():
    global _ROUGE_METRIC
    if _ROUGE_METRIC is None:
        _ROUGE_METRIC = evaluate.load("rouge")
    return _ROUGE_METRIC


def _get_bleu_metric():
    global _BLEU_METRIC
    if _BLEU_METRIC is None:
        _BLEU_METRIC = evaluate.load("bleu")
    return _BLEU_METRIC


def compute_squad_metrics(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Compute SQUAD-style EM and F1 metrics.

    Args:
        predictions: List of predicted answers
        references: List of lists of reference answers (multi-ref support)

    Returns:
        Dict with 'exact_match' and 'f1' scores (0-1 scale)
    """
    squad_metric = _get_squad_metric()

    # Format for squad metric
    formatted_predictions = [
        {"id": str(i), "prediction_text": pred}
        for i, pred in enumerate(predictions)
    ]
    formatted_references = [
        {"id": str(i), "answers": {"text": refs, "answer_start": [0] * len(refs)}}
        for i, refs in enumerate(references)
    ]

    results = squad_metric.compute(
        predictions=formatted_predictions,
        references=formatted_references,
    )

    return {
        "exact_match": results["exact_match"] / 100.0,  # Convert from percentage
        "f1": results["f1"] / 100.0,
    }


def compute_em_f1_multi_ref(
    predictions: List[str],
    ground_truths_list: List[List[str]],
) -> Tuple[float, float]:
    """
    Compute EM and F1 with multiple reference answers per example.

    Args:
        predictions: List of predicted answers
        ground_truths_list: List of lists of ground truth answers

    Returns:
        Tuple of (average_em, average_f1)
    """
    results = compute_squad_metrics(predictions, ground_truths_list)
    return results["exact_match"], results["f1"]


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE scores.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dict with rouge1, rouge2, rougeL, rougeLsum scores
    """
    rouge_metric = _get_rouge_metric()

    results = rouge_metric.compute(
        predictions=predictions,
        references=references,
    )

    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "rougeLsum": results["rougeLsum"],
    }


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Compute BLEU score.

    Args:
        predictions: List of predicted texts
        references: List of lists of reference texts

    Returns:
        Dict with 'bleu' score
    """
    bleu_metric = _get_bleu_metric()

    results = bleu_metric.compute(
        predictions=predictions,
        references=references,
    )

    return {
        "bleu": results["bleu"],
    }


# Registry of metric functions
_METRICS: Dict[str, MetricFn] = {}


def _em_metric(predictions: List[str], example_miscs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Exact match using 'all_answers' from misc."""
    refs = [m.get("all_answers", [m.get("target_text", "")]) for m in example_miscs]
    em, _ = compute_em_f1_multi_ref(predictions, refs)
    return {"exact_match": em}


def _f1_metric(predictions: List[str], example_miscs: List[Dict[str, Any]]) -> Dict[str, float]:
    """F1 using 'all_answers' from misc."""
    refs = [m.get("all_answers", [m.get("target_text", "")]) for m in example_miscs]
    _, f1 = compute_em_f1_multi_ref(predictions, refs)
    return {"f1": f1}


def _rouge_metric(predictions: List[str], example_miscs: List[Dict[str, Any]]) -> Dict[str, float]:
    """ROUGE using 'target_text' from misc."""
    refs = [m.get("target_text", "") for m in example_miscs]
    return compute_rouge(predictions, refs)


_METRICS["exact_match"] = _em_metric
_METRICS["f1"] = _f1_metric
_METRICS["rouge"] = _rouge_metric
_METRICS["rouge1"] = _rouge_metric
_METRICS["rouge2"] = _rouge_metric
_METRICS["rougeL"] = _rouge_metric


def get_metric(name: str) -> MetricFn:
    """
    Get metric function by name.

    Args:
        name: Metric name (e.g., "exact_match", "f1", "rouge")

    Returns:
        Metric function: (predictions, example_miscs) -> scores dict
    """
    if name not in _METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(_METRICS.keys())}")
    return _METRICS[name]
