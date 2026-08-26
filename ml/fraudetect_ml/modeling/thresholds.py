from __future__ import annotations

from typing import Any

import numpy as np

from ml.fraudetect_ml.modeling.evaluation import metrics_at_threshold


def threshold_curve(target: np.ndarray, probabilities: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-probabilities, kind="stable")
    sorted_scores = probabilities[order]
    sorted_target = target[order].astype("int64")
    cumulative_true = np.cumsum(sorted_target)
    cumulative_false = np.cumsum(1 - sorted_target)
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    reviews = group_ends + 1
    true_positives = cumulative_true[group_ends]
    false_positives = cumulative_false[group_ends]
    total_fraud = max(1, int(sorted_target.sum()))
    precision = true_positives / reviews
    recall = true_positives / total_fraud
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype="float64"),
        where=(precision + recall) > 0,
    )
    return {
        "threshold": sorted_scores[group_ends],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "review_rate": reviews / len(target),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": int(sorted_target.sum()) - true_positives,
    }


def _best_index(curve: dict[str, np.ndarray], candidates: np.ndarray, keys: tuple[str, ...]) -> int:
    if not candidates.any():
        raise ValueError("No threshold candidates satisfy the operating-mode constraint")
    indices = np.flatnonzero(candidates)
    return int(
        max(
            indices,
            key=lambda index: tuple(float(curve[key][index]) for key in keys),
        )
    )


def select_threshold_policies(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Select three documented operating modes using validation data only."""

    curve = threshold_curve(target, probabilities)
    has_detection = curve["true_positives"] > 0
    high_precision_index = _best_index(
        curve,
        (curve["review_rate"] <= 0.001) & has_detection,
        ("precision", "recall", "threshold"),
    )
    balanced_index = _best_index(
        curve,
        has_detection,
        ("f1", "recall", "precision", "threshold"),
    )
    high_recall_index = _best_index(
        curve,
        (curve["review_rate"] <= 0.01) & has_detection,
        ("recall", "precision", "threshold"),
    )
    selected = {
        "HIGH_PRECISION": high_precision_index,
        "BALANCED": balanced_index,
        "HIGH_RECALL": high_recall_index,
    }
    policies: dict[str, dict[str, Any]] = {}
    for name, index in selected.items():
        metrics = metrics_at_threshold(target, probabilities, float(curve["threshold"][index]))
        metrics["selection_logic"] = {
            "HIGH_PRECISION": "Highest precision at no more than 0.1% validation review rate.",
            "BALANCED": "Maximum validation F1; ties prefer recall, precision, then threshold.",
            "HIGH_RECALL": "Highest recall at no more than 1.0% validation review rate.",
        }[name]
        policies[name] = metrics
    return policies


def downsample_threshold_curve(
    curve: dict[str, np.ndarray],
    *,
    max_points: int = 2_000,
) -> list[dict[str, float | int]]:
    size = len(curve["threshold"])
    indices = np.unique(np.linspace(0, size - 1, min(size, max_points), dtype=int))
    keys = tuple(curve)
    return [
        {
            key: (
                int(curve[key][index])
                if key in {"true_positives", "false_positives", "false_negatives"}
                else float(curve[key][index])
            )
            for key in keys
        }
        for index in indices
    ]

