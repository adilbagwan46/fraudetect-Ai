from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

REVIEW_CAPACITIES = (0.001, 0.005, 0.01)


def metrics_at_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    precision = precision_score(target, predictions, zero_division=0)
    recall = recall_score(target, predictions, zero_division=0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "review_rate": float(predictions.mean()),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def ranking_metrics_at_capacity(
    target: np.ndarray,
    probabilities: np.ndarray,
    review_rate: float,
) -> dict[str, Any]:
    """Evaluate exactly the top-k ranked transactions for an analyst capacity."""

    review_count = max(1, int(np.floor(len(target) * review_rate)))
    order = np.argsort(-probabilities, kind="stable")
    reviewed = order[:review_count]
    true_positives = int(target[reviewed].sum())
    total_fraud = int(target.sum())
    return {
        "target_review_rate": review_rate,
        "review_count": review_count,
        "actual_review_rate": review_count / len(target),
        "precision": true_positives / review_count,
        "recall": true_positives / total_fraud if total_fraud else 0.0,
        "true_positives": true_positives,
        "score_floor": float(probabilities[reviewed[-1]]),
        "methodology": "Stable descending risk ranking; evaluate exactly floor(N * rate) rows.",
    }


def probability_metrics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return {
        "pr_auc": float(average_precision_score(target, probabilities)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "class_prevalence": float(target.mean()),
        "rows": int(len(target)),
        "fraud_rows": int(target.sum()),
        "operating_points": {
            f"{rate * 100:.1f}%": ranking_metrics_at_capacity(target, probabilities, rate)
            for rate in REVIEW_CAPACITIES
        },
    }

