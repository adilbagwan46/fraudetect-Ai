from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance

from ml.fraudetect_ml.data.contracts import ML_FEATURE_COLUMNS, ML_TARGET_COLUMN
from ml.fraudetect_ml.modeling.data import load_prepared_split

PERCENTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999)


def _percentiles(series: pd.Series) -> dict[str, float]:
    values = series.quantile(PERCENTILES)
    return {f"{percentile:.3f}": float(values.loc[percentile]) for percentile in PERCENTILES}


def _distribution(series: pd.Series) -> dict[str, Any]:
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "percentiles": _percentiles(series),
    }


def _group_profile(frame: pd.DataFrame, group_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, group in frame.groupby(group_column, observed=True, sort=True):
        fraud_rows = int(group[ML_TARGET_COLUMN].sum())
        result[str(value)] = {
            "rows": len(group),
            "share": len(group) / len(frame),
            "fraud_rows": fraud_rows,
            "historical_fraud_prevalence": fraud_rows / len(group),
            "amount": _distribution(group["amount"]),
            "amount_to_origin_balance": _distribution(group["amount_to_origin_balance"]),
        }
    return result


def _type_hour_profile(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    type_counts = frame.groupby("transaction_type", observed=True).size().to_dict()
    grouped = frame.groupby(["transaction_type", "hour_of_day"], observed=True, sort=True)
    for (transaction_type, hour), group in grouped:
        type_key = str(transaction_type)
        fraud_rows = int(group[ML_TARGET_COLUMN].sum())
        result.setdefault(type_key, {})[str(int(hour))] = {
            "rows": len(group),
            "share_within_type": len(group) / int(type_counts[transaction_type]),
            "fraud_rows": fraud_rows,
            "historical_fraud_prevalence": fraud_rows / len(group),
        }
    return result


def derive_reference_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    """Derive factual statistics from an approved training frame only."""

    required = set((*ML_FEATURE_COLUMNS, ML_TARGET_COLUMN, "step"))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Reference frame is missing columns: {', '.join(missing)}")
    fraud_rows = int(frame[ML_TARGET_COLUMN].sum())
    hour_profile: dict[str, Any] = {}
    for hour, group in frame.groupby("hour_of_day", sort=True):
        hour_fraud_rows = int(group[ML_TARGET_COLUMN].sum())
        hour_profile[str(int(hour))] = {
            "rows": len(group),
            "share": len(group) / len(frame),
            "fraud_rows": hour_fraud_rows,
            "historical_fraud_prevalence": hour_fraud_rows / len(group),
        }
    return {
        "rows": len(frame),
        "fraud_rows": fraud_rows,
        "historical_fraud_prevalence": fraud_rows / len(frame),
        "amount": _distribution(frame["amount"]),
        "origin_balance_before": _distribution(frame["origin_balance_before"]),
        "amount_to_origin_balance": _distribution(frame["amount_to_origin_balance"]),
        "transaction_types": _group_profile(frame, "transaction_type"),
        "hours": hour_profile,
        "transaction_type_hours": _type_hour_profile(frame),
    }


def validate_training_reference_path(
    training_path: Path,
    *,
    manifest: dict[str, Any],
) -> None:
    expected = Path(manifest["splits"]["train"]["path"]).resolve()
    if training_path.resolve() != expected:
        raise ValueError(
            "Reference profiles may only be derived from the manifest-approved training split"
        )


def global_permutation_importance(
    model: Any,
    frame: pd.DataFrame,
    *,
    sample_size: int,
    random_state: int,
) -> dict[str, Any]:
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=random_state)
    features = sample.loc[:, ML_FEATURE_COLUMNS]
    target = sample[ML_TARGET_COLUMN]
    result = permutation_importance(
        model,
        features,
        target,
        scoring="average_precision",
        n_repeats=3,
        random_state=random_state,
        n_jobs=1,
    )
    raw = {
        feature: {
            "importance_mean": float(result.importances_mean[index]),
            "importance_std": float(result.importances_std[index]),
        }
        for index, feature in enumerate(ML_FEATURE_COLUMNS)
    }
    positive_total = sum(max(0.0, item["importance_mean"]) for item in raw.values())
    for item in raw.values():
        item["normalized_positive_importance"] = (
            max(0.0, item["importance_mean"]) / positive_total if positive_total else 0.0
        )
    return {
        "method": "permutation_importance",
        "scoring": "average_precision",
        "repeats": 3,
        "sample_rows": len(sample),
        "sample_fraud_rows": int(target.sum()),
        "sample_source": "approved_training_split_only",
        "random_state": random_state,
        "interpretation": (
            "Global predictive reliance on the sampled training reference; not causal and not "
            "a transaction-level contribution decomposition."
        ),
        "features": raw,
    }


def build_reference_profile(
    *,
    training_path: Path,
    manifest_path: Path,
    model: Any,
    model_version: str,
    sample_size: int = 250_000,
    random_state: int = 20260826,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_training_reference_path(training_path, manifest=manifest)
    frame = load_prepared_split(training_path)
    statistics = derive_reference_statistics(frame)
    importance = global_permutation_importance(
        model,
        frame,
        sample_size=sample_size,
        random_state=random_state,
    )
    profile: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "dataset": {
            "kind": manifest["source"]["kind"],
            "sha256": manifest["source"]["sha256"],
        },
        "source_boundary": {
            "split": "train",
            "path": str(training_path),
            "min_step": int(frame["step"].min()),
            "max_step": int(frame["step"].max()),
            "validation_used": False,
            "test_used": False,
        },
        "statistics": statistics,
        "global_model_importance": importance,
        "limitations": [
            "Statistics describe public synthetic PaySim training data, not live merchant traffic.",
            "Historical prevalence is context, not a probability for an individual transaction.",
            "Permutation importance is global predictive reliance, not causal attribution.",
        ],
    }
    fingerprint_payload = json.dumps(
        {
            "model_version": model_version,
            "dataset_sha256": manifest["source"]["sha256"],
            "statistics": statistics,
            "global_model_importance": importance,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    profile["reference_profile_version"] = (
        f"reference-{hashlib.sha256(fingerprint_payload).hexdigest()[:12]}"
    )
    return profile
