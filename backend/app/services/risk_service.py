from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.app.schemas.risk import RiskPredictionRequest, RiskPredictionResponse


class ModelUnavailableError(RuntimeError):
    """Raised when no complete frozen model bundle can be loaded."""


@dataclass(frozen=True)
class LoadedModelBundle:
    model: Any
    metadata: dict[str, Any]
    threshold_policy: dict[str, Any]
    evaluation: dict[str, Any]
    model_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelUnavailableError(f"Unreadable model artifact: {path.name}") from error


def load_active_bundle(artifact_root: Path) -> LoadedModelBundle:
    latest = _read_json(artifact_root / "latest.json")
    model_version = latest.get("model_version")
    if not isinstance(model_version, str) or not model_version:
        raise ModelUnavailableError("Active model pointer has no version")
    model_dir = artifact_root / model_version
    try:
        model = joblib.load(model_dir / "model.joblib")
    except (OSError, ValueError, EOFError) as error:
        raise ModelUnavailableError("Frozen model cannot be loaded") from error
    return LoadedModelBundle(
        model=model,
        metadata=_read_json(model_dir / "metadata.json"),
        threshold_policy=_read_json(model_dir / "threshold-policy.json"),
        evaluation={
            "validation": _read_json(model_dir / "validation-metrics.json"),
            "test": _read_json(model_dir / "test-metrics.json"),
            "candidate_comparison": _read_json(model_dir / "candidate-comparison.json"),
            "operating_points": _read_json(model_dir / "operating-points.json"),
        },
        model_dir=model_dir,
    )


def prediction_frame(request: RiskPredictionRequest) -> pd.DataFrame:
    amount_to_balance = (
        request.amount / request.origin_balance_before
        if request.origin_balance_before > 0
        else 0.0
    )
    return pd.DataFrame(
        [
            {
                "transaction_type": request.transaction_type,
                "amount": request.amount,
                "origin_balance_before": request.origin_balance_before,
                "hour_of_day": request.hour_of_day,
                "log_amount": math.log1p(request.amount),
                "amount_to_origin_balance": amount_to_balance,
            }
        ]
    )


def predict_risk(
    bundle: LoadedModelBundle,
    request: RiskPredictionRequest,
) -> RiskPredictionResponse:
    probability = float(bundle.model.predict_proba(prediction_frame(request))[0, 1])
    policies = bundle.threshold_policy["policies_selected_on_validation"]
    recommended_mode = bundle.threshold_policy["recommended_mode"]
    threshold = float(policies[recommended_mode]["threshold"])
    high_threshold = max(threshold, float(policies["HIGH_PRECISION"]["threshold"]))
    if probability >= high_threshold:
        risk_level = "HIGH"
        action = "HOLD_FOR_INVESTIGATION"
    elif probability >= threshold:
        risk_level = "MEDIUM"
        action = "MANUAL_REVIEW"
    else:
        risk_level = "LOW"
        action = "NORMAL_PROCESSING"
    return RiskPredictionResponse(
        fraud_probability=probability,
        risk_score=probability * 100,
        risk_level=risk_level,
        fraud_prediction=probability >= threshold,
        classification_threshold=threshold,
        operating_mode="BALANCED",
        model_version=bundle.metadata["model_version"],
        recommended_action=action,
    )

