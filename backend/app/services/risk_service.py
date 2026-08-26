from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.app.schemas.risk import (
    BehavioralContext,
    DerivedFeatures,
    InvestigationContext,
    ModelOutputContext,
    RecommendedAction,
    RelationshipContext,
    RiskPredictionRequest,
    RiskPredictionResponse,
)
from backend.app.services.evidence_service import build_investigation_context

MAX_AMOUNT_TO_BALANCE = 1_000_000_000_000.0


class ModelUnavailableError(RuntimeError):
    """Raised when no complete frozen model bundle can be loaded."""


@dataclass(frozen=True)
class LoadedModelBundle:
    model: Any
    metadata: dict[str, Any]
    threshold_policy: dict[str, Any]
    evaluation: dict[str, Any]
    model_dir: Path
    reference_profile: dict[str, Any] | None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelUnavailableError(f"Unreadable model artifact: {path.name}") from error


def load_active_bundle(
    artifact_root: Path,
    *,
    require_reference_profile: bool = True,
) -> LoadedModelBundle:
    latest = _read_json(artifact_root / "latest.json")
    model_version = latest.get("model_version")
    if not isinstance(model_version, str) or not model_version:
        raise ModelUnavailableError("Active model pointer has no version")
    model_dir = artifact_root / model_version
    try:
        model = joblib.load(model_dir / "model.joblib")
    except (OSError, ValueError, EOFError) as error:
        raise ModelUnavailableError("Frozen model cannot be loaded") from error
    reference_profile_path = model_dir / "reference-profile.json"
    reference_profile = (
        _read_json(reference_profile_path) if reference_profile_path.is_file() else None
    )
    if require_reference_profile and reference_profile is None:
        raise ModelUnavailableError("Evidence reference profile is unavailable")
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
        reference_profile=reference_profile,
    )


def derived_features(request: RiskPredictionRequest) -> DerivedFeatures:
    if request.origin_balance_before > 0:
        raw_ratio = request.amount / request.origin_balance_before
        amount_to_balance = min(
            raw_ratio if math.isfinite(raw_ratio) else MAX_AMOUNT_TO_BALANCE,
            MAX_AMOUNT_TO_BALANCE,
        )
    else:
        amount_to_balance = 0.0
    return DerivedFeatures(
        log_amount=math.log1p(request.amount),
        amount_to_origin_balance=amount_to_balance,
    )


def prediction_frame(
    request: RiskPredictionRequest,
    derived: DerivedFeatures | None = None,
) -> pd.DataFrame:
    derived = derived or derived_features(request)
    return pd.DataFrame(
        [
            {
                "transaction_type": request.transaction_type,
                "amount": request.amount,
                "origin_balance_before": request.origin_balance_before,
                "hour_of_day": request.hour_of_day,
                "log_amount": derived.log_amount,
                "amount_to_origin_balance": derived.amount_to_origin_balance,
            }
        ]
    )


def score_transaction(
    bundle: LoadedModelBundle,
    request: RiskPredictionRequest,
) -> tuple[DerivedFeatures, ModelOutputContext, RecommendedAction]:
    derived = derived_features(request)
    probability = float(bundle.model.predict_proba(prediction_frame(request, derived))[0, 1])
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
    return (
        derived,
        ModelOutputContext(
            fraud_probability=probability,
            risk_score=probability * 100,
            risk_level=risk_level,
            fraud_prediction=probability >= threshold,
            classification_threshold=threshold,
            operating_mode="BALANCED",
            model_version=bundle.metadata["model_version"],
        ),
        action,
    )


def investigate_risk(
    bundle: LoadedModelBundle,
    request: RiskPredictionRequest,
    *,
    behavioral_context: BehavioralContext | None = None,
    relationship_context: RelationshipContext | None = None,
) -> tuple[InvestigationContext, RecommendedAction]:
    if bundle.reference_profile is None:
        raise ModelUnavailableError("Evidence reference profile is unavailable")
    derived, model_output, action = score_transaction(bundle, request)
    context = build_investigation_context(
        transaction=request,
        derived_features=derived,
        model_output=model_output,
        reference_profile=bundle.reference_profile,
        behavioral_context=behavioral_context,
        relationship_context=relationship_context,
    )
    return context, action


def predict_risk(
    bundle: LoadedModelBundle,
    request: RiskPredictionRequest,
) -> RiskPredictionResponse:
    context, action = investigate_risk(bundle, request)
    model_output = context.model_output
    return RiskPredictionResponse(
        fraud_probability=model_output.fraud_probability,
        risk_score=model_output.risk_score,
        risk_level=model_output.risk_level,
        fraud_prediction=model_output.fraud_prediction,
        classification_threshold=model_output.classification_threshold,
        operating_mode=model_output.operating_mode,
        model_version=model_output.model_version,
        evidence=context.evidence,
        recommended_action=action,
    )
