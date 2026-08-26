import math
from pathlib import Path

import pytest

from backend.app.schemas.risk import (
    DerivedFeatures,
    ModelOutputContext,
    RiskPredictionRequest,
)
from backend.app.services.evidence_service import generate_evidence
from backend.app.services.risk_service import derived_features
from ml.fraudetect_ml.modeling.reference_profile import validate_training_reference_path
from tests.helpers import reference_profile


def model_output(probability: float, prediction: bool) -> ModelOutputContext:
    return ModelOutputContext(
        fraud_probability=probability,
        risk_score=probability * 100,
        risk_level="HIGH" if prediction else "LOW",
        fraud_prediction=prediction,
        classification_threshold=0.4,
        operating_mode="BALANCED",
        model_version="test-v1",
    )


def test_evidence_is_deterministic_and_bounded() -> None:
    transaction = RiskPredictionRequest(
        transaction_type="TRANSFER",
        amount=99.9,
        origin_balance_before=100,
        hour_of_day=4,
    )
    derived = DerivedFeatures(log_amount=math.log1p(99.9), amount_to_origin_balance=0.999)

    first = generate_evidence(
        transaction=transaction,
        derived_features=derived,
        model_output=model_output(0.95, True),
        reference_profile=reference_profile(),
    )
    second = generate_evidence(
        transaction=transaction,
        derived_features=derived,
        model_output=model_output(0.95, True),
        reference_profile=reference_profile(),
    )

    assert first == second
    assert len(first) <= 5
    assert any(item.id == "model_risk_above_threshold" for item in first)


def test_zero_balance_evidence_has_no_nan_or_infinity() -> None:
    transaction = RiskPredictionRequest(
        transaction_type="PAYMENT",
        amount=10,
        origin_balance_before=0,
        hour_of_day=2,
    )
    evidence = generate_evidence(
        transaction=transaction,
        derived_features=DerivedFeatures(log_amount=math.log1p(10), amount_to_origin_balance=0),
        model_output=model_output(0.01, False),
        reference_profile=reference_profile(),
    )

    numeric_facts = [
        value
        for item in evidence
        for value in item.facts.values()
        if isinstance(value, (int, float))
    ]
    assert all(math.isfinite(value) for value in numeric_facts)


def test_near_zero_balance_ratio_is_finite_and_bounded() -> None:
    transaction = RiskPredictionRequest(
        transaction_type="PAYMENT",
        amount=1e300,
        origin_balance_before=1e-300,
        hour_of_day=2,
    )

    derived = derived_features(transaction)

    assert math.isfinite(derived.amount_to_origin_balance)
    assert derived.amount_to_origin_balance == 1_000_000_000_000.0


def test_low_risk_is_not_artificially_described_as_suspicious() -> None:
    transaction = RiskPredictionRequest(
        transaction_type="PAYMENT",
        amount=50,
        origin_balance_before=1_000,
        hour_of_day=12,
    )
    evidence = generate_evidence(
        transaction=transaction,
        derived_features=DerivedFeatures(log_amount=math.log1p(50), amount_to_origin_balance=0.05),
        model_output=model_output(0.01, False),
        reference_profile=reference_profile(),
    )

    assert evidence[0].id == "model_risk_below_threshold"
    assert all(item.severity not in {"CRITICAL", "HIGH"} for item in evidence)


def test_reference_profile_rejects_validation_and_test_sources(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    test = tmp_path / "test.csv"
    manifest = {"splits": {"train": {"path": str(train)}}}

    validate_training_reference_path(train, manifest=manifest)
    with pytest.raises(ValueError, match="training split"):
        validate_training_reference_path(validation, manifest=manifest)
    with pytest.raises(ValueError, match="training split"):
        validate_training_reference_path(test, manifest=manifest)
