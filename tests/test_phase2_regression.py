import hashlib
from pathlib import Path

import pytest

from backend.app.schemas.risk import RiskPredictionRequest
from backend.app.services.risk_service import load_active_bundle, predict_risk

FROZEN_MODEL_SHA256 = "9664e4f43e48dcf86f0dc4e2293092a55af97c92d9f9b0b3ff93cd885ac99e92"


def test_frozen_phase2_prediction_and_evidence_regression() -> None:
    artifact_root = Path("artifacts/models")
    latest = artifact_root / "latest.json"
    if not latest.is_file():
        pytest.skip("Frozen local Phase 2A artifact is intentionally not stored in Git")

    bundle = load_active_bundle(artifact_root)
    model_sha256 = hashlib.sha256((bundle.model_dir / "model.joblib").read_bytes()).hexdigest()
    low = predict_risk(
        bundle,
        RiskPredictionRequest(
            transaction_type="PAYMENT",
            amount=50,
            origin_balance_before=1_000,
            hour_of_day=12,
        ),
    )
    high = predict_risk(
        bundle,
        RiskPredictionRequest(
            transaction_type="TRANSFER",
            amount=250_000,
            origin_balance_before=250_000,
            hour_of_day=3,
        ),
    )

    assert model_sha256 == FROZEN_MODEL_SHA256
    assert low.fraud_probability == pytest.approx(1.1030544844301375e-10)
    assert low.classification_threshold == pytest.approx(0.4002576812593272)
    assert low.fraud_prediction is False
    assert [item.id for item in low.evidence] == [
        "model_risk_below_threshold",
        "origin_balance_ratio_context",
        "hour_training_context",
        "transaction_type_training_context",
        "amount_reference_context",
    ]
    assert high.fraud_probability == pytest.approx(0.9381233582863879)
    assert high.classification_threshold == pytest.approx(0.4002576812593272)
    assert high.fraud_prediction is True
    assert [item.id for item in high.evidence] == [
        "model_risk_above_threshold",
        "origin_balance_ratio_context",
        "transaction_type_training_context",
        "hour_training_context",
        "amount_reference_context",
    ]
