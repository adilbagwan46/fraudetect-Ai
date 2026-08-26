import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from backend.app.core.config import get_settings
from backend.app.main import app
from ml.fraudetect_ml.data.contracts import ML_FEATURE_COLUMNS
from tests.helpers import reference_profile


def create_test_bundle(root: Path) -> None:
    version = "test-v1"
    model_dir = root / version
    model_dir.mkdir(parents=True)
    features = pd.DataFrame(
        [
            ["PAYMENT", 10.0, 100.0, 2, 2.397895, 0.1],
            ["TRANSFER", 95.0, 100.0, 3, 4.564348, 0.95],
        ],
        columns=ML_FEATURE_COLUMNS,
    )
    model = DummyClassifier(strategy="constant", constant=1).fit(features, [0, 1])
    joblib.dump(model, model_dir / "model.joblib")
    payloads = {
        "metadata.json": {
            "model_version": version,
            "model_family": "test",
            "dataset": {"sha256": "abc"},
        },
        "threshold-policy.json": {
            "recommended_mode": "BALANCED",
            "policies_selected_on_validation": {
                "HIGH_PRECISION": {"threshold": 0.8},
                "BALANCED": {"threshold": 0.5},
                "HIGH_RECALL": {"threshold": 0.2},
            },
        },
        "validation-metrics.json": {},
        "test-metrics.json": {},
        "candidate-comparison.json": {},
        "operating-points.json": {},
        "reference-profile.json": reference_profile(),
    }
    for name, payload in payloads.items():
        (model_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (root / "latest.json").write_text(
        json.dumps({"model_version": version, "path": str(model_dir)}),
        encoding="utf-8",
    )


def test_risk_api_validates_scoring_time_inputs() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/risk/predict",
        json={
            "transaction_type": "TRANSFER",
            "amount": -1,
            "origin_balance_before": 100,
            "hour_of_day": 24,
        },
    )

    assert response.status_code == 422


def test_model_endpoints_load_bundle_and_derive_features(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "models"
    create_test_bundle(artifact_root)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    get_settings.cache_clear()
    try:
        client = TestClient(app)
        status = client.get("/api/v1/model/status")
        prediction = client.post(
            "/api/v1/risk/predict",
            json={
                "transaction_type": "TRANSFER",
                "amount": 95,
                "origin_balance_before": 100,
                "hour_of_day": 3,
            },
        )
        evaluation = client.get("/api/v1/model/evaluation")

        assert status.status_code == 200
        assert status.json()["status"] == "ready"
        assert prediction.status_code == 200
        assert prediction.json()["recommendation_is_simulated"] is True
        assert len(prediction.json()["evidence"]) <= 5
        assert status.json()["reference_profile_version"] == "reference-test"
        investigation = client.post(
            "/api/v1/risk/investigate",
            json={
                "transaction_type": "TRANSFER",
                "amount": 95,
                "origin_balance_before": 100,
                "hour_of_day": 3,
            },
        )
        assert investigation.status_code == 200
        assert investigation.json()["reference_profile_version"] == "reference-test"
        assert set(evaluation.json()) == {
            "validation",
            "test",
            "candidate_comparison",
            "operating_points",
        }
    finally:
        get_settings.cache_clear()
