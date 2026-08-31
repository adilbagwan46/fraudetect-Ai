import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from backend.app.core.config import get_settings
from backend.app.main import app
from backend.app.services import risk_service
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


def create_test_history(database: Path) -> None:
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE transactions (
                transaction_reference TEXT PRIMARY KEY,
                step INTEGER NOT NULL,
                transaction_type TEXT NOT NULL,
                amount REAL NOT NULL,
                origin_balance_before REAL NOT NULL,
                origin_key TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.executemany(
            "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("TX-000000001", 1, "PAYMENT", 10.0, 100.0, "C-secret"),
                ("TX-000000002", 2, "TRANSFER", 95.0, 100.0, "C-secret"),
                ("TX-000000003", 2, "CASH_OUT", 900.0, 100.0, "C-secret"),
            ],
        )
        connection.execute(
            "CREATE INDEX transactions_origin_step_idx "
            "ON transactions (origin_key, step, transaction_reference)"
        )


def create_test_relationship_history(database: Path) -> None:
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE relationship_transactions (
                transaction_reference TEXT PRIMARY KEY,
                step INTEGER NOT NULL,
                amount REAL NOT NULL,
                origin_key TEXT NOT NULL,
                destination_key TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.executemany(
            "INSERT INTO relationship_transactions VALUES (?, ?, ?, ?, ?)",
            [
                ("TX-000000001", 1, 10.0, "C-secret", "M-secret"),
                ("TX-000000002", 2, 95.0, "C-secret", "M-secret"),
                ("TX-000000003", 2, 900.0, "C-secret", "M-other-secret"),
            ],
        )
        connection.execute(
            "CREATE INDEX relationship_pair_step_idx ON relationship_transactions "
            "(origin_key, destination_key, step, transaction_reference)"
        )
        connection.execute(
            "CREATE INDEX relationship_origin_step_idx ON relationship_transactions "
            "(origin_key, step, destination_key)"
        )
        connection.execute(
            "CREATE INDEX relationship_destination_step_idx ON relationship_transactions "
            "(destination_key, step, origin_key)"
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
    actual_joblib_load = risk_service.joblib.load
    model_loads = 0

    def counted_joblib_load(path: Path):
        nonlocal model_loads
        model_loads += 1
        return actual_joblib_load(path)

    monkeypatch.setattr(risk_service.joblib, "load", counted_joblib_load)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    get_settings.cache_clear()
    risk_service.load_active_bundle.cache_clear()
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
        assert "behavioral_context" not in prediction.json()
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
        assert investigation.json()["behavioral_context"]["history_available"] is False
        assert investigation.json()["relationship_context"]["context_available"] is False
        assert investigation.json()["relationship_evidence"] == []
        assert set(evaluation.json()) == {
            "validation",
            "test",
            "candidate_comparison",
            "operating_points",
        }
        assert model_loads == 1
    finally:
        risk_service.load_active_bundle.cache_clear()
        get_settings.cache_clear()


def test_investigation_reference_returns_aggregate_history_without_identifiers(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "models"
    history_database = tmp_path / "behavior" / "history.sqlite"
    relationship_database = tmp_path / "relationship" / "history.sqlite"
    create_test_bundle(artifact_root)
    create_test_history(history_database)
    create_test_relationship_history(relationship_database)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FRAUDETECT_BEHAVIORAL_HISTORY_DB", str(history_database))
    monkeypatch.setenv(
        "FRAUDETECT_RELATIONSHIP_HISTORY_DB", str(relationship_database)
    )
    get_settings.cache_clear()
    try:
        response = TestClient(app).post(
            "/api/v1/risk/investigate",
            json={"transaction_reference": "TX-000000002"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["behavioral_context"]["history_available"] is True
        assert payload["behavioral_context"]["prior_transaction_count"] == 1
        assert payload["behavioral_context"]["prior_total_amount"] == 10
        assert payload["relationship_context"]["relationship_seen_before"] is True
        assert payload["relationship_context"]["prior_interaction_count"] == 1
        serialized = response.text
        assert "C-secret" not in serialized
        assert "TX-000000001" not in serialized
        assert "TX-000000002" not in serialized
        assert "origin_key" not in serialized
        assert "M-secret" not in serialized
        assert "destination_key" not in serialized
    finally:
        get_settings.cache_clear()


def test_investigation_invalid_reference_is_safely_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "models"
    history_database = tmp_path / "behavior" / "history.sqlite"
    create_test_bundle(artifact_root)
    create_test_history(history_database)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FRAUDETECT_BEHAVIORAL_HISTORY_DB", str(history_database))
    monkeypatch.setenv("FRAUDETECT_CASE_DATABASE", str(tmp_path / "cases.sqlite"))
    get_settings.cache_clear()
    try:
        client = TestClient(app)
        responses = [
            client.post(
                endpoint,
                json={"transaction_reference": "TX-999999999"},
            )
            for endpoint in (
                "/api/v1/risk/investigate",
                "/api/v1/risk/investigate/copilot",
                "/api/v1/cases",
            )
        ]

        for response in responses:
            assert response.status_code == 404
            assert response.json()["detail"] == "Transaction reference was not found"
            assert "TX-999999999" not in response.text
    finally:
        get_settings.cache_clear()


def test_investigation_reports_unavailable_history_index(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "models"
    create_test_bundle(artifact_root)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv(
        "FRAUDETECT_BEHAVIORAL_HISTORY_DB",
        str(tmp_path / "missing-history.sqlite"),
    )
    get_settings.cache_clear()
    try:
        response = TestClient(app).post(
            "/api/v1/risk/investigate",
            json={"transaction_reference": "TX-000000002"},
        )

        assert response.status_code == 503
        assert "build-behavior-history" in response.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_investigation_reports_unavailable_relationship_index(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "models"
    history_database = tmp_path / "behavior" / "history.sqlite"
    create_test_bundle(artifact_root)
    create_test_history(history_database)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FRAUDETECT_BEHAVIORAL_HISTORY_DB", str(history_database))
    monkeypatch.setenv(
        "FRAUDETECT_RELATIONSHIP_HISTORY_DB",
        str(tmp_path / "missing-relationship.sqlite"),
    )
    get_settings.cache_clear()
    try:
        response = TestClient(app).post(
            "/api/v1/risk/investigate",
            json={"transaction_reference": "TX-000000002"},
        )

        assert response.status_code == 503
        assert "build-relationship-history" in response.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_investigation_reference_rejects_mixed_manual_input() -> None:
    response = TestClient(app).post(
        "/api/v1/risk/investigate",
        json={
            "transaction_reference": "TX-000000002",
            "amount": 95,
            "transaction_type": "TRANSFER",
            "origin_balance_before": 100,
            "hour_of_day": 2,
        },
    )

    assert response.status_code == 422
