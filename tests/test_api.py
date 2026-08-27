from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import app
from backend.app.services.case_service import SQLiteCaseRepository
from tests.test_risk_api import (
    create_test_bundle,
    create_test_history,
    create_test_relationship_history,
)


def test_health_endpoint() -> None:
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dataset_status_handles_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FRAUDETECT_DATASET_MANIFEST", str(tmp_path / "missing.json"))
    get_settings.cache_clear()
    try:
        response = TestClient(app).get("/api/v1/dataset/status")
        assert response.status_code == 200
        assert response.json()["status"] == "not_prepared"
        assert str(tmp_path) not in response.text
        assert "manifest_path" not in response.json()
    finally:
        get_settings.cache_clear()


def test_validation_errors_do_not_echo_submitted_transaction_references(
    tmp_path: Path, monkeypatch
) -> None:
    private_reference = "PRIVATE-REFERENCE-DO-NOT-ECHO"
    monkeypatch.setenv("FRAUDETECT_CASE_DATABASE", str(tmp_path / "cases.sqlite"))
    get_settings.cache_clear()
    try:
        client = TestClient(app)
        responses = [
            client.post(
                "/api/v1/risk/investigate",
                json={"transaction_reference": f"{private_reference}!"},
            ),
            client.post(
                "/api/v1/risk/investigate/copilot",
                json={
                    "transaction_reference": private_reference,
                    "transaction_type": "TRANSFER",
                    "amount": 1,
                    "origin_balance_before": 2,
                    "hour_of_day": 3,
                },
            ),
            client.post(
                "/api/v1/cases",
                json={"transaction_reference": f"{private_reference}!"},
            ),
        ]
    finally:
        get_settings.cache_clear()

    for response in responses:
        assert response.status_code == 422
        assert private_reference not in response.text
        assert all("input" not in item for item in response.json()["detail"])
        assert all("ctx" not in item for item in response.json()["detail"])


def test_system_readiness_reports_safe_component_state(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "models"
    behavior_database = tmp_path / "behavior" / "history.sqlite"
    relationship_database = tmp_path / "relationship" / "history.sqlite"
    case_database = tmp_path / "cases" / "cases.sqlite"
    create_test_bundle(artifact_root)
    create_test_history(behavior_database)
    create_test_relationship_history(relationship_database)
    SQLiteCaseRepository(case_database)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FRAUDETECT_BEHAVIORAL_HISTORY_DB", str(behavior_database))
    monkeypatch.setenv("FRAUDETECT_RELATIONSHIP_HISTORY_DB", str(relationship_database))
    monkeypatch.setenv("FRAUDETECT_CASE_DATABASE", str(case_database))
    monkeypatch.setenv("FRAUDETECT_LLM_ENABLED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-expose-this-secret")
    get_settings.cache_clear()
    try:
        response = TestClient(app).get("/api/v1/system/readiness")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert {item["key"] for item in payload["components"]} == {
        "ml_model",
        "deterministic_evidence",
        "reference_profile",
        "behavioral_history",
        "relationship_history",
        "case_store",
        "llm_copilot",
    }
    copilot = next(item for item in payload["components"] if item["key"] == "llm_copilot")
    assert copilot["mode"] == "deterministic_fallback"
    assert copilot["fallback_available"] is True
    assert copilot["provider_enabled"] is False
    assert copilot["provider_configured"] is True
    assert copilot["external_availability"] == "not_applicable"
    serialized = response.text
    assert "do-not-expose-this-secret" not in serialized
    assert str(tmp_path) not in serialized
    assert "OPENAI_API_KEY" not in serialized


def test_system_readiness_degrades_when_optional_artifacts_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("FRAUDETECT_BEHAVIORAL_HISTORY_DB", str(tmp_path / "behavior.sqlite"))
    monkeypatch.setenv(
        "FRAUDETECT_RELATIONSHIP_HISTORY_DB", str(tmp_path / "relationship.sqlite")
    )
    monkeypatch.setenv("FRAUDETECT_CASE_DATABASE", str(tmp_path / "cases.sqlite"))
    get_settings.cache_clear()
    try:
        response = TestClient(app).get("/api/v1/system/readiness")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert all(
        item["status"] == "unavailable"
        for item in payload["components"]
        if item["key"] != "llm_copilot"
    )
    assert not (tmp_path / "cases.sqlite").exists()


def test_system_readiness_reports_configuration_without_external_provider_call(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("FRAUDETECT_LLM_ENABLED", "true")
    monkeypatch.setenv("FRAUDETECT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-never-called-secret")
    get_settings.cache_clear()
    try:
        response = TestClient(app).get("/api/v1/system/readiness")
    finally:
        get_settings.cache_clear()

    copilot = next(
        item for item in response.json()["components"] if item["key"] == "llm_copilot"
    )
    assert copilot["status"] == "ready"
    assert copilot["mode"] == "real_llm_configured"
    assert copilot["provider_enabled"] is True
    assert copilot["provider_configured"] is True
    assert copilot["external_availability"] == "not_checked"
    assert "configured-but-never-called-secret" not in response.text
