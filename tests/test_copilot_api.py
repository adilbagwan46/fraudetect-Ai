from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api.routes.risk import get_copilot_service
from backend.app.core.config import get_settings
from backend.app.main import app
from backend.app.schemas.copilot import InvestigationReport, SanitizedInvestigationContext
from backend.app.services.copilot.service import CopilotService
from tests.test_risk_api import (
    create_test_bundle,
    create_test_history,
    create_test_relationship_history,
)


class ApiCapturingProvider:
    name = "deterministic_test_provider"
    model = "test-model"

    def __init__(self) -> None:
        self.payloads: list[SanitizedInvestigationContext] = []

    def generate(self, context: SanitizedInvestigationContext) -> InvestigationReport:
        self.payloads.append(context)
        return CopilotService(None).investigate(context).report


def test_manual_copilot_endpoint_uses_explicit_fallback_without_fabricated_history(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "models"
    create_test_bundle(artifact_root)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FRAUDETECT_LLM_ENABLED", "false")
    get_settings.cache_clear()
    try:
        response = TestClient(app).post(
            "/api/v1/risk/investigate/copilot",
            json={
                "transaction_type": "PAYMENT",
                "amount": 50,
                "origin_balance_before": 1_000,
                "hour_of_day": 12,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "deterministic_fallback"
        assert payload["provider"] == "deterministic_fallback"
        assert payload["ai_available"] is False
        assert payload["execution"] == {
            "generated_by": "deterministic_fallback",
            "provider_attempted": False,
            "provider_succeeded": False,
            "generation_latency_ms": None,
            "failure_category": "disabled",
        }
        assert payload["relationship_context"]["context_available"] is False
        assert "unavailable" in (
            payload["report"]["relationship_analysis"]["history_limitation"]
        ).lower()
        assert "No prior behavioral history" in (
            payload["report"]["behavioral_analysis"]["history_limitation"]
        )
        assert "not LLM-generated" in payload["report"]["disclaimer"]
    finally:
        get_settings.cache_clear()


def test_reference_copilot_sends_only_sanitized_payload_and_never_echoes_reference(
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
    provider = ApiCapturingProvider()
    app.dependency_overrides[get_copilot_service] = lambda: CopilotService(provider)
    get_settings.cache_clear()
    try:
        response = TestClient(app).post(
            "/api/v1/risk/investigate/copilot",
            json={"transaction_reference": "TX-000000002"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "real_llm"
        assert payload["provider"] == "deterministic_test_provider"
        assert payload["execution"]["generated_by"] == "real_provider"
        assert payload["execution"]["provider_succeeded"] is True
        assert len(provider.payloads) == 1
        provider_payload = provider.payloads[0].model_dump_json()
        for forbidden in (
            "TX-000000001",
            "TX-000000002",
            "C-secret",
            "M-secret",
            "transaction_reference",
            "origin_key",
            "destination_key",
            "raw_transaction_history",
            "DATA_CONTEXT",
            "You are Fraudetect AI",
        ):
            assert forbidden not in provider_payload
            assert forbidden not in response.text
        assert provider.payloads[0].behavioral_context.history_available is True
        assert provider.payloads[0].behavioral_context.prior_transaction_count == 1
        assert provider.payloads[0].relationship_context.prior_interaction_count == 1
    finally:
        app.dependency_overrides.pop(get_copilot_service, None)
        get_settings.cache_clear()


def test_api_provider_failure_returns_controlled_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    class FailedProvider:
        name = "failed_provider"
        model = "failed-model"

        def generate(self, context):
            raise TimeoutError("secret provider diagnostic")

    artifact_root = tmp_path / "models"
    create_test_bundle(artifact_root)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    app.dependency_overrides[get_copilot_service] = lambda: CopilotService(FailedProvider())
    get_settings.cache_clear()
    try:
        response = TestClient(app).post(
            "/api/v1/risk/investigate/copilot",
            json={
                "transaction_type": "TRANSFER",
                "amount": 95,
                "origin_balance_before": 100,
                "hour_of_day": 3,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "deterministic_fallback"
        assert payload["execution"]["failure_category"] == "provider_timeout"
        assert "secret provider diagnostic" not in response.text
    finally:
        app.dependency_overrides.pop(get_copilot_service, None)
        get_settings.cache_clear()


def test_copilot_endpoint_rejects_mixed_reference_and_manual_input() -> None:
    response = TestClient(app).post(
        "/api/v1/risk/investigate/copilot",
        json={
            "transaction_reference": "TX-000000002",
            "transaction_type": "TRANSFER",
            "amount": 95,
            "origin_balance_before": 100,
            "hour_of_day": 3,
        },
    )

    assert response.status_code == 422
