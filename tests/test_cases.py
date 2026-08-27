import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes.cases import get_case_copilot
from backend.app.core.config import get_settings
from backend.app.main import app
from backend.app.schemas.case import CaseUpdateRequest
from backend.app.schemas.copilot import InvestigationReport, SanitizedEvidence
from backend.app.services.case_service import SQLiteCaseRepository, assign_case_priority
from backend.app.services.copilot.service import CopilotService
from tests.test_copilot import sanitized_context
from tests.test_risk_api import (
    create_test_bundle,
    create_test_history,
    create_test_relationship_history,
)


@pytest.fixture
def case_client(tmp_path: Path, monkeypatch) -> tuple[TestClient, Path]:
    artifact_root = tmp_path / "models"
    behavior_database = tmp_path / "behavior" / "history.sqlite"
    relationship_database = tmp_path / "relationship" / "history.sqlite"
    case_database = tmp_path / "cases" / "cases.sqlite"
    create_test_bundle(artifact_root)
    create_test_history(behavior_database)
    create_test_relationship_history(relationship_database)
    monkeypatch.setenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FRAUDETECT_BEHAVIORAL_HISTORY_DB", str(behavior_database))
    monkeypatch.setenv(
        "FRAUDETECT_RELATIONSHIP_HISTORY_DB", str(relationship_database)
    )
    monkeypatch.setenv("FRAUDETECT_CASE_DATABASE", str(case_database))
    monkeypatch.setenv("FRAUDETECT_LLM_ENABLED", "false")
    get_settings.cache_clear()
    try:
        yield TestClient(app), case_database
    finally:
        app.dependency_overrides.pop(get_case_copilot, None)
        get_settings.cache_clear()


def create_reference_case(client: TestClient) -> dict:
    response = client.post(
        "/api/v1/cases",
        json={"transaction_reference": "TX-000000002"},
    )
    assert response.status_code == 201
    return response.json()


def test_create_retrieve_and_persist_identifier_free_case(
    case_client: tuple[TestClient, Path],
) -> None:
    client, database = case_client
    created = create_reference_case(client)
    case_id = created["case"]["case_id"]
    retrieved = TestClient(app).get(f"/api/v1/cases/{case_id}")

    assert retrieved.status_code == 200
    assert retrieved.json() == created
    assert created["snapshot_is_immutable"] is True
    assert created["case"]["source_type"] == "REFERENCE"
    assert created["case"]["transaction_reference_available"] is True
    assert created["case"]["analyst_decision"]["is_model_ground_truth"] is False
    assert database.is_file()
    serialized = retrieved.text
    for forbidden in (
        "TX-000000001",
        "TX-000000002",
        "C-secret",
        "M-secret",
        '"transaction_reference":',
        "origin_key",
        "destination_key",
        "raw_transaction_history",
        "raw_relationship_history",
        "is_fraud",
        "is_flagged_fraud",
    ):
        assert forbidden not in serialized


def test_case_database_stores_no_source_reference_or_raw_history(
    case_client: tuple[TestClient, Path],
) -> None:
    client, database = case_client
    create_reference_case(client)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cases)")
        }
        stored = " ".join(
            str(value)
            for row in connection.execute("SELECT * FROM cases")
            for value in row
            if value is not None
        )

    assert "transaction_reference" not in columns
    assert "origin_key" not in columns
    assert "destination_key" not in columns
    assert "TX-000000002" not in stored
    assert "C-secret" not in stored
    assert "M-secret" not in stored


def test_case_queue_lists_filters_and_paginates(
    case_client: tuple[TestClient, Path],
) -> None:
    client, _ = case_client
    first = create_reference_case(client)
    second = create_reference_case(client)
    client.patch(
        f"/api/v1/cases/{second['case']['case_id']}",
        json={"status": "IN_REVIEW"},
    )

    all_cases = client.get("/api/v1/cases?limit=1&offset=0")
    open_cases = client.get("/api/v1/cases?status=OPEN")
    reviewed_cases = client.get("/api/v1/cases?status=IN_REVIEW")
    priority = first["case"]["priority"]
    priority_cases = client.get(f"/api/v1/cases?priority={priority}")

    assert all_cases.status_code == 200
    assert all_cases.json()["total"] == 2
    assert len(all_cases.json()["items"]) == 1
    assert open_cases.json()["total"] == 1
    assert reviewed_cases.json()["total"] == 1
    assert priority_cases.json()["total"] == 2


def test_valid_lifecycle_note_and_immutable_intelligence_snapshot(
    case_client: tuple[TestClient, Path],
) -> None:
    client, _ = case_client
    created = create_reference_case(client)
    case_id = created["case"]["case_id"]
    original_snapshot = created["intelligence_snapshot"]
    original_probability = created["case"]["fraud_probability"]

    reviewed = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"status": "IN_REVIEW", "analyst_note": "Verified by demo analyst."},
    )
    escalated = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"status": "ESCALATED", "analyst_note": "Needs secondary review."},
    )
    closed = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"status": "CLOSED"},
    )

    assert reviewed.status_code == 200
    assert escalated.status_code == 200
    assert closed.status_code == 200
    payload = closed.json()
    assert payload["case"]["status"] == "CLOSED"
    assert payload["case"]["analyst_decision"]["disposition"] == "ESCALATED"
    assert payload["case"]["analyst_decision"]["is_model_ground_truth"] is False
    assert payload["intelligence_snapshot"] == original_snapshot
    assert payload["case"]["fraud_probability"] == original_probability
    assert [item["new_status"] for item in payload["status_history"]] == [
        "OPEN",
        "IN_REVIEW",
        "ESCALATED",
        "CLOSED",
    ]


def test_invalid_transition_and_closed_case_update_are_rejected(
    case_client: tuple[TestClient, Path],
) -> None:
    client, _ = case_client
    case_id = create_reference_case(client)["case"]["case_id"]

    invalid = client.patch(
        f"/api/v1/cases/{case_id}", json={"status": "CLEARED"}
    )
    assert invalid.status_code == 409
    assert "OPEN -> CLEARED" in invalid.json()["detail"]

    client.patch(f"/api/v1/cases/{case_id}", json={"status": "IN_REVIEW"})
    client.patch(f"/api/v1/cases/{case_id}", json={"status": "CLEARED"})
    client.patch(f"/api/v1/cases/{case_id}", json={"status": "CLOSED"})
    closed_update = client.patch(
        f"/api/v1/cases/{case_id}", json={"analyst_note": "Too late"}
    )
    assert closed_update.status_code == 409


def test_closed_case_rejects_copilot_without_invoking_provider_or_mutating_case(
    case_client: tuple[TestClient, Path],
) -> None:
    client, _ = case_client
    case_id = create_reference_case(client)["case"]["case_id"]
    client.patch(f"/api/v1/cases/{case_id}", json={"status": "IN_REVIEW"})
    client.patch(f"/api/v1/cases/{case_id}", json={"status": "CLEARED"})
    client.patch(f"/api/v1/cases/{case_id}", json={"status": "CLOSED"})
    closed = client.get(f"/api/v1/cases/{case_id}").json()
    provider_calls = []

    class ProviderThatMustNotRun:
        name = "must-not-run"
        model = "must-not-run"

        def generate(self, context) -> InvestigationReport:
            provider_calls.append(context)
            raise AssertionError("Closed case reached the Copilot provider")

    app.dependency_overrides[get_case_copilot] = lambda: CopilotService(
        ProviderThatMustNotRun()
    )
    response = client.post(f"/api/v1/cases/{case_id}/copilot")
    after = client.get(f"/api/v1/cases/{case_id}").json()

    assert response.status_code == 409
    assert response.json()["detail"] == "Closed cases are immutable"
    assert provider_calls == []
    assert after == closed


def test_case_copilot_fallback_is_stored_without_changing_workflow_or_model(
    case_client: tuple[TestClient, Path],
) -> None:
    client, _ = case_client
    created = create_reference_case(client)
    case_id = created["case"]["case_id"]

    generated = client.post(f"/api/v1/cases/{case_id}/copilot")
    detail = client.get(f"/api/v1/cases/{case_id}").json()

    assert generated.status_code == 200
    assert generated.json()["mode"] == "deterministic_fallback"
    assert generated.json()["ai_available"] is False
    assert detail["copilot"] == generated.json()
    assert detail["case"]["status"] == "OPEN"
    assert detail["case"]["fraud_probability"] == created["case"]["fraud_probability"]
    assert detail["intelligence_snapshot"] == created["intelligence_snapshot"]
    assert [item["event"] for item in detail["decision_trace"]] == [
        "CASE_CREATED",
        "INTELLIGENCE_CAPTURED",
        "COPILOT_GENERATED",
    ]
    assert detail["decision_trace"][0]["occurred_at"] == created["case"]["created_at"]
    assert detail["decision_trace"][1]["occurred_at"] == created["case"]["created_at"]


def test_decision_trace_contains_only_recorded_case_events(
    case_client: tuple[TestClient, Path],
) -> None:
    client, _ = case_client
    created = create_reference_case(client)
    case_id = created["case"]["case_id"]

    assert [item["event"] for item in created["decision_trace"]] == [
        "CASE_CREATED",
        "INTELLIGENCE_CAPTURED",
    ]
    reviewed = client.patch(
        f"/api/v1/cases/{case_id}",
        json={"status": "IN_REVIEW", "analyst_note": "Reviewed."},
    ).json()
    cleared = client.patch(
        f"/api/v1/cases/{case_id}", json={"status": "CLEARED"}
    ).json()

    assert [item["event"] for item in cleared["decision_trace"]] == [
        "CASE_CREATED",
        "INTELLIGENCE_CAPTURED",
        "NOTE_ADDED",
        "ANALYST_REVIEWED",
        "CASE_CLEARED",
    ]
    assert "COPILOT_GENERATED" not in {
        item["event"] for item in reviewed["decision_trace"]
    }
    assert all(item["occurred_at"] for item in cleared["decision_trace"])


def test_case_copilot_provider_receives_only_approved_snapshot_and_failure_falls_back(
    case_client: tuple[TestClient, Path],
) -> None:
    client, _ = case_client
    case_id = create_reference_case(client)["case"]["case_id"]
    captured = []

    class FailingCapturingProvider:
        name = "case-test-provider"
        model = "case-test-model"

        def generate(self, context) -> InvestigationReport:
            captured.append(context)
            raise TimeoutError("private provider detail")

    app.dependency_overrides[get_case_copilot] = lambda: CopilotService(
        FailingCapturingProvider()
    )
    response = client.post(f"/api/v1/cases/{case_id}/copilot")

    assert response.status_code == 200
    assert response.json()["mode"] == "deterministic_fallback"
    assert len(captured) == 1
    payload = captured[0].model_dump_json()
    assert "transaction_reference" not in payload
    assert "origin_key" not in payload
    assert "destination_key" not in payload
    assert "analyst_note" not in payload
    assert "private provider detail" not in response.text


def test_priority_policy_is_deterministic_and_does_not_modify_model_risk() -> None:
    high = sanitized_context(probability=0.99, prediction=True)
    low = sanitized_context(probability=0.01, prediction=False)
    low.evidence = []
    strong = low.model_copy(deep=True)
    strong.evidence = [
        SanitizedEvidence(
            evidence_id="behavior_amount_above_typical",
            title="Amount is above prior typical behavior",
            severity="MEDIUM",
            category="BEHAVIORAL_CONTEXT",
            facts={"amount_vs_prior_average": 8.0},
        )
    ]
    original_low_model = low.model_output.model_copy(deep=True)

    assert assign_case_priority(high) == "CRITICAL"
    assert assign_case_priority(low) == "LOW"
    assert assign_case_priority(strong) == "HIGH"
    assert low.model_output == original_low_model
    assert high.model_output.risk_level == "HIGH"
    assert high.model_output.risk_level != "CRITICAL"


def test_case_input_rejects_internal_fields_and_unknown_case_returns_404(
    case_client: tuple[TestClient, Path],
) -> None:
    client, _ = case_client
    invalid = client.post(
        "/api/v1/cases",
        json={
            "transaction_type": "PAYMENT",
            "amount": 10,
            "origin_balance_before": 100,
            "hour_of_day": 1,
            "origin_key": "C-forbidden",
        },
    )
    missing = client.get("/api/v1/cases/CASE-0000000000000000")

    assert invalid.status_code == 422
    assert missing.status_code == 404


def _snapshot_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload["intelligence_snapshot"], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def test_audit_events_are_append_only_ordered_and_snapshot_safe(
    case_client: tuple[TestClient, Path],
) -> None:
    client, database = case_client
    created = create_reference_case(client)
    case_id = created["case"]["case_id"]
    original_hash = _snapshot_sha256(created)
    original_events = created["decision_trace"]

    note_only = client.patch(
        f"/api/v1/cases/{case_id}", json={"analyst_note": "Checked safe facts."}
    ).json()
    reviewed = client.patch(
        f"/api/v1/cases/{case_id}", json={"status": "IN_REVIEW"}
    ).json()
    copilot = client.post(f"/api/v1/cases/{case_id}/copilot")
    final = client.get(f"/api/v1/cases/{case_id}").json()

    assert copilot.status_code == 200
    assert final["decision_trace"][:2] == original_events
    assert [item["event"] for item in final["decision_trace"]] == [
        "CASE_CREATED",
        "INTELLIGENCE_CAPTURED",
        "NOTE_ADDED",
        "ANALYST_REVIEWED",
        "COPILOT_GENERATED",
    ]
    assert note_only["decision_trace"][2]["note_recorded"] is True
    assert reviewed["decision_trace"][3]["previous_status"] == "OPEN"
    assert reviewed["decision_trace"][3]["new_status"] == "IN_REVIEW"
    assert final["decision_trace"][-1]["copilot_mode"] == "deterministic_fallback"
    assert [item["occurred_at"] for item in final["decision_trace"]] == sorted(
        item["occurred_at"] for item in final["decision_trace"]
    )
    assert _snapshot_sha256(final) == original_hash

    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE case_audit_events SET actor = 'SYSTEM' WHERE case_id = ?",
                (case_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM case_audit_events WHERE case_id = ?", (case_id,)
            )


def test_rejected_operations_create_no_false_audit_events(
    case_client: tuple[TestClient, Path],
) -> None:
    client, database = case_client
    case_id = create_reference_case(client)["case"]["case_id"]

    def audit_count() -> int:
        with sqlite3.connect(database) as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM case_audit_events WHERE case_id = ?", (case_id,)
                ).fetchone()[0]
            )

    initial = audit_count()
    assert client.patch(f"/api/v1/cases/{case_id}", json={"status": "CLEARED"}).status_code == 409
    assert audit_count() == initial

    client.patch(f"/api/v1/cases/{case_id}", json={"status": "IN_REVIEW"})
    client.patch(f"/api/v1/cases/{case_id}", json={"status": "CLEARED"})
    client.patch(f"/api/v1/cases/{case_id}", json={"status": "CLOSED"})
    closed_count = audit_count()
    assert client.patch(
        f"/api/v1/cases/{case_id}", json={"analyst_note": "Rejected"}
    ).status_code == 409
    assert client.post(f"/api/v1/cases/{case_id}/copilot").status_code == 409
    assert audit_count() == closed_count


def test_phase7_database_migration_adds_empty_audit_log_without_backfill(
    case_client: tuple[TestClient, Path],
) -> None:
    client, database = case_client
    created = create_reference_case(client)
    case_id = created["case"]["case_id"]
    client.post(f"/api/v1/cases/{case_id}/copilot")
    before = client.get(f"/api/v1/cases/{case_id}").json()
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER case_audit_events_no_update")
        connection.execute("DROP TRIGGER case_audit_events_no_delete")
        connection.execute("DROP TABLE case_audit_events")

    repository = SQLiteCaseRepository(database)
    migrated = repository.get(case_id).model_dump(mode="json")

    assert migrated["intelligence_snapshot"] == before["intelligence_snapshot"]
    assert migrated["status_history"] == before["status_history"]
    assert [item["event"] for item in migrated["decision_trace"]] == [
        "CASE_CREATED",
        "INTELLIGENCE_CAPTURED",
        "COPILOT_GENERATED",
    ]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM case_audit_events").fetchone()[0] == 0

    updated = repository.update(
        case_id,
        CaseUpdateRequest(status="IN_REVIEW", analyst_note="Post-migration review."),
    ).model_dump(mode="json")
    assert [item["event"] for item in updated["decision_trace"]] == [
        "CASE_CREATED",
        "INTELLIGENCE_CAPTURED",
        "COPILOT_GENERATED",
        "NOTE_ADDED",
        "ANALYST_REVIEWED",
    ]
    assert updated["intelligence_snapshot"] == before["intelligence_snapshot"]


def test_operational_metrics_are_correct_aggregate_only_and_non_mutating(
    case_client: tuple[TestClient, Path],
) -> None:
    client, database = case_client
    cases = [create_reference_case(client) for _ in range(4)]
    second, third, fourth = [item["case"]["case_id"] for item in cases[1:]]
    client.patch(f"/api/v1/cases/{second}", json={"status": "IN_REVIEW"})
    client.patch(f"/api/v1/cases/{second}", json={"status": "ESCALATED"})
    client.patch(f"/api/v1/cases/{third}", json={"status": "IN_REVIEW"})
    client.patch(f"/api/v1/cases/{third}", json={"status": "CLEARED"})
    client.patch(f"/api/v1/cases/{fourth}", json={"status": "IN_REVIEW"})
    client.patch(f"/api/v1/cases/{fourth}", json={"status": "CLEARED"})
    client.patch(f"/api/v1/cases/{fourth}", json={"status": "CLOSED"})
    client.post(f"/api/v1/cases/{second}/copilot")
    before = database.read_bytes()

    response = client.get("/api/v1/system/metrics")
    after = database.read_bytes()

    assert response.status_code == 200
    metrics = response.json()
    assert metrics["total_cases"] == 4
    assert metrics["active_cases"] == 3
    assert metrics["status_counts"] == {
        "open": 1,
        "in_review": 0,
        "escalated": 1,
        "cleared": 1,
        "closed": 1,
    }
    assert sum(metrics["priority_counts"].values()) == 4
    assert metrics["copilot_reports"] == {
        "total": 1,
        "deterministic_fallback": 1,
        "real_llm": 0,
    }
    assert metrics["case_priority_is_distinct_from_ml_risk"] is True
    assert set(metrics) == {
        "total_cases",
        "active_cases",
        "status_counts",
        "priority_counts",
        "disposition_counts",
        "copilot_reports",
        "case_priority_is_distinct_from_ml_risk",
    }
    serialized = response.text
    assert cases[0]["case"]["case_id"] not in serialized
    assert "TX-000000002" not in serialized
    assert "C-secret" not in serialized
    assert "M-secret" not in serialized
    assert before == after


def test_operational_metrics_empty_database(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "empty" / "cases.sqlite"
    monkeypatch.setenv("FRAUDETECT_CASE_DATABASE", str(database))
    get_settings.cache_clear()
    try:
        response = TestClient(app).get("/api/v1/system/metrics")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_cases"] == 0
    assert payload["active_cases"] == 0
    assert all(value == 0 for value in payload["status_counts"].values())
    assert all(value == 0 for value in payload["priority_counts"].values())
