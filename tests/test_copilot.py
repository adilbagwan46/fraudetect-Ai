import math
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.schemas.copilot import InvestigationReport, SanitizedInvestigationContext
from backend.app.schemas.risk import (
    BehavioralContext,
    DerivedFeatures,
    EvidenceItem,
    ModelOutputContext,
    RelationshipContext,
    RiskPredictionRequest,
)
from backend.app.services.behavioral_service import (
    HistoricalTransaction,
    build_behavioral_context,
    unavailable_behavioral_context,
)
from backend.app.services.copilot.context_builder import build_sanitized_context
from backend.app.services.copilot.prompts import SYSTEM_PROMPT
from backend.app.services.copilot.provider import OpenAIInvestigationProvider
from backend.app.services.copilot.service import CopilotService, create_copilot_service
from backend.app.services.evidence_service import build_investigation_context
from backend.app.services.relationship_service import (
    RelationshipTransaction,
    build_relationship_context,
)
from tests.helpers import reference_profile


def historical(
    reference: str,
    step: int,
    amount: float,
    *,
    transaction_type: str = "TRANSFER",
) -> HistoricalTransaction:
    return HistoricalTransaction(
        transaction_reference=reference,
        step=step,
        transaction_type=transaction_type,
        amount=amount,
        origin_balance_before=250_000,
        origin_key="C-forbidden-origin",
    )


def investigation_context(
    *,
    probability: float = 0.95,
    prediction: bool = True,
    behavioral: BehavioralContext | None = None,
    relationship: RelationshipContext | None = None,
):
    transaction = RiskPredictionRequest(
        transaction_type="TRANSFER" if prediction else "PAYMENT",
        amount=250_000 if prediction else 50,
        origin_balance_before=250_000 if prediction else 1_000,
        hour_of_day=3 if prediction else 12,
    )
    derived = DerivedFeatures(
        log_amount=math.log1p(transaction.amount),
        amount_to_origin_balance=transaction.amount / transaction.origin_balance_before,
    )
    model = ModelOutputContext(
        fraud_probability=probability,
        risk_score=probability * 100,
        risk_level="HIGH" if prediction else "LOW",
        fraud_prediction=prediction,
        classification_threshold=0.4,
        operating_mode="BALANCED",
        model_version="frozen-test",
    )
    return build_investigation_context(
        transaction=transaction,
        derived_features=derived,
        model_output=model,
        reference_profile=reference_profile(),
        behavioral_context=behavioral,
        relationship_context=relationship,
    )


def sanitized_context(
    *,
    probability: float = 0.95,
    prediction: bool = True,
    behavioral: BehavioralContext | None = None,
    relationship: RelationshipContext | None = None,
) -> SanitizedInvestigationContext:
    context = investigation_context(
        probability=probability,
        prediction=prediction,
        behavioral=behavioral,
        relationship=relationship,
    )
    action = "HOLD_FOR_INVESTIGATION" if prediction else "NORMAL_PROCESSING"
    return build_sanitized_context(context, action)


class CapturingProvider:
    name = "test_provider"
    model = "test-structured-model"

    def __init__(self) -> None:
        self.contexts: list[SanitizedInvestigationContext] = []

    def generate(self, context: SanitizedInvestigationContext) -> InvestigationReport:
        self.contexts.append(context)
        return CopilotService(None).investigate(context).report


def test_sanitizer_positively_selects_fields_and_blocks_identifiers_and_history() -> None:
    context = investigation_context(behavioral=unavailable_behavioral_context())
    context.evidence[0] = EvidenceItem(
        id=context.evidence[0].id,
        category=context.evidence[0].category,
        severity=context.evidence[0].severity,
        title=context.evidence[0].title,
        description=context.evidence[0].description,
        facts={
            **context.evidence[0].facts,
            "transaction_reference": "TX-999999999",
            "origin_key": "C123456789",
            "nameDest": "M123456789",
            "raw_transaction_history": [{"amount": 1}],
        },
    )
    context.approved_reference_statistics["raw_transaction_history"] = [
        {"transaction_reference": "TX-111111111"}
    ]

    sanitized = build_sanitized_context(context, "HOLD_FOR_INVESTIGATION")
    payload = sanitized.model_dump_json()

    assert set(sanitized.model_dump()) == {
        "transaction",
        "model_output",
        "evidence",
        "reference_context",
        "behavioral_context",
        "relationship_context",
    }
    for forbidden in (
        "TX-999999999",
        "TX-111111111",
        "C123456789",
        "M123456789",
        "transaction_reference",
        "origin_key",
        "nameDest",
        "raw_transaction_history",
        "model_version",
        "derived_features",
    ):
        assert forbidden not in payload


def test_provider_receives_only_sanitized_context() -> None:
    provider = CapturingProvider()
    context = sanitized_context(behavioral=unavailable_behavioral_context())

    first = CopilotService(provider).investigate(context)
    second = CopilotService(provider).investigate(context)

    assert first == second
    assert first.mode == "real_llm"
    assert len(provider.contexts) == 2
    assert provider.contexts[0] == context
    assert "transaction_reference" not in provider.contexts[0].model_dump_json()


def test_openai_provider_uses_separate_system_and_inert_data_messages() -> None:
    captured: dict[str, Any] = {}
    expected = CopilotService(None).investigate(
        sanitized_context(behavioral=unavailable_behavioral_context())
    ).report

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=expected)

    fake_client = SimpleNamespace(responses=FakeResponses())
    provider = OpenAIInvestigationProvider(
        api_key="test-key-not-real",
        model="test-model",
        timeout_seconds=1,
        client=fake_client,
    )

    report = provider.generate(sanitized_context(behavioral=unavailable_behavioral_context()))

    assert report == expected
    assert captured["store"] is False
    assert captured["text_format"] is InvestigationReport
    assert captured["input"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert captured["input"][1]["role"] == "user"
    assert "<DATA_CONTEXT>" in captured["input"][1]["content"]
    assert "TX-" not in captured["input"][1]["content"]
    assert "origin_key" not in captured["input"][1]["content"]


def test_fallback_handles_no_and_limited_history_without_fabrication() -> None:
    no_history = sanitized_context(behavioral=unavailable_behavioral_context())
    current = historical("TX-current", 2, 250_000)
    limited_behavior = build_behavioral_context(
        current,
        [historical("TX-prior", 1, 10)],
    )
    limited = sanitized_context(behavioral=limited_behavior)

    no_history_response = CopilotService(None).investigate(no_history)
    limited_response = CopilotService(None).investigate(limited)

    assert no_history_response.mode == "deterministic_fallback"
    assert no_history_response.ai_available is False
    assert "No prior behavioral history" in (
        no_history_response.report.behavioral_analysis.history_limitation or ""
    )
    assert "baseline" not in no_history_response.report.behavioral_analysis.summary.lower()
    assert "limited" in (
        limited_response.report.behavioral_analysis.history_limitation or ""
    ).lower()
    assert "does not prove fraud" in limited_response.report.behavioral_analysis.summary


def test_low_risk_fallback_does_not_force_suspicion() -> None:
    response = CopilotService(None).investigate(
        sanitized_context(
            probability=0.01,
            prediction=False,
            behavioral=unavailable_behavioral_context(),
        )
    )
    text = response.report.model_dump_json().lower()

    assert response.report.risk_assessment.level == "LOW"
    assert "does not classify this transaction as fraud" in text
    assert "continue normal processing" in text
    assert "likely fraud" not in text


def test_deterministic_evidence_percentage_titles_remain_grounded() -> None:
    context = sanitized_context(
        probability=0.01,
        prediction=False,
        behavioral=unavailable_behavioral_context(),
    )
    context.evidence.insert(
        0,
        context.evidence[-1].model_copy(
            update={
                "evidence_id": "origin_balance_ratio_context",
                "title": "Amount is below 75% of the origin balance",
                "facts": {"amount_to_origin_balance": 0.05},
            }
        ),
    )

    response = CopilotService(None).investigate(context)

    assert response.mode == "deterministic_fallback"
    assert "75%" in response.report.model_dump_json()


def test_normal_history_does_not_create_a_deviation_narrative() -> None:
    current = historical(
        "TX-current",
        5,
        50,
        transaction_type="PAYMENT",
    )
    behavior = build_behavioral_context(
        current,
        [
            historical("TX-prior-1", 1, 40, transaction_type="PAYMENT"),
            historical("TX-prior-2", 2, 50, transaction_type="PAYMENT"),
            historical("TX-prior-3", 3, 60, transaction_type="PAYMENT"),
        ],
    )
    response = CopilotService(None).investigate(
        sanitized_context(probability=0.01, prediction=False, behavioral=behavior)
    )

    assert response.report.behavioral_analysis.history_limitation is None
    assert "No strong behavioral deviation" in response.report.behavioral_analysis.summary
    assert "suspicious behavior" not in response.report.model_dump_json().lower()


def test_strong_deviation_is_described_without_becoming_a_fraud_claim() -> None:
    behavior = build_behavioral_context(
        historical("TX-current", 2, 250_000),
        [historical("TX-prior", 1, 10)],
    )
    response = CopilotService(None).investigate(sanitized_context(behavioral=behavior))
    analysis = response.report.behavioral_analysis.summary

    assert "exceeds the maximum" in analysis
    assert "does not prove fraud" in analysis
    assert "limited" in (
        response.report.behavioral_analysis.history_limitation or ""
    ).lower()


@pytest.mark.parametrize(
    ("settings", "reason"),
    [
        (Settings(llm_enabled=False), "disabled"),
        (Settings(llm_enabled=True, llm_api_key=None), "not configured"),
        (
            Settings(llm_enabled=True, llm_provider="unsupported", llm_api_key="unused"),
            "unsupported",
        ),
    ],
)
def test_disabled_or_misconfigured_real_provider_falls_back(
    settings: Settings, reason: str
) -> None:
    response = create_copilot_service(settings).investigate(
        sanitized_context(behavioral=unavailable_behavioral_context())
    )

    assert response.mode == "deterministic_fallback"
    assert response.provider == "deterministic_fallback"
    assert reason in (response.fallback_reason or "").lower()


def test_provider_failure_and_invalid_output_are_safely_replaced() -> None:
    class FailedProvider:
        name = "failed"
        model = "failed-model"

        def generate(self, context):
            raise TimeoutError("provider timeout with internal detail")

    class InvalidProvider:
        name = "invalid"
        model = "invalid-model"

        def generate(self, context):
            return {"summary": "missing required report fields"}

    context = sanitized_context(behavioral=unavailable_behavioral_context())
    failed = CopilotService(FailedProvider()).investigate(context)
    invalid = CopilotService(InvalidProvider()).investigate(context)

    assert failed.mode == "deterministic_fallback"
    assert invalid.mode == "deterministic_fallback"
    assert "internal detail" not in (failed.fallback_reason or "")


def test_unsupported_claim_from_valid_provider_report_is_rejected() -> None:
    class UnsupportedClaimProvider(CapturingProvider):
        def generate(self, context):
            report = super().generate(context)
            report.summary = "This proves fraud through money laundering."
            return report

    response = CopilotService(UnsupportedClaimProvider()).investigate(
        sanitized_context(behavioral=unavailable_behavioral_context())
    )

    assert response.mode == "deterministic_fallback"
    assert "money laundering" not in response.report.model_dump_json().lower()


def test_provider_cannot_override_deterministic_simulated_action() -> None:
    class ContradictingProvider(CapturingProvider):
        def generate(self, context):
            report = super().generate(context)
            report.recommended_actions[0].action = "Approve and release the transaction."
            return report

    response = CopilotService(ContradictingProvider()).investigate(
        sanitized_context(behavioral=unavailable_behavioral_context())
    )

    assert response.mode == "deterministic_fallback"
    assert "Approve and release" not in response.report.model_dump_json()


def test_relationship_payload_is_aggregate_only_and_fallback_preserves_limitations() -> None:
    current = RelationshipTransaction(
        transaction_reference="TX-current-private",
        step=2,
        amount=250_000,
        origin_key="C-private-origin",
        destination_key="M-private-destination",
    )
    prior = RelationshipTransaction(
        transaction_reference="TX-prior-private",
        step=1,
        amount=10,
        origin_key="C-private-origin",
        destination_key="M-private-destination",
    )
    relationship = build_relationship_context(current, [prior])
    context = sanitized_context(
        behavioral=unavailable_behavioral_context(),
        relationship=relationship,
    )
    response = CopilotService(None).investigate(context)
    payload = context.model_dump_json()

    assert context.relationship_context.prior_interaction_count == 1
    assert "limited" in (
        response.report.relationship_analysis.history_limitation or ""
    ).lower()
    assert "do not prove fraud" in response.report.relationship_analysis.summary
    for forbidden in (
        "TX-current-private",
        "TX-prior-private",
        "C-private-origin",
        "M-private-destination",
        "origin_key",
        "destination_key",
        "raw_relationship_history",
    ):
        assert forbidden not in payload


def test_provider_cannot_invent_hidden_network_relationships() -> None:
    class InventingProvider(CapturingProvider):
        def generate(self, context):
            report = super().generate(context)
            report.relationship_analysis.summary = (
                "A hidden relationship and shared identity prove network fraud."
            )
            return report

    response = CopilotService(InventingProvider()).investigate(
        sanitized_context(behavioral=unavailable_behavioral_context())
    )

    assert response.mode == "deterministic_fallback"
    assert "hidden relationship" not in response.report.model_dump_json().lower()


def test_provider_cannot_claim_a_seen_relationship_is_new() -> None:
    relationship = build_relationship_context(
        RelationshipTransaction("TX-current", 2, 20, "C-private", "M-private"),
        [RelationshipTransaction("TX-prior", 1, 10, "C-private", "M-private")],
    )

    class ContradictingRelationshipProvider(CapturingProvider):
        def generate(self, context):
            report = super().generate(context)
            report.relationship_analysis.summary = "This is a new relationship."
            return report

    response = CopilotService(ContradictingRelationshipProvider()).investigate(
        sanitized_context(
            behavioral=unavailable_behavioral_context(),
            relationship=relationship,
        )
    )

    assert response.mode == "deterministic_fallback"
    assert "new relationship" not in response.report.relationship_analysis.summary.lower()


def test_settings_repr_never_exposes_api_key() -> None:
    settings = Settings(llm_enabled=True, llm_api_key="sk-test-secret-not-real")

    assert "sk-test-secret-not-real" not in repr(settings)


def test_investigation_report_schema_rejects_extra_or_malformed_fields() -> None:
    valid = CopilotService(None).investigate(
        sanitized_context(behavioral=unavailable_behavioral_context())
    ).report.model_dump()
    valid["unapproved_internal_metadata"] = {"raw_history": []}

    with pytest.raises(ValidationError):
        InvestigationReport.model_validate(valid)

    malformed = {key: value for key, value in valid.items() if key != "risk_assessment"}
    with pytest.raises(ValidationError):
        InvestigationReport.model_validate(malformed)
