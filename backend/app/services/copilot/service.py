from __future__ import annotations

import re
import time
from typing import Any

from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.schemas.copilot import (
    CopilotExecutionMetadata,
    CopilotFailureCategory,
    CopilotInvestigationResponse,
    InvestigationReport,
    ReportBehavioralAnalysis,
    ReportRecommendedAction,
    ReportRelationshipAnalysis,
    ReportRiskAssessment,
    ReportSignal,
    SanitizedInvestigationContext,
)
from backend.app.services.copilot.provider import (
    CopilotProviderInvalidOutputError,
    CopilotProviderTimeoutError,
    CopilotProviderUnavailableError,
    GeminiInvestigationProvider,
    InvestigationLLMProvider,
    OpenAIInvestigationProvider,
)

REAL_LLM_DISCLAIMER = (
    "This is an AI-generated advisory investigation summary based only on supplied model output "
    "and deterministic evidence. It uses synthetic PaySim data and is not a production fraud "
    "decision system."
)
FALLBACK_DISCLAIMER = (
    "This is a deterministic fallback investigation summary, not LLM-generated output. It uses "
    "synthetic PaySim data and is not a production fraud decision system."
)

FORBIDDEN_REPORT_PATTERNS = (
    r"\bTX-[A-Za-z0-9]",
    r"\b[CM]\d{6,}\b",
    r"\bnameOrig\b",
    r"\bnameDest\b",
    r"\btransaction_reference\b",
    r"\borigin_key\b",
    r"\bdestination_key\b",
    r"\braw_relationship_history\b",
    r"\bunfamiliar location\b",
    r"\bchanged devices?\b",
    r"\bnew devices?\b",
    r"\bmoney laundering\b",
    r"\bknown criminal\b",
    r"(?<!not )\bproves? fraud\b",
    r"\b(?:confirms?|demonstrates?|establishes?) (?:that )?(?:this |the )?"
    r"(?:transaction )?(?:is )?fraud\b",
    r"\bfraudulent transaction\b",
    r"\bclose (?:the )?(?:customer'?s )?account\b",
    r"\bterminate (?:the )?(?:customer|account|relationship)\b",
    r"\breport (?:the )?(?:customer|account holder|transaction) to "
    r"(?:law enforcement|police|authorities)\b",
    r"\bblacklist\b",
    r"\bpermanently block\b",
    r"\bidentit(?:y|ies)\b",
    r"\bshared identit(?:y|ies)\b",
    r"\bhidden relationships?\b",
    r"\bunknown network connections?\b",
    r"\bnovelty proves? fraud\b",
)

SIGNAL_EXPLANATIONS = {
    "model_risk_above_threshold": (
        "The frozen model score is above its active BALANCED threshold. This is model output, "
        "not a causal explanation."
    ),
    "model_risk_below_threshold": (
        "The frozen model score remains below its active BALANCED threshold."
    ),
    "amount_reference_context": (
        "Deterministic evidence compares the amount with approved training-reference aggregates."
    ),
    "origin_balance_ratio_context": (
        "Deterministic evidence compares the amount with the recorded pre-transaction balance."
    ),
    "transaction_type_training_context": (
        "The transaction type is described using aggregate PaySim training-reference context."
    ),
    "hour_training_context": (
        "The PaySim step-derived hour is described using aggregate training-reference context."
    ),
    "behavior_history_unavailable": (
        "No earlier-step origin history is present, so no behavioral baseline is available."
    ),
    "behavior_amount_above_typical": (
        "The amount differs substantially from the available earlier-step amount baseline."
    ),
    "behavior_amount_above_prior_maximum": (
        "The amount exceeds the maximum observed in eligible earlier-step origin history."
    ),
    "behavior_recent_activity": (
        "Eligible origin activity exists in the immediately preceding PaySim step."
    ),
    "behavior_new_transaction_type": (
        "This type was not observed in the available earlier-step history for the origin."
    ),
    "relationship_context_unavailable": (
        "Relationship aggregates are unavailable for this investigation input."
    ),
    "relationship_new_counterparty": (
        "No earlier-step interaction was observed for this origin-destination pair."
    ),
    "relationship_previously_observed": (
        "The pair has at least one interaction at a strictly earlier PaySim step."
    ),
    "relationship_limited_history": (
        "The relationship baseline contains no more than two earlier interactions."
    ),
    "relationship_amount_deviation": (
        "The amount is at least five times the deterministic prior relationship average."
    ),
    "relationship_exceeds_prior_maximum": (
        "The amount exceeds the maximum in strictly earlier relationship history."
    ),
}


class CopilotGroundingError(ValueError):
    """Raised when a structured report contradicts or exceeds its approved context."""


def _report_text(report: InvestigationReport) -> str:
    return report.model_dump_json()


def validate_report_grounding(
    report: InvestigationReport,
    context: SanitizedInvestigationContext,
) -> None:
    if report.risk_assessment.level != context.model_output.risk_level:
        raise CopilotGroundingError("Report risk level contradicts frozen model output")

    approved_ids = {item.evidence_id for item in context.evidence}
    for signal in report.key_signals:
        if not set(signal.evidence_ids).issubset(approved_ids):
            raise CopilotGroundingError("Report cites evidence outside the approved context")
    if not set(report.relationship_analysis.evidence_ids).issubset(approved_ids):
        raise CopilotGroundingError(
            "Relationship analysis cites evidence outside the approved context"
        )

    behavior = context.behavioral_context
    limitation = (report.behavioral_analysis.history_limitation or "").lower()
    behavior_summary = report.behavioral_analysis.summary.lower()
    if not behavior.history_available and not (
        "no prior" in limitation or "unavailable" in limitation
    ):
        raise CopilotGroundingError("No-history report must state the behavioral limitation")
    if not behavior.history_available and not any(
        marker in behavior_summary
        for marker in ("no behavioral comparison", "no prior", "unavailable")
    ):
        raise CopilotGroundingError("No-history behavioral summary must not imply observations")
    if not behavior.history_available and re.search(
        r"\b(?:prior transactions? show|previous transactions? (?:show|indicate)|history shows)\b",
        report.behavioral_analysis.summary,
        flags=re.IGNORECASE,
    ):
        raise CopilotGroundingError("Report invents behavioral history")
    if behavior.history_available and behavior.prior_transaction_count <= 2:
        if "limited" not in limitation and "only" not in limitation:
            raise CopilotGroundingError("Limited behavioral history must be qualified")

    relationship = context.relationship_context
    relationship_text = (
        f"{report.relationship_analysis.summary} "
        f"{report.relationship_analysis.history_limitation or ''}"
    ).lower()
    relationship_summary = report.relationship_analysis.summary.lower()
    relationship_ids = {
        item.evidence_id
        for item in context.evidence
        if item.category == "RELATIONSHIP_CONTEXT"
    }
    if relationship.context_available and relationship_ids and not (
        set(report.relationship_analysis.evidence_ids) & relationship_ids
    ):
        raise CopilotGroundingError("Relationship analysis must cite relationship evidence")
    if not relationship.context_available and "unavailable" not in relationship_summary:
        raise CopilotGroundingError("Unavailable relationship context must be disclosed")
    if relationship.context_available and not relationship.history_available:
        if (
            "no prior" not in relationship_summary
            and "new relationship" not in relationship_summary
        ):
            raise CopilotGroundingError("New relationship status must be stated accurately")
        if "previously observed" in relationship_text:
            raise CopilotGroundingError("Report invents prior relationship history")
    if relationship.history_available:
        if "no prior" in relationship_text or "new relationship" in relationship_text:
            raise CopilotGroundingError("Report contradicts prior relationship history")
        if relationship.prior_interaction_count <= 2 and not any(
            marker in relationship_text for marker in ("limited", "only", "sparse")
        ):
            raise CopilotGroundingError("Limited relationship history must be qualified")

    text = _report_text(report)
    for pattern in FORBIDDEN_REPORT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise CopilotGroundingError("Report contains an unsupported or forbidden claim")

    allowed_percentages = [
        context.model_output.risk_score,
        context.model_output.classification_threshold * 100,
        context.reference_context.overall_training_fraud_prevalence * 100,
    ]
    for evidence in context.evidence:
        allowed_percentages.extend(
            float(match.group(1))
            for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)%", evidence.title)
        )
        for key, value in evidence.facts.items():
            if isinstance(value, (int, float)) and any(
                marker in key
                for marker in ("probability", "threshold", "percentile", "prevalence", "share")
            ):
                allowed_percentages.extend((float(value), float(value) * 100))
    for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)%", text):
        percentage = float(match.group(1))
        if not any(abs(percentage - allowed) <= 0.11 for allowed in allowed_percentages):
            raise CopilotGroundingError("Report introduced an unsupported percentage")
    for match in re.finditer(
        r"\bfraud probability(?: is| of)?\s+(0(?:\.\d+)?|1(?:\.0+)?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        if abs(float(match.group(1)) - context.model_output.fraud_probability) > 0.0001:
            raise CopilotGroundingError("Report changed the frozen fraud probability")

    action_text = " ".join(
        item.action.lower() for item in report.recommended_actions
    )
    simulated_action = context.model_output.recommended_simulated_action
    if simulated_action == "HOLD_FOR_INVESTIGATION" and any(
        phrase in action_text for phrase in ("continue normal processing", "approve", "release")
    ):
        raise CopilotGroundingError("Report contradicts the deterministic simulated action")
    if simulated_action == "NORMAL_PROCESSING" and any(
        phrase in action_text for phrase in ("temporary hold", "block", "freeze")
    ):
        raise CopilotGroundingError("Report contradicts the deterministic simulated action")


def _behavioral_analysis(
    context: SanitizedInvestigationContext,
) -> ReportBehavioralAnalysis:
    behavior = context.behavioral_context
    if not behavior.history_available:
        return ReportBehavioralAnalysis(
            summary="No behavioral comparison can be made from the supplied context.",
            history_limitation=(
                "No prior behavioral history is available in the current investigation "
                "context, so behavioral comparison is limited."
            ),
        )

    amount = behavior.current_amount_context
    observations = []
    if amount is not None and amount.exceeds_prior_maximum:
        observations.append("The amount exceeds the maximum in eligible prior history")
    if behavior.transaction_type_context.is_new_transaction_type_for_origin:
        observations.append("The transaction type is new in available prior history")
    if behavior.recent_activity.steps_since_previous_transaction == 1:
        observations.append("Origin activity occurred in the immediately preceding step")
    if not observations:
        observations.append("No strong behavioral deviation is established by the supplied context")
    summary = ". ".join(observations) + ". Behavioral deviation does not prove fraud."

    limitation = None
    if behavior.prior_transaction_count <= 2:
        singular = behavior.prior_transaction_count == 1
        noun = "transaction" if singular else "transactions"
        verb = "is" if singular else "are"
        limitation = (
            f"The behavioral baseline is limited because only "
            f"{behavior.prior_transaction_count} prior {noun} {verb} available."
        )
    return ReportBehavioralAnalysis(summary=summary, history_limitation=limitation)


def _relationship_analysis(
    context: SanitizedInvestigationContext,
) -> ReportRelationshipAnalysis:
    relationship = context.relationship_context
    evidence_ids = [
        item.evidence_id
        for item in context.evidence
        if item.category == "RELATIONSHIP_CONTEXT"
    ][:4]
    if not relationship.context_available:
        return ReportRelationshipAnalysis(
            summary="Relationship comparison is unavailable for this investigation input.",
            history_limitation=(
                "Relationship context is unavailable, so no relationship or network baseline "
                "can be described."
            ),
            evidence_ids=evidence_ids,
        )
    if not relationship.history_available:
        return ReportRelationshipAnalysis(
            summary=(
                "No prior origin-destination interaction was observed before the current step. "
                "Relationship novelty does not prove fraud."
            ),
            history_limitation=(
                "No prior relationship baseline is available for amount comparison."
            ),
            evidence_ids=evidence_ids,
        )

    observations = [
        f"The relationship was observed in {relationship.prior_interaction_count} prior "
        f"interaction{'s' if relationship.prior_interaction_count != 1 else ''}"
    ]
    amount = relationship.current_amount_context
    if amount is not None and amount.exceeds_prior_relationship_maximum:
        observations.append("the amount exceeds the prior relationship maximum")
    else:
        observations.append("the amount does not exceed the prior relationship maximum")
    limitation = None
    if relationship.baseline_is_limited:
        limitation = (
            "The relationship baseline is limited because only "
            f"{relationship.prior_interaction_count} prior interaction"
            f"{'s are' if relationship.prior_interaction_count != 1 else ' is'} available."
        )
    return ReportRelationshipAnalysis(
        summary="; ".join(observations) + ". Relationship patterns do not prove fraud.",
        history_limitation=limitation,
        evidence_ids=evidence_ids,
    )


def _fallback_report(context: SanitizedInvestigationContext) -> InvestigationReport:
    model = context.model_output
    if model.fraud_prediction:
        summary = (
            f"The frozen model assigns {model.risk_level} risk and places the transaction above "
            "the active threshold. Deterministic evidence supports analyst review, but does not "
            "prove fraud."
        )
        assessment = (
            "The model classification supports investigation under the existing simulated "
            "operating policy. The model output has not been altered by the Copilot."
        )
    else:
        summary = (
            f"The frozen model assigns {model.risk_level} risk and remains below the active "
            "threshold. The supplied evidence does not justify an artificial suspicious narrative."
        )
        assessment = (
            "The model does not classify this transaction as fraud. Continue normal processing "
            "subject to standard controls unless independent evidence warrants review."
        )

    key_signals = [
        ReportSignal(
            signal=item.title,
            importance="HIGH" if item.severity in {"CRITICAL", "HIGH"} else item.severity,
            explanation=SIGNAL_EXPLANATIONS.get(
                item.evidence_id,
                "This signal is present in the approved deterministic evidence context.",
            ),
            evidence_ids=[item.evidence_id],
        )
        for item in context.evidence[:3]
    ]

    uncertainties = [
        "The demonstration uses synthetic PaySim data and cannot establish production behavior."
    ]
    behavior = context.behavioral_context
    if not behavior.history_available:
        uncertainties.append("No prior origin history is available for behavioral comparison.")
    elif behavior.prior_transaction_count <= 2:
        uncertainties.append(
            "The behavioral baseline contains too few prior transactions for a strong conclusion."
        )

    if model.recommended_simulated_action == "NORMAL_PROCESSING":
        actions = [
            ReportRecommendedAction(
                action="Continue normal processing subject to standard controls.",
                reason="The frozen model score is below the active classification threshold.",
            )
        ]
    else:
        actions = [
            ReportRecommendedAction(
                action="Review the transaction using authorized internal systems.",
                reason="Confirm the strongest deterministic signals before taking further action.",
            ),
            ReportRecommendedAction(
                action="Verify the transaction with the account holder when policy permits.",
                reason=(
                    "Independent verification can resolve uncertainty without treating "
                    "deviation as proof."
                ),
            ),
        ]
        if model.recommended_simulated_action == "HOLD_FOR_INVESTIGATION":
            actions.append(
                ReportRecommendedAction(
                    action="Consider a temporary hold according to organizational policy.",
                    reason=(
                        "The frozen operating policy recommends investigation before processing."
                    ),
                )
            )

    return InvestigationReport(
        summary=summary,
        risk_assessment=ReportRiskAssessment(
            level=model.risk_level,
            assessment=assessment,
        ),
        key_signals=key_signals,
        behavioral_analysis=_behavioral_analysis(context),
        relationship_analysis=_relationship_analysis(context),
        uncertainties=uncertainties,
        recommended_actions=actions,
        analyst_note=(
            "This report organizes approved evidence for review; the human analyst retains "
            "responsibility for any decision."
        ),
        disclaimer=FALLBACK_DISCLAIMER,
    )


class CopilotService:
    def __init__(
        self,
        provider: InvestigationLLMProvider | None,
        *,
        fallback_reason: str | None = None,
        fallback_category: CopilotFailureCategory | None = None,
    ) -> None:
        self.provider = provider
        self.fallback_reason = fallback_reason
        self.fallback_category = fallback_category

    def investigate(
        self,
        context: SanitizedInvestigationContext,
    ) -> CopilotInvestigationResponse:
        started_at: float | None = None
        failure_category: CopilotFailureCategory | None = None
        if self.provider is not None:
            started_at = time.perf_counter()
            try:
                raw_report: Any = self.provider.generate(context)
                report = InvestigationReport.model_validate(raw_report).model_copy(
                    update={"disclaimer": REAL_LLM_DISCLAIMER}
                )
                validate_report_grounding(report, context)
                return CopilotInvestigationResponse(
                    report=report,
                    provider=self.provider.name,
                    mode="real_llm",
                    ai_available=True,
                    model=self.provider.model,
                    relationship_context=context.relationship_context,
                    execution=CopilotExecutionMetadata(
                        generated_by="real_provider",
                        provider_attempted=True,
                        provider_succeeded=True,
                        generation_latency_ms=max(
                            0, round((time.perf_counter() - started_at) * 1000)
                        ),
                    ),
                )
            except (CopilotProviderTimeoutError, TimeoutError):
                fallback_reason = "LLM provider timed out"
                failure_category = "provider_timeout"
            except CopilotGroundingError:
                fallback_reason = "LLM provider output failed grounding validation"
                failure_category = "grounding_rejected"
            except (CopilotProviderInvalidOutputError, ValidationError, ValueError):
                fallback_reason = "LLM provider returned invalid structured output"
                failure_category = "invalid_output"
            except CopilotProviderUnavailableError:
                fallback_reason = "LLM provider request failed"
                failure_category = "provider_error"
            except Exception:
                fallback_reason = "Provider failed unexpectedly"
                failure_category = "unexpected_error"
        else:
            fallback_reason = self.fallback_reason or "LLM is disabled"
            failure_category = self.fallback_category or "disabled"

        report = _fallback_report(context)
        validate_report_grounding(report, context)
        return CopilotInvestigationResponse(
            report=report,
            provider="deterministic_fallback",
            mode="deterministic_fallback",
            ai_available=False,
            fallback_reason=fallback_reason,
            relationship_context=context.relationship_context,
            execution=CopilotExecutionMetadata(
                generated_by="deterministic_fallback",
                provider_attempted=self.provider is not None,
                provider_succeeded=False,
                generation_latency_ms=(
                    max(0, round((time.perf_counter() - started_at) * 1000))
                    if started_at is not None
                    else None
                ),
                failure_category=failure_category,
            ),
        )


def create_copilot_service(settings: Settings) -> CopilotService:
    if not settings.llm_enabled:
        return CopilotService(
            None,
            fallback_reason="LLM is disabled by configuration",
            fallback_category="disabled",
        )
    if settings.llm_provider == "openai":
        api_key = settings.llm_api_key
        model = settings.llm_model
        provider_class = OpenAIInvestigationProvider
        provider_label = "OpenAI"
    elif settings.llm_provider == "gemini":
        api_key = settings.gemini_api_key
        model = settings.gemini_model
        provider_class = GeminiInvestigationProvider
        provider_label = "Gemini"
    else:
        return CopilotService(
            None,
            fallback_reason="Configured LLM provider is unsupported",
            fallback_category="unsupported_provider",
        )
    if not api_key:
        return CopilotService(
            None,
            fallback_reason=f"{provider_label} API key is not configured",
            fallback_category="missing_credentials",
        )
    try:
        provider = provider_class(
            api_key=api_key,
            model=model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    except CopilotProviderUnavailableError:
        return CopilotService(
            None,
            fallback_reason=f"{provider_label} provider is unavailable",
            fallback_category="provider_unavailable",
        )
    return CopilotService(provider)
