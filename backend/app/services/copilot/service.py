from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.schemas.copilot import (
    CopilotInvestigationResponse,
    InvestigationReport,
    ReportBehavioralAnalysis,
    ReportRecommendedAction,
    ReportRiskAssessment,
    ReportSignal,
    SanitizedInvestigationContext,
)
from backend.app.services.copilot.provider import (
    CopilotProviderError,
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
    r"\bunfamiliar location\b",
    r"\bchanged devices?\b",
    r"\bnew devices?\b",
    r"\bmoney laundering\b",
    r"\bknown criminal\b",
    r"(?<!not )\bproves? fraud\b",
    r"\bfraudulent transaction\b",
    r"\bclose (?:the )?(?:customer'?s )?account\b",
    r"\bblacklist\b",
    r"\bpermanently block\b",
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

    behavior = context.behavioral_context
    limitation = (report.behavioral_analysis.history_limitation or "").lower()
    if not behavior.history_available and not (
        "no prior" in limitation or "unavailable" in limitation
    ):
        raise CopilotGroundingError("No-history report must state the behavioral limitation")
    if behavior.history_available and behavior.prior_transaction_count <= 2:
        if "limited" not in limitation and "only" not in limitation:
            raise CopilotGroundingError("Limited behavioral history must be qualified")

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
        observations.append("the transaction type is new in available prior history")
    if behavior.recent_activity.steps_since_previous_transaction == 1:
        observations.append("origin activity occurred in the immediately preceding step")
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
    ) -> None:
        self.provider = provider
        self.fallback_reason = fallback_reason

    def investigate(
        self,
        context: SanitizedInvestigationContext,
    ) -> CopilotInvestigationResponse:
        if self.provider is not None:
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
                )
            except (CopilotProviderError, CopilotGroundingError, ValidationError, ValueError):
                fallback_reason = "Provider unavailable or returned an invalid grounded report"
            except Exception:
                fallback_reason = "Provider failed unexpectedly"
        else:
            fallback_reason = self.fallback_reason or "LLM is disabled"

        report = _fallback_report(context)
        validate_report_grounding(report, context)
        return CopilotInvestigationResponse(
            report=report,
            provider="deterministic_fallback",
            mode="deterministic_fallback",
            ai_available=False,
            fallback_reason=fallback_reason,
        )


def create_copilot_service(settings: Settings) -> CopilotService:
    if not settings.llm_enabled:
        return CopilotService(None, fallback_reason="LLM is disabled by configuration")
    if settings.llm_provider != "openai":
        return CopilotService(None, fallback_reason="Configured LLM provider is unsupported")
    if not settings.llm_api_key:
        return CopilotService(None, fallback_reason="OpenAI API key is not configured")
    try:
        provider = OpenAIInvestigationProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    except CopilotProviderError:
        return CopilotService(None, fallback_reason="OpenAI provider is unavailable")
    return CopilotService(provider)
