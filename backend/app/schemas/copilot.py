from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.risk import (
    BehavioralContext,
    EvidenceCategory,
    EvidenceSeverity,
    RecommendedAction,
    RelationshipContext,
    RiskLevel,
    TransactionType,
)

CopilotMode = Literal["real_llm", "deterministic_fallback"]
CopilotFailureCategory = Literal[
    "disabled",
    "missing_credentials",
    "unsupported_provider",
    "provider_unavailable",
    "provider_timeout",
    "provider_error",
    "invalid_output",
    "grounding_rejected",
    "unexpected_error",
]
SignalImportance = Literal["HIGH", "MEDIUM", "LOW", "INFO"]


class SanitizedTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_type: TransactionType
    amount: float = Field(ge=0, allow_inf_nan=False)
    origin_balance_before: float = Field(ge=0, allow_inf_nan=False)
    hour_of_day: int = Field(ge=0, le=23)


class SanitizedModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fraud_probability: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    fraud_prediction: bool
    classification_threshold: float = Field(ge=0, le=1)
    operating_mode: Literal["BALANCED"]
    recommended_simulated_action: RecommendedAction


class SanitizedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    title: str
    severity: EvidenceSeverity
    category: EvidenceCategory
    facts: dict[str, Any]


class SanitizedReferenceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_split: Literal["train"]
    source_step_range: tuple[int, int]
    overall_training_fraud_prevalence: float = Field(ge=0, le=1)
    dataset_kind: Literal["PaySim synthetic simulation"] = "PaySim synthetic simulation"


class SanitizedInvestigationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction: SanitizedTransaction
    model_output: SanitizedModelOutput
    evidence: list[SanitizedEvidence]
    reference_context: SanitizedReferenceContext
    behavioral_context: BehavioralContext
    relationship_context: RelationshipContext


class ReportRiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: RiskLevel
    assessment: str = Field(min_length=1, max_length=600)


class ReportSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str = Field(min_length=1, max_length=240)
    importance: SignalImportance
    explanation: str = Field(min_length=1, max_length=600)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)


class ReportBehavioralAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=700)
    history_limitation: str | None = Field(default=None, max_length=500)


class ReportRelationshipAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=700)
    history_limitation: str | None = Field(default=None, max_length=500)
    evidence_ids: list[str] = Field(max_length=4)


class ReportRecommendedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, max_length=240)
    reason: str = Field(min_length=1, max_length=500)


class InvestigationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=900)
    risk_assessment: ReportRiskAssessment
    key_signals: list[ReportSignal] = Field(max_length=5)
    behavioral_analysis: ReportBehavioralAnalysis
    relationship_analysis: ReportRelationshipAnalysis
    uncertainties: list[str] = Field(min_length=1, max_length=5)
    recommended_actions: list[ReportRecommendedAction] = Field(min_length=1, max_length=4)
    analyst_note: str = Field(min_length=1, max_length=500)
    disclaimer: str = Field(min_length=1, max_length=500)


class CopilotExecutionMetadata(BaseModel):
    """Safe facts about this generation attempt; absent on legacy stored reports."""

    model_config = ConfigDict(extra="forbid")

    generated_by: Literal["real_provider", "deterministic_fallback"]
    provider_attempted: bool
    provider_succeeded: bool
    generation_latency_ms: int | None = Field(default=None, ge=0)
    failure_category: CopilotFailureCategory | None = None

    @model_validator(mode="after")
    def require_consistent_provenance(self) -> CopilotExecutionMetadata:
        if self.generated_by == "real_provider":
            if not self.provider_attempted or not self.provider_succeeded:
                raise ValueError("real-provider output requires a successful provider attempt")
            if self.failure_category is not None:
                raise ValueError("successful provider output cannot have a failure category")
        else:
            if self.provider_succeeded:
                raise ValueError("deterministic fallback cannot report provider success")
            if self.failure_category is None:
                raise ValueError("deterministic fallback requires a bounded failure category")
        if self.provider_attempted != (self.generation_latency_ms is not None):
            raise ValueError("latency is present exactly when a provider was attempted")
        return self


class CopilotInvestigationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: InvestigationReport
    provider: str
    mode: CopilotMode
    ai_available: bool
    model: str | None = None
    fallback_reason: str | None = None
    relationship_context: RelationshipContext
    execution: CopilotExecutionMetadata | None = None

    @model_validator(mode="after")
    def require_truthful_mode_metadata(self) -> CopilotInvestigationResponse:
        if self.mode == "real_llm":
            if not self.ai_available or self.provider == "deterministic_fallback":
                raise ValueError("real LLM mode requires an available real provider")
            if self.fallback_reason is not None:
                raise ValueError("real LLM output cannot have a fallback reason")
            if self.execution is not None and self.execution.generated_by != "real_provider":
                raise ValueError("real LLM mode contradicts execution provenance")
        else:
            if self.ai_available or self.provider != "deterministic_fallback":
                raise ValueError("fallback mode must identify deterministic fallback")
            if self.fallback_reason is None:
                raise ValueError("fallback mode requires a safe fallback reason")
            if (
                self.execution is not None
                and self.execution.generated_by != "deterministic_fallback"
            ):
                raise ValueError("fallback mode contradicts execution provenance")
        return self
