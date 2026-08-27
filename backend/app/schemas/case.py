from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.copilot import (
    CopilotInvestigationResponse,
    SanitizedInvestigationContext,
)
from backend.app.schemas.risk import (
    EvidenceCategory,
    EvidenceSeverity,
    RiskInvestigationRequest,
    RiskLevel,
)

CaseStatus = Literal["OPEN", "IN_REVIEW", "ESCALATED", "CLEARED", "CLOSED"]
CasePriority = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
CaseSourceType = Literal["REFERENCE", "MANUAL"]
AnalystDisposition = Literal["NONE", "CLEARED", "SUSPICIOUS", "ESCALATED"]


class CaseCreateRequest(RiskInvestigationRequest):
    """Create an immutable case snapshot from either supported investigation mode."""

    model_config = ConfigDict(extra="forbid")


class CaseUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CaseStatus | None = None
    analyst_note: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def require_update(self) -> CaseUpdateRequest:
        if self.status is None and self.analyst_note is None:
            raise ValueError("provide status or analyst_note")
        if self.analyst_note is not None:
            normalized = self.analyst_note.strip()
            if not normalized:
                raise ValueError("analyst_note cannot be blank")
            self.analyst_note = normalized
        return self


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_count: int = Field(ge=0)
    highest_severity: EvidenceSeverity | None
    severity_counts: dict[EvidenceSeverity, int]
    category_counts: dict[EvidenceCategory, int]


class AnalystDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: AnalystDisposition
    note: str | None
    disposition_at: datetime | None
    is_model_ground_truth: Literal[False] = False


class CaseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^CASE-[A-F0-9]{16}$")
    status: CaseStatus
    priority: CasePriority
    created_at: datetime
    updated_at: datetime
    source_type: CaseSourceType
    transaction_reference_available: bool
    model_version: str
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    operating_mode: Literal["BALANCED"]
    evidence_summary: EvidenceSummary
    analyst_decision: AnalystDecision


class CaseStatusHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    occurred_at: datetime
    previous_status: CaseStatus | None
    new_status: CaseStatus
    disposition: AnalystDisposition
    note_recorded: bool


class DecisionTraceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: Literal[
        "CASE_CREATED",
        "INTELLIGENCE_CAPTURED",
        "COPILOT_GENERATED",
        "ANALYST_REVIEWED",
        "CASE_ESCALATED",
        "CASE_CLEARED",
        "CASE_CLOSED",
    ]
    occurred_at: datetime
    actor: Literal["SYSTEM", "COPILOT", "ANALYST"]
    label: str
    detail: str


class CaseDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: CaseSummary
    intelligence_snapshot: SanitizedInvestigationContext
    investigation_limitations: list[str]
    copilot: CopilotInvestigationResponse | None
    status_history: list[CaseStatusHistoryItem]
    decision_trace: list[DecisionTraceItem]
    snapshot_is_immutable: Literal[True] = True


class CaseListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CaseSummary]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
