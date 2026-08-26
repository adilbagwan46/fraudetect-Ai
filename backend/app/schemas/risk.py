from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
RecommendedAction = Literal["NORMAL_PROCESSING", "MANUAL_REVIEW", "HOLD_FOR_INVESTIGATION"]
EvidenceCategory = Literal[
    "MODEL_RISK",
    "AMOUNT_CONTEXT",
    "BALANCE_CONTEXT",
    "TRANSACTION_TYPE_CONTEXT",
    "TIME_CONTEXT",
    "BEHAVIORAL_CONTEXT",
    "RELATIONSHIP_CONTEXT",
]
EvidenceSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


class RiskPredictionRequest(BaseModel):
    transaction_type: TransactionType
    amount: float = Field(ge=0, allow_inf_nan=False)
    origin_balance_before: float = Field(ge=0, allow_inf_nan=False)
    hour_of_day: int = Field(ge=0, le=23)


class RiskInvestigationRequest(BaseModel):
    """Either a manual scoring input or a safe reference to prepared history."""

    transaction_reference: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    transaction_type: TransactionType | None = None
    amount: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    origin_balance_before: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    hour_of_day: int | None = Field(default=None, ge=0, le=23)

    @model_validator(mode="after")
    def validate_request_mode(self) -> RiskInvestigationRequest:
        manual_values = (
            self.transaction_type,
            self.amount,
            self.origin_balance_before,
            self.hour_of_day,
        )
        supplied_manual_fields = sum(value is not None for value in manual_values)
        if self.transaction_reference is not None and supplied_manual_fields:
            raise ValueError(
                "transaction_reference cannot be combined with manual transaction fields"
            )
        if self.transaction_reference is None and supplied_manual_fields != len(manual_values):
            raise ValueError(
                "provide transaction_reference or all four manual transaction fields"
            )
        return self

    def manual_transaction(self) -> RiskPredictionRequest:
        if self.transaction_reference is not None:
            raise ValueError("Referenced investigation has no manual transaction")
        return RiskPredictionRequest(
            transaction_type=self.transaction_type,
            amount=self.amount,
            origin_balance_before=self.origin_balance_before,
            hour_of_day=self.hour_of_day,
        )


class EvidenceItem(BaseModel):
    id: str
    category: EvidenceCategory
    severity: EvidenceSeverity
    title: str
    description: str
    facts: dict[str, Any]


class DerivedFeatures(BaseModel):
    log_amount: float = Field(allow_inf_nan=False)
    amount_to_origin_balance: float = Field(allow_inf_nan=False)


class ModelOutputContext(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    fraud_prediction: bool
    classification_threshold: float = Field(ge=0, le=1)
    operating_mode: Literal["BALANCED"]
    model_version: str


class PriorAmountContext(BaseModel):
    average: float = Field(allow_inf_nan=False)
    median: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)


class CurrentAmountBehavior(BaseModel):
    amount_vs_prior_average: float | None = Field(default=None, allow_inf_nan=False)
    amount_vs_prior_median: float | None = Field(default=None, allow_inf_nan=False)
    amount_vs_prior_maximum: float | None = Field(default=None, allow_inf_nan=False)
    prior_empirical_percentile: float = Field(ge=0, le=1, allow_inf_nan=False)
    exceeds_prior_maximum: bool


class RecentWindowActivity(BaseModel):
    window_steps: int = Field(gt=0)
    prior_transaction_count: int = Field(ge=0)
    prior_amount_total: float = Field(ge=0, allow_inf_nan=False)


class RecentActivityContext(BaseModel):
    windows: list[RecentWindowActivity]
    steps_since_previous_transaction: int | None = Field(default=None, ge=1)


class TransactionTypeBehavior(BaseModel):
    prior_transaction_type_count: int = Field(ge=0)
    is_new_transaction_type_for_origin: bool


class BehavioralContext(BaseModel):
    history_available: bool
    availability_explanation: str
    prior_transaction_count: int = Field(ge=0)
    prior_total_amount: float = Field(ge=0, allow_inf_nan=False)
    prior_amount: PriorAmountContext | None = None
    current_amount_context: CurrentAmountBehavior | None = None
    recent_activity: RecentActivityContext
    transaction_type_context: TransactionTypeBehavior


class RelationshipAmountContext(BaseModel):
    average: float = Field(ge=0, allow_inf_nan=False)
    median: float = Field(ge=0, allow_inf_nan=False)
    maximum: float = Field(ge=0, allow_inf_nan=False)


class CurrentRelationshipAmountContext(BaseModel):
    amount_vs_prior_average: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    amount_vs_prior_median: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    amount_vs_prior_maximum: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    prior_empirical_percentile: float = Field(ge=0, le=1, allow_inf_nan=False)
    exceeds_prior_relationship_maximum: bool


class OriginNetworkContext(BaseModel):
    prior_unique_counterparty_count: int = Field(ge=0)
    prior_transaction_count: int = Field(ge=0)
    current_destination_is_new: bool | None


class DestinationNetworkContext(BaseModel):
    prior_unique_origin_count: int = Field(ge=0)
    prior_transaction_count: int = Field(ge=0)
    current_origin_is_new_for_destination: bool | None


class RelationshipContext(BaseModel):
    """Identifier-free aggregates computed with historical.step < current.step."""

    context_available: bool
    history_available: bool
    availability_explanation: str
    relationship_seen_before: bool | None
    relationship_first_seen: bool | None
    prior_interaction_count: int = Field(ge=0)
    prior_total_amount: float = Field(ge=0, allow_inf_nan=False)
    prior_amount: RelationshipAmountContext | None
    current_amount_context: CurrentRelationshipAmountContext | None
    steps_since_previous_interaction: int | None = Field(default=None, ge=1)
    baseline_is_limited: bool
    origin_network: OriginNetworkContext
    destination_network: DestinationNetworkContext


class InvestigationContext(BaseModel):
    transaction: RiskPredictionRequest
    derived_features: DerivedFeatures
    model_output: ModelOutputContext
    evidence: list[EvidenceItem]
    reference_profile_version: str
    approved_reference_statistics: dict[str, Any]
    behavioral_context: BehavioralContext
    relationship_context: RelationshipContext
    relationship_evidence: list[EvidenceItem]


class RiskPredictionResponse(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    fraud_prediction: bool
    classification_threshold: float = Field(ge=0, le=1)
    operating_mode: Literal["BALANCED"]
    model_version: str
    evidence: list[EvidenceItem]
    recommended_action: RecommendedAction
    recommendation_is_simulated: Literal[True] = True


class ModelStatusResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    model_version: str | None = None
    model_family: str | None = None
    operating_mode: str | None = None
    dataset_sha256: str | None = None
    reference_profile_version: str | None = None
    message: str
