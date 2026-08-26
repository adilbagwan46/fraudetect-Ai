from typing import Any, Literal

from pydantic import BaseModel, Field

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
RecommendedAction = Literal["NORMAL_PROCESSING", "MANUAL_REVIEW", "HOLD_FOR_INVESTIGATION"]
EvidenceCategory = Literal[
    "MODEL_RISK",
    "AMOUNT_CONTEXT",
    "BALANCE_CONTEXT",
    "TRANSACTION_TYPE_CONTEXT",
    "TIME_CONTEXT",
]
EvidenceSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


class RiskPredictionRequest(BaseModel):
    transaction_type: TransactionType
    amount: float = Field(ge=0, allow_inf_nan=False)
    origin_balance_before: float = Field(ge=0, allow_inf_nan=False)
    hour_of_day: int = Field(ge=0, le=23)


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


class InvestigationContext(BaseModel):
    transaction: RiskPredictionRequest
    derived_features: DerivedFeatures
    model_output: ModelOutputContext
    evidence: list[EvidenceItem]
    reference_profile_version: str
    approved_reference_statistics: dict[str, Any]


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
