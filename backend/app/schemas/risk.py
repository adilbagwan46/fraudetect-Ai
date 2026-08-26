from typing import Literal

from pydantic import BaseModel, Field

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
RecommendedAction = Literal["NORMAL_PROCESSING", "MANUAL_REVIEW", "HOLD_FOR_INVESTIGATION"]


class RiskPredictionRequest(BaseModel):
    transaction_type: TransactionType
    amount: float = Field(ge=0, allow_inf_nan=False)
    origin_balance_before: float = Field(ge=0, allow_inf_nan=False)
    hour_of_day: int = Field(ge=0, le=23)


class RiskPredictionResponse(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    fraud_prediction: bool
    classification_threshold: float = Field(ge=0, le=1)
    operating_mode: Literal["BALANCED"]
    model_version: str
    recommended_action: RecommendedAction
    recommendation_is_simulated: Literal[True] = True


class ModelStatusResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    model_version: str | None = None
    model_family: str | None = None
    operating_mode: str | None = None
    dataset_sha256: str | None = None
    message: str

