from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.copilot import CopilotInvestigationResponse
from backend.app.schemas.risk import (
    InvestigationContext,
    ModelStatusResponse,
    RiskInvestigationRequest,
    RiskPredictionRequest,
    RiskPredictionResponse,
)
from backend.app.services.behavioral_service import (
    BehaviorHistoryUnavailableError,
    SQLitePaySimHistoryProvider,
    TransactionReferenceNotFoundError,
)
from backend.app.services.copilot.context_builder import build_sanitized_context
from backend.app.services.copilot.service import CopilotService, create_copilot_service
from backend.app.services.relationship_service import (
    RelationshipHistoryUnavailableError,
    RelationshipTransactionNotFoundError,
    SQLiteRelationshipHistoryProvider,
)
from backend.app.services.risk_service import (
    ModelUnavailableError,
    investigate_risk,
    load_active_bundle,
    predict_risk,
)

router = APIRouter(tags=["model"])
SettingsDependency = Annotated[Settings, Depends(get_settings)]


def get_copilot_service(settings: SettingsDependency) -> CopilotService:
    return create_copilot_service(settings)


CopilotServiceDependency = Annotated[CopilotService, Depends(get_copilot_service)]


def _build_investigation(
    request: RiskInvestigationRequest,
    settings: Settings,
) -> tuple[InvestigationContext, str]:
    bundle = load_active_bundle(settings.model_artifact_root)
    if request.transaction_reference is None:
        return investigate_risk(bundle, request.manual_transaction())

    provider = SQLitePaySimHistoryProvider(settings.behavioral_history_db)
    historical_transaction, behavioral_context = provider.context_for(
        request.transaction_reference
    )
    relationship_context = SQLiteRelationshipHistoryProvider(
        settings.relationship_history_db
    ).context_for(request.transaction_reference)
    return investigate_risk(
        bundle,
        historical_transaction.scoring_request(),
        behavioral_context=behavioral_context,
        relationship_context=relationship_context,
    )


@router.get("/model/status", response_model=ModelStatusResponse)
def model_status(settings: SettingsDependency) -> ModelStatusResponse:
    try:
        bundle = load_active_bundle(settings.model_artifact_root)
    except ModelUnavailableError as error:
        return ModelStatusResponse(status="unavailable", message=str(error))
    return ModelStatusResponse(
        status="ready",
        model_version=bundle.metadata["model_version"],
        model_family=bundle.metadata["model_family"],
        operating_mode=bundle.threshold_policy["recommended_mode"],
        dataset_sha256=bundle.metadata["dataset"]["sha256"],
        reference_profile_version=bundle.reference_profile["reference_profile_version"],
        message="Frozen Phase 2A model and Phase 2B evidence profile are available.",
    )


@router.post("/risk/predict", response_model=RiskPredictionResponse)
def risk_prediction(
    request: RiskPredictionRequest,
    settings: SettingsDependency,
) -> RiskPredictionResponse:
    try:
        bundle = load_active_bundle(settings.model_artifact_root)
        return predict_risk(bundle, request)
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/risk/investigate", response_model=InvestigationContext)
def risk_investigation(
    request: RiskInvestigationRequest,
    settings: SettingsDependency,
) -> InvestigationContext:
    try:
        context, _ = _build_investigation(request, settings)
        return context
    except TransactionReferenceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except BehaviorHistoryUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RelationshipTransactionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RelationshipHistoryUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post(
    "/risk/investigate/copilot",
    response_model=CopilotInvestigationResponse,
)
def risk_copilot_investigation(
    request: RiskInvestigationRequest,
    settings: SettingsDependency,
    copilot: CopilotServiceDependency,
) -> CopilotInvestigationResponse:
    try:
        context, action = _build_investigation(request, settings)
        sanitized = build_sanitized_context(context, action)
        return copilot.investigate(sanitized)
    except TransactionReferenceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except BehaviorHistoryUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RelationshipTransactionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RelationshipHistoryUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/model/evaluation", response_model=dict[str, Any])
def model_evaluation(settings: SettingsDependency) -> dict[str, Any]:
    try:
        return load_active_bundle(settings.model_artifact_root).evaluation
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
