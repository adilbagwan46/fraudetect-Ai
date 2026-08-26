from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.risk import (
    ModelStatusResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
)
from backend.app.services.risk_service import (
    ModelUnavailableError,
    load_active_bundle,
    predict_risk,
)

router = APIRouter(tags=["model"])
SettingsDependency = Annotated[Settings, Depends(get_settings)]


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
        message="Frozen Phase 2A model bundle is available.",
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


@router.get("/model/evaluation", response_model=dict[str, Any])
def model_evaluation(settings: SettingsDependency) -> dict[str, Any]:
    try:
        return load_active_bundle(settings.model_artifact_root).evaluation
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
