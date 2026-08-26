from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.system import DatasetStatusResponse, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service="fraudetect-api", version="0.1.0")


@router.get("/dataset/status", response_model=DatasetStatusResponse)
def dataset_status(settings: Annotated[Settings, Depends(get_settings)]) -> DatasetStatusResponse:
    manifest_path = settings.dataset_manifest
    if not manifest_path.is_file():
        return DatasetStatusResponse(
            status="not_prepared",
            manifest_path=str(manifest_path),
            message="Run the data preparation command before model training.",
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DatasetStatusResponse(
            status="not_prepared",
            manifest_path=str(manifest_path),
            message="The dataset manifest is unreadable or invalid.",
        )

    return DatasetStatusResponse(
        status="ready",
        manifest_path=str(manifest_path),
        source_kind=manifest.get("source", {}).get("kind"),
        rows=manifest.get("dataset", {}).get("rows"),
        generated_at=manifest.get("generated_at"),
        message="Prepared dataset splits are available.",
    )
