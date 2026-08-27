from __future__ import annotations

import json
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.case import OperationalMetricsResponse
from backend.app.schemas.system import (
    DatasetStatusResponse,
    HealthResponse,
    ReadinessComponent,
    SystemReadinessResponse,
)
from backend.app.services.case_service import CaseStoreUnavailableError, SQLiteCaseRepository
from backend.app.services.risk_service import ModelUnavailableError, load_active_bundle

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


def _sqlite_component(
    *, key: str, label: str, database_path, expected_table: str
) -> ReadinessComponent:
    if not database_path.is_file():
        return ReadinessComponent(
            key=key,
            label=label,
            status="unavailable",
            message="Generated local index is not available.",
        )
    try:
        with sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (expected_table,),
            ).fetchone()
    except sqlite3.Error:
        table = None
    return ReadinessComponent(
        key=key,
        label=label,
        status="ready" if table else "unavailable",
        message=(
            "Generated local index is readable."
            if table
            else "Generated local index is missing its expected schema."
        ),
    )


@router.get("/system/readiness", response_model=SystemReadinessResponse)
def system_readiness(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SystemReadinessResponse:
    """Report safe component readiness without exposing paths, secrets, or identifiers."""

    try:
        bundle = load_active_bundle(settings.model_artifact_root)
    except ModelUnavailableError:
        bundle = None

    model_version = bundle.metadata.get("model_version") if bundle else None
    profile_version = (
        bundle.reference_profile.get("reference_profile_version")
        if bundle and bundle.reference_profile
        else None
    )
    model_ready = bundle is not None
    profile_ready = profile_version is not None
    components = [
        ReadinessComponent(
            key="ml_model",
            label="ML Model",
            status="ready" if model_ready else "unavailable",
            version=model_version,
            mode=(bundle.threshold_policy.get("recommended_mode") if bundle else None),
            message=(
                "Frozen fraud-risk model is loaded."
                if model_ready
                else "Frozen fraud-risk model is unavailable."
            ),
        ),
        ReadinessComponent(
            key="deterministic_evidence",
            label="Deterministic Evidence",
            status="ready" if profile_ready else "unavailable",
            version=profile_version,
            mode="deterministic",
            message=(
                "Evidence engine and approved reference statistics are available."
                if profile_ready
                else "Evidence engine requires the approved reference profile."
            ),
        ),
        ReadinessComponent(
            key="reference_profile",
            label="Reference Profile",
            status="ready" if profile_ready else "unavailable",
            version=profile_version,
            message=(
                "Training-only reference profile is loaded."
                if profile_ready
                else "Training-only reference profile is unavailable."
            ),
        ),
        _sqlite_component(
            key="behavioral_history",
            label="Behavioral History",
            database_path=settings.behavioral_history_db,
            expected_table="transactions",
        ),
        _sqlite_component(
            key="relationship_history",
            label="Relationship History",
            database_path=settings.relationship_history_db,
            expected_table="relationship_transactions",
        ),
        _sqlite_component(
            key="case_store",
            label="Case Store",
            database_path=settings.case_database,
            expected_table="cases",
        ),
    ]

    provider_configured = settings.llm_provider == "openai" and bool(settings.llm_api_key)
    if settings.llm_enabled and provider_configured:
        copilot_mode = "real_llm_configured"
        copilot_message = (
            "LLM provider is configured; external availability was not checked and "
            "deterministic fallback remains available."
        )
    elif settings.llm_enabled:
        copilot_mode = "deterministic_fallback"
        copilot_message = "LLM provider is unavailable; deterministic fallback is active."
    else:
        copilot_mode = "deterministic_fallback"
        copilot_message = "LLM is disabled; deterministic fallback is active."
    components.append(
        ReadinessComponent(
            key="llm_copilot",
            label="LLM Copilot",
            status="ready",
            version=settings.llm_model if copilot_mode == "real_llm_configured" else None,
            mode=copilot_mode,
            fallback_available=True,
            provider_enabled=settings.llm_enabled,
            provider_configured=provider_configured,
            external_availability=(
                "not_checked"
                if settings.llm_enabled and provider_configured
                else "not_applicable"
            ),
            message=copilot_message,
        )
    )
    overall = "ready" if all(item.status == "ready" for item in components) else "degraded"
    return SystemReadinessResponse(status=overall, components=components)


@router.get("/system/metrics", response_model=OperationalMetricsResponse)
def system_metrics(
    settings: Annotated[Settings, Depends(get_settings)],
) -> OperationalMetricsResponse:
    """Return aggregate workflow metrics without case content or identifiers."""

    try:
        return SQLiteCaseRepository(settings.case_database).metrics()
    except CaseStoreUnavailableError as error:
        raise HTTPException(
            status_code=503, detail="Operational metrics are unavailable."
        ) from error
