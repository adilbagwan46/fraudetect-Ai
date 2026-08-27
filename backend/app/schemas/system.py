from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class DatasetStatusResponse(BaseModel):
    status: Literal["ready", "not_prepared"]
    manifest_path: str
    source_kind: str | None = None
    rows: int | None = Field(default=None, ge=0)
    generated_at: str | None = None
    message: str


ComponentStatus = Literal["ready", "unavailable"]


class ReadinessComponent(BaseModel):
    key: Literal[
        "ml_model",
        "deterministic_evidence",
        "reference_profile",
        "behavioral_history",
        "relationship_history",
        "case_store",
        "llm_copilot",
    ]
    label: str
    status: ComponentStatus
    version: str | None = None
    mode: str | None = None
    fallback_available: bool = False
    provider_enabled: bool | None = None
    provider_configured: bool | None = None
    external_availability: Literal["not_checked", "not_applicable"] | None = None
    message: str


class SystemReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    components: list[ReadinessComponent]
    advisory_only: Literal[True] = True
