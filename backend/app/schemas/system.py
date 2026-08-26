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

