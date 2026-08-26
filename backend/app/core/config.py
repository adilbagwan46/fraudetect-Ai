from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "Fraudetect AI"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    dataset_manifest: Path = Path("data/processed/manifest.json")
    model_artifact_root: Path = Path("artifacts/models")
    behavioral_history_db: Path = Path("artifacts/behavioral/history.sqlite")
    relationship_history_db: Path = Path("artifacts/relationship/history.sqlite")
    llm_enabled: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-5.6"
    llm_api_key: str | None = field(default=None, repr=False)
    llm_timeout_seconds: float = 20.0
    enrichment_seed: str = "fraudetect-demo-v1"
    enrichment_device_buckets: int = 10_000
    enrichment_ip_buckets: int = 5_000
    low_risk_max: float = 0.30
    high_risk_min: float = 0.70
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    def __post_init__(self) -> None:
        if not 0 <= self.low_risk_max < self.high_risk_min <= 1:
            raise ValueError("Risk thresholds must satisfy 0 <= low < high <= 1")
        if self.enrichment_device_buckets <= 0 or self.enrichment_ip_buckets <= 0:
            raise ValueError("Relationship enrichment bucket counts must be positive")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("LLM timeout must be positive")


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@lru_cache
def get_settings() -> Settings:
    return Settings(
        environment=os.getenv("FRAUDETECT_ENV", "development"),
        api_prefix=os.getenv("FRAUDETECT_API_PREFIX", "/api/v1"),
        dataset_manifest=Path(
            os.getenv("FRAUDETECT_DATASET_MANIFEST", "data/processed/manifest.json")
        ),
        model_artifact_root=Path(
            os.getenv("FRAUDETECT_MODEL_ARTIFACT_ROOT", "artifacts/models")
        ),
        behavioral_history_db=Path(
            os.getenv(
                "FRAUDETECT_BEHAVIORAL_HISTORY_DB",
                "artifacts/behavioral/history.sqlite",
            )
        ),
        relationship_history_db=Path(
            os.getenv(
                "FRAUDETECT_RELATIONSHIP_HISTORY_DB",
                "artifacts/relationship/history.sqlite",
            )
        ),
        llm_enabled=_environment_bool("FRAUDETECT_LLM_ENABLED", False),
        llm_provider=os.getenv("FRAUDETECT_LLM_PROVIDER", "openai").strip().lower(),
        llm_model=os.getenv("FRAUDETECT_LLM_MODEL", "gpt-5.6"),
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_timeout_seconds=float(os.getenv("FRAUDETECT_LLM_TIMEOUT_SECONDS", "20")),
        enrichment_seed=os.getenv("FRAUDETECT_ENRICHMENT_SEED", "fraudetect-demo-v1"),
        enrichment_device_buckets=int(os.getenv("FRAUDETECT_DEVICE_BUCKETS", "10000")),
        enrichment_ip_buckets=int(os.getenv("FRAUDETECT_IP_BUCKETS", "5000")),
        low_risk_max=float(os.getenv("FRAUDETECT_LOW_RISK_MAX", "0.30")),
        high_risk_min=float(os.getenv("FRAUDETECT_HIGH_RISK_MIN", "0.70")),
    )
