from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dataset_status_handles_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FRAUDETECT_DATASET_MANIFEST", str(tmp_path / "missing.json"))
    get_settings.cache_clear()
    try:
        response = TestClient(app).get("/api/v1/dataset/status")
        assert response.status_code == 200
        assert response.json()["status"] == "not_prepared"
    finally:
        get_settings.cache_clear()

