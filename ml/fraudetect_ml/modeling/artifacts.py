from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def environment_metadata() -> dict[str, str]:
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "fastapi")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{package: version(package) for package in packages},
    }


def save_model_bundle(
    artifact_root: Path,
    model_version: str,
    *,
    model: Any,
    files: dict[str, Any],
) -> Path:
    model_dir = artifact_root / model_version
    model_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, model_dir / "model.joblib", compress=3)
    for filename, payload in files.items():
        write_json(model_dir / filename, payload)
    write_json(
        artifact_root / "latest.json",
        {"model_version": model_version, "path": str(model_dir)},
    )
    return model_dir

