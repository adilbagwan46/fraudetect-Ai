import sqlite3
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from scripts.seed_demo_cases import seed_demo
from tests.test_risk_api import create_test_bundle


def _stable_scenarios(payload: dict) -> list[dict]:
    return [
        {key: value for key, value in scenario.items() if key != "case_id"}
        for scenario in payload["showcase_cases"]
    ]


def test_demo_seeding_is_reproducible_and_isolated(tmp_path: Path) -> None:
    artifact_root = tmp_path / "models"
    create_test_bundle(artifact_root)
    settings = Settings(model_artifact_root=artifact_root)

    first = seed_demo(
        demo_root=tmp_path / "demo-a", base_settings=settings
    )
    second = seed_demo(
        demo_root=tmp_path / "demo-b", base_settings=settings
    )

    assert _stable_scenarios(first) == _stable_scenarios(second)
    assert [item["scenario"] for item in first["showcase_cases"]] == [
        "low-risk",
        "high-risk",
        "strong-behavioral-deviation",
        "new-relationship",
        "history-unavailable",
    ]
    high = next(
        item for item in first["showcase_cases"] if item["scenario"] == "high-risk"
    )
    assert high["status"] == "ESCALATED"
    assert high["copilot_mode"] == "deterministic_fallback"
    assert {item["status"] for item in first["showcase_cases"]} == {
        "OPEN",
        "IN_REVIEW",
        "ESCALATED",
        "CLEARED",
        "CLOSED",
    }
    with sqlite3.connect(tmp_path / "demo-a" / "cases.sqlite") as connection:
        metrics = dict(
            connection.execute("SELECT status, COUNT(*) FROM cases GROUP BY status")
        )
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM case_audit_events"
        ).fetchone()[0]
    assert metrics == {
        "CLEARED": 1,
        "CLOSED": 1,
        "ESCALATED": 1,
        "IN_REVIEW": 1,
        "OPEN": 1,
    }
    assert audit_count > len(first["showcase_cases"])
    assert not settings.case_database.exists()


def test_demo_seeding_requires_explicit_force_before_replacement(tmp_path: Path) -> None:
    artifact_root = tmp_path / "models"
    create_test_bundle(artifact_root)
    settings = Settings(model_artifact_root=artifact_root)
    demo_root = tmp_path / "demo"
    seed_demo(demo_root=demo_root, base_settings=settings)
    with sqlite3.connect(demo_root / "cases.sqlite") as connection:
        original_count = connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]

    with pytest.raises(FileExistsError, match="use --force"):
        seed_demo(demo_root=demo_root, base_settings=settings)

    with sqlite3.connect(demo_root / "cases.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == original_count

    replaced = seed_demo(demo_root=demo_root, force=True, base_settings=settings)
    assert len(replaced["showcase_cases"]) == original_count
