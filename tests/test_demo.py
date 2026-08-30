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


def _create_source_indexes(root: Path) -> tuple[Path, Path]:
    behavior = root / "source-behavior.sqlite"
    relationship = root / "source-relationship.sqlite"
    behavior_rows = [
        ("TX-000000001", 1, "PAYMENT", 10.0, 1_000.0, "origin-a"),
        ("TX-000000002", 1, "PAYMENT", 15.0, 900.0, "origin-network-a"),
        ("TX-000000003", 3, "TRANSFER", 800.0, 1_000.0, "origin-a"),
        ("TX-000000004", 1, "PAYMENT", 20.0, 2_000.0, "origin-b"),
        ("TX-000000005", 2, "PAYMENT", 30.0, 1_980.0, "origin-b"),
        ("TX-000000006", 1, "PAYMENT", 25.0, 800.0, "origin-network-b"),
        ("TX-000000007", 4, "CASH_OUT", 100.0, 1_950.0, "origin-b"),
        ("TX-000000008", 1, "PAYMENT", 5.0, 100.0, "origin-limited"),
    ]
    relationship_rows = [
        ("TX-000000001", 1, 10.0, "origin-a", "destination-old-a"),
        ("TX-000000002", 1, 15.0, "origin-network-a", "destination-strong"),
        ("TX-000000003", 3, 800.0, "origin-a", "destination-strong"),
        ("TX-000000004", 1, 20.0, "origin-b", "destination-old-b-1"),
        ("TX-000000005", 2, 30.0, "origin-b", "destination-old-b-2"),
        ("TX-000000006", 1, 25.0, "origin-network-b", "destination-resolution"),
        ("TX-000000007", 4, 100.0, "origin-b", "destination-resolution"),
        ("TX-000000008", 1, 5.0, "origin-limited", "destination-limited"),
    ]
    with sqlite3.connect(behavior) as connection:
        connection.execute(
            """
            CREATE TABLE transactions (
                transaction_reference TEXT PRIMARY KEY, step INTEGER NOT NULL,
                transaction_type TEXT NOT NULL, amount REAL NOT NULL,
                origin_balance_before REAL NOT NULL, origin_key TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)", behavior_rows)
        connection.execute(
            "CREATE INDEX transactions_origin_step_idx "
            "ON transactions (origin_key, step, transaction_reference)"
        )
    with sqlite3.connect(relationship) as connection:
        connection.execute(
            """
            CREATE TABLE relationship_transactions (
                transaction_reference TEXT PRIMARY KEY, step INTEGER NOT NULL,
                amount REAL NOT NULL, origin_key TEXT NOT NULL,
                destination_key TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.executemany(
            "INSERT INTO relationship_transactions VALUES (?, ?, ?, ?, ?)",
            relationship_rows,
        )
        connection.execute(
            "CREATE INDEX relationship_pair_step_idx ON relationship_transactions "
            "(origin_key, destination_key, step, transaction_reference)"
        )
        connection.execute(
            "CREATE INDEX relationship_origin_step_idx ON relationship_transactions "
            "(origin_key, step, destination_key)"
        )
        connection.execute(
            "CREATE INDEX relationship_destination_step_idx ON relationship_transactions "
            "(destination_key, step, origin_key)"
        )
    return behavior, relationship


def test_demo_seeding_is_reproducible_and_isolated(tmp_path: Path) -> None:
    artifact_root = tmp_path / "models"
    create_test_bundle(artifact_root)
    behavior, relationship = _create_source_indexes(tmp_path)
    settings = Settings(
        model_artifact_root=artifact_root,
        behavioral_history_db=behavior,
        relationship_history_db=relationship,
        case_database=tmp_path / "default-cases.sqlite",
    )

    first = seed_demo(
        demo_root=tmp_path / "demo-a", base_settings=settings
    )
    second = seed_demo(
        demo_root=tmp_path / "demo-b", base_settings=settings
    )

    assert _stable_scenarios(first) == _stable_scenarios(second)
    assert [item["scenario"] for item in first["showcase_cases"]] == [
        "strong-investigation",
        "limited-context",
        "lower-risk-resolution",
    ]
    strong = next(
        item
        for item in first["showcase_cases"]
        if item["scenario"] == "strong-investigation"
    )
    assert strong["status"] == "IN_REVIEW"
    assert strong["copilot_mode"] == "deterministic_fallback"
    assert strong["behavioral_prior_transaction_count"] > 0
    assert strong["origin_network_count"] > 0
    assert strong["destination_network_count"] > 0
    limited = next(
        item for item in first["showcase_cases"] if item["scenario"] == "limited-context"
    )
    assert limited["behavioral_prior_transaction_count"] == 0
    assert limited["origin_network_count"] == 0
    assert limited["destination_network_count"] == 0
    resolution = next(
        item
        for item in first["showcase_cases"]
        if item["scenario"] == "lower-risk-resolution"
    )
    assert resolution["status"] == "CLEARED"
    assert resolution["behavioral_prior_transaction_count"] > 0
    assert resolution["destination_network_count"] > 0
    with sqlite3.connect(tmp_path / "demo-a" / "cases.sqlite") as connection:
        metrics = dict(
            connection.execute("SELECT status, COUNT(*) FROM cases GROUP BY status")
        )
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM case_audit_events"
        ).fetchone()[0]
    assert metrics == {"CLEARED": 1, "IN_REVIEW": 1, "OPEN": 1}
    assert audit_count > len(first["showcase_cases"])
    assert not settings.case_database.exists()


def test_demo_seeding_requires_explicit_force_before_replacement(tmp_path: Path) -> None:
    artifact_root = tmp_path / "models"
    create_test_bundle(artifact_root)
    behavior, relationship = _create_source_indexes(tmp_path)
    settings = Settings(
        model_artifact_root=artifact_root,
        behavioral_history_db=behavior,
        relationship_history_db=relationship,
    )
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
