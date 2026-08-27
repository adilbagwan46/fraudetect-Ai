from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.case import CaseCreateRequest, CaseUpdateRequest
from backend.app.services.case_service import (
    SQLiteCaseRepository,
    assign_case_priority,
    evidence_summary,
    investigation_limitations,
)
from backend.app.services.copilot.context_builder import build_sanitized_context
from backend.app.services.copilot.service import create_copilot_service
from backend.app.services.investigation_service import build_investigation

DEMO_ROOT = Path("artifacts/demo")
BEHAVIOR_DATABASE = DEMO_ROOT / "behavior.sqlite"
RELATIONSHIP_DATABASE = DEMO_ROOT / "relationship.sqlite"
CASE_DATABASE = DEMO_ROOT / "cases.sqlite"

BEHAVIOR_ROWS = (
    ("DEMO-LOW", 10, "PAYMENT", 10.0, 1_000.0, "origin-low"),
    ("DEMO-HIGH", 20, "TRANSFER", 8_317_724.27, 8_317_724.27, "origin-high"),
    ("DEMO-DEVIATION-PRIOR-1", 1, "PAYMENT", 10.0, 2_000.0, "origin-deviation"),
    ("DEMO-DEVIATION-PRIOR-2", 2, "PAYMENT", 20.0, 1_990.0, "origin-deviation"),
    ("DEMO-DEVIATION", 8, "TRANSFER", 10_000.0, 20_000.0, "origin-deviation"),
    ("DEMO-NEW-PAIR-PRIOR-1", 2, "PAYMENT", 25.0, 2_000.0, "origin-network"),
    ("DEMO-NEW-PAIR-PRIOR-2", 3, "PAYMENT", 35.0, 1_975.0, "origin-network"),
    ("DEMO-NEW-PAIR", 9, "PAYMENT", 120.0, 1_940.0, "origin-network"),
)
RELATIONSHIP_ROWS = (
    ("DEMO-LOW", 10, 10.0, "origin-low", "destination-low"),
    ("DEMO-HIGH", 20, 8_317_724.27, "origin-high", "destination-high"),
    ("DEMO-DEVIATION-PRIOR-1", 1, 10.0, "origin-deviation", "destination-repeat"),
    ("DEMO-DEVIATION-PRIOR-2", 2, 20.0, "origin-deviation", "destination-repeat"),
    ("DEMO-DEVIATION", 8, 10_000.0, "origin-deviation", "destination-repeat"),
    ("DEMO-NEW-PAIR-PRIOR-1", 2, 25.0, "origin-network", "destination-old-a"),
    ("DEMO-NEW-PAIR-PRIOR-2", 3, 35.0, "origin-network", "destination-old-b"),
    ("DEMO-NEW-PAIR", 9, 120.0, "origin-network", "destination-new"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate five safe, deterministic Phase 6 showcase cases."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated demo databases under artifacts/demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEMO_ROOT,
        help="Ignored directory for generated demo databases (default: artifacts/demo).",
    )
    return parser.parse_args()


def _prepare_target(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Generated demo artifact already exists: {path}; use --force")
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_behavior_database(path: Path, *, force: bool) -> None:
    _prepare_target(path, force=force)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE transactions (
                transaction_reference TEXT PRIMARY KEY,
                step INTEGER NOT NULL,
                transaction_type TEXT NOT NULL,
                amount REAL NOT NULL,
                origin_balance_before REAL NOT NULL,
                origin_key TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)", BEHAVIOR_ROWS)
        connection.execute(
            "CREATE INDEX transactions_origin_step_idx "
            "ON transactions (origin_key, step, transaction_reference)"
        )


def _build_relationship_database(path: Path, *, force: bool) -> None:
    _prepare_target(path, force=force)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE relationship_transactions (
                transaction_reference TEXT PRIMARY KEY,
                step INTEGER NOT NULL,
                amount REAL NOT NULL,
                origin_key TEXT NOT NULL,
                destination_key TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.executemany(
            "INSERT INTO relationship_transactions VALUES (?, ?, ?, ?, ?)",
            RELATIONSHIP_ROWS,
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


def seed_demo(
    *,
    demo_root: Path = DEMO_ROOT,
    force: bool = False,
    base_settings: Settings | None = None,
) -> dict:
    """Build a repeatable isolated showcase without touching the configured case store."""

    behavior_database = demo_root / "behavior.sqlite"
    relationship_database = demo_root / "relationship.sqlite"
    case_database = demo_root / "cases.sqlite"
    _build_behavior_database(behavior_database, force=force)
    _build_relationship_database(relationship_database, force=force)
    _prepare_target(case_database, force=force)

    settings = replace(
        base_settings or get_settings(),
        behavioral_history_db=behavior_database,
        relationship_history_db=relationship_database,
        case_database=case_database,
        llm_enabled=False,
        llm_api_key=None,
    )
    repository = SQLiteCaseRepository(settings.case_database)
    scenarios = (
        ("low-risk", CaseCreateRequest(transaction_reference="DEMO-LOW")),
        ("high-risk", CaseCreateRequest(transaction_reference="DEMO-HIGH")),
        ("strong-behavioral-deviation", CaseCreateRequest(transaction_reference="DEMO-DEVIATION")),
        ("new-relationship", CaseCreateRequest(transaction_reference="DEMO-NEW-PAIR")),
        (
            "history-unavailable",
            CaseCreateRequest(
                transaction_type="PAYMENT",
                amount=75.0,
                origin_balance_before=1_000.0,
                hour_of_day=12,
            ),
        ),
    )
    created = []
    for name, request in scenarios:
        context, action = build_investigation(request, settings)
        snapshot = build_sanitized_context(context, action)
        detail = repository.create(
            source_type="REFERENCE" if request.transaction_reference else "MANUAL",
            transaction_reference_available=request.transaction_reference is not None,
            model_version=context.model_output.model_version,
            snapshot=snapshot,
            priority=assign_case_priority(snapshot),
            evidence_summary=evidence_summary(snapshot),
            limitations=investigation_limitations(snapshot),
        )
        if name == "high-risk":
            report = create_copilot_service(settings).investigate(snapshot)
            detail = repository.save_copilot(detail.case.case_id, report)
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(
                    status="IN_REVIEW",
                    analyst_note="Demo analyst reviewed the grounded evidence.",
                ),
            )
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(
                    status="ESCALATED",
                    analyst_note="Escalated for a simulated secondary review.",
                ),
            )
        elif name == "strong-behavioral-deviation":
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(
                    status="IN_REVIEW",
                    analyst_note="Behavioral deviation queued for analyst review.",
                ),
            )
        elif name == "new-relationship":
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(status="IN_REVIEW"),
            )
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(
                    status="CLEARED",
                    analyst_note="Synthetic relationship context reviewed and cleared.",
                ),
            )
        elif name == "history-unavailable":
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(status="IN_REVIEW"),
            )
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(status="CLEARED"),
            )
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(status="CLOSED"),
            )
        created.append(
            {
                "scenario": name,
                "case_id": detail.case.case_id,
                "case_priority": detail.case.priority,
                "ml_risk_level": detail.case.risk_level,
                "fraud_probability": detail.case.fraud_probability,
                "status": detail.case.status,
                "copilot_mode": detail.copilot.mode if detail.copilot else None,
            }
        )

    return {
        "generated_artifacts": {
            "cases": str(case_database),
            "behavior": str(behavior_database),
            "relationship": str(relationship_database),
        },
        "showcase_cases": created,
        "api_environment": {
            "FRAUDETECT_CASE_DATABASE": str(case_database),
            "FRAUDETECT_BEHAVIORAL_HISTORY_DB": str(behavior_database),
            "FRAUDETECT_RELATIONSHIP_HISTORY_DB": str(relationship_database),
        },
        "disclosure": "Generated synthetic showcase context; not PaySim evaluation data.",
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(seed_demo(demo_root=args.output_dir, force=args.force), indent=2))


if __name__ == "__main__":
    main()
