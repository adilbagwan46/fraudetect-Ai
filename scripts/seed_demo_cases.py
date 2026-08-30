from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.case import CaseCreateRequest, CaseUpdateRequest
from backend.app.schemas.risk import RiskPredictionRequest
from backend.app.services.case_service import (
    SQLiteCaseRepository,
    assign_case_priority,
    evidence_summary,
    investigation_limitations,
)
from backend.app.services.copilot.context_builder import build_sanitized_context
from backend.app.services.copilot.service import create_copilot_service
from backend.app.services.investigation_service import build_investigation
from backend.app.services.risk_service import derived_features, load_active_bundle
from ml.fraudetect_ml.data.contracts import ML_FEATURE_COLUMNS

DEMO_ROOT = Path("artifacts/demo")


@dataclass(frozen=True)
class ShowcaseCandidate:
    """Internal selection record; its reference never enters public output."""

    transaction_reference: str
    transaction_type: str
    amount: float
    origin_balance_before: float
    step: int
    prior_behavior_count: int
    origin_network_count: int
    destination_network_count: int

    def scoring_request(self) -> RiskPredictionRequest:
        return RiskPredictionRequest(
            transaction_type=self.transaction_type,
            amount=self.amount,
            origin_balance_before=self.origin_balance_before,
            hour_of_day=self.step % 24,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate three genuine-PaySim deterministic showcase cases."
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


def _ensure_targets_available(paths: tuple[Path, ...], *, force: bool) -> None:
    if force:
        return
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise FileExistsError(
            f"Generated demo artifact already exists: {existing}; use --force"
        )


def _readonly_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Required generated history index is unavailable: {path}")
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def _candidate_from_row(row: sqlite3.Row | tuple) -> ShowcaseCandidate:
    return ShowcaseCandidate(
        transaction_reference=str(row[0]),
        transaction_type=str(row[1]),
        amount=float(row[2]),
        origin_balance_before=float(row[3]),
        step=int(row[4]),
        prior_behavior_count=int(row[5]),
        origin_network_count=int(row[6]),
        destination_network_count=int(row[7]),
    )


def _rich_candidates(settings: Settings) -> list[ShowcaseCandidate]:
    with _readonly_connection(settings.relationship_history_db) as connection:
        connection.execute(
            f'ATTACH DATABASE "file:{settings.behavioral_history_db.resolve()}?mode=ro" '
            "AS behavioral"
        )
        rows = connection.execute(
            """
            WITH origin_ranked AS (
                SELECT transaction_reference, origin_key, destination_key, step,
                       MIN(step) OVER (PARTITION BY origin_key) AS first_origin_step
                FROM relationship_transactions
            ), candidates AS (
                SELECT * FROM origin_ranked WHERE step > first_origin_step
            )
            SELECT current.transaction_reference, txn.transaction_type,
                   txn.amount, txn.origin_balance_before,
                   txn.step,
                   (SELECT COUNT(*) FROM relationship_transactions AS prior
                    WHERE prior.origin_key = current.origin_key
                      AND prior.step < current.step) AS prior_behavior_count,
                   (SELECT COUNT(DISTINCT prior.destination_key)
                    FROM relationship_transactions AS prior
                    WHERE prior.origin_key = current.origin_key
                      AND prior.step < current.step) AS origin_network_count,
                   (SELECT COUNT(DISTINCT prior.origin_key)
                    FROM relationship_transactions AS prior
                    WHERE prior.destination_key = current.destination_key
                      AND prior.step < current.step) AS destination_network_count
            FROM candidates AS current
            JOIN behavioral.transactions AS txn USING (transaction_reference)
            WHERE EXISTS (
                SELECT 1 FROM relationship_transactions AS prior
                WHERE prior.destination_key = current.destination_key
                  AND prior.step < current.step
            )
            ORDER BY current.step, current.transaction_reference
            """
        ).fetchall()
    return [_candidate_from_row(row) for row in rows]


def _limited_candidates(settings: Settings) -> list[ShowcaseCandidate]:
    with _readonly_connection(settings.relationship_history_db) as connection:
        connection.execute(
            f'ATTACH DATABASE "file:{settings.behavioral_history_db.resolve()}?mode=ro" '
            "AS behavioral"
        )
        rows = connection.execute(
            """
            SELECT txn.transaction_reference, txn.transaction_type,
                   txn.amount, txn.origin_balance_before,
                   txn.step, 0, 0, 0
            FROM behavioral.transactions AS txn
            JOIN relationship_transactions AS relationship USING (transaction_reference)
            WHERE txn.step = (SELECT MIN(step) FROM behavioral.transactions)
            ORDER BY txn.transaction_reference
            """
        ).fetchall()
    return [_candidate_from_row(row) for row in rows]


def _probabilities(
    candidates: list[ShowcaseCandidate], settings: Settings
) -> list[tuple[ShowcaseCandidate, float]]:
    if not candidates:
        return []
    bundle = load_active_bundle(settings.model_artifact_root)
    rows = []
    for candidate in candidates:
        request = candidate.scoring_request()
        derived = derived_features(request)
        rows.append(
            {
                "transaction_type": request.transaction_type,
                "amount": request.amount,
                "origin_balance_before": request.origin_balance_before,
                "hour_of_day": request.hour_of_day,
                "log_amount": derived.log_amount,
                "amount_to_origin_balance": derived.amount_to_origin_balance,
            }
        )
    frame = pd.DataFrame(rows, columns=ML_FEATURE_COLUMNS)
    values = bundle.model.predict_proba(frame)[:, 1]
    return [
        (candidate, float(probability))
        for candidate, probability in zip(candidates, values, strict=True)
    ]


def _select_scenarios(settings: Settings) -> dict[str, ShowcaseCandidate]:
    rich_scored = _probabilities(_rich_candidates(settings), settings)
    if len(rich_scored) < 2:
        raise RuntimeError(
            "Genuine PaySim indexes do not contain enough rich-context showcase candidates"
        )

    bundle = load_active_bundle(settings.model_artifact_root)
    policies = bundle.threshold_policy["policies_selected_on_validation"]
    threshold = float(policies[bundle.threshold_policy["recommended_mode"]]["threshold"])
    high_threshold = max(threshold, float(policies["HIGH_PRECISION"]["threshold"]))
    strong = sorted(
        rich_scored,
        key=lambda item: (
            -item[1],
            -item[0].prior_behavior_count,
            -item[0].destination_network_count,
            item[0].transaction_reference,
        ),
    )[0][0]

    lower_pool = [
        item
        for item in rich_scored
        if item[0].transaction_reference != strong.transaction_reference
        and item[1] < threshold
    ]
    if not lower_pool:
        lower_pool = [
            item
            for item in rich_scored
            if item[0].transaction_reference != strong.transaction_reference
        ]
    lower = sorted(
        lower_pool,
        key=lambda item: (
            -item[0].prior_behavior_count,
            -item[0].destination_network_count,
            item[1],
            item[0].transaction_reference,
        ),
    )[0][0]

    limited_scored = _probabilities(_limited_candidates(settings), settings)
    if not limited_scored:
        raise RuntimeError("Genuine PaySim indexes do not contain a limited-context candidate")
    medium_pool = [
        item for item in limited_scored if threshold <= item[1] < high_threshold
    ]
    limited = sorted(
        medium_pool or limited_scored,
        key=lambda item: (-item[1], item[0].transaction_reference),
    )[0][0]
    return {
        "strong-investigation": strong,
        "limited-context": limited,
        "lower-risk-resolution": lower,
    }


def _build_behavior_database(
    path: Path,
    source: Path,
    selected: list[ShowcaseCandidate],
    *,
    force: bool,
) -> None:
    _prepare_target(path, force=force)
    with _readonly_connection(source) as source_connection, sqlite3.connect(path) as output:
        output.execute(
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
        for candidate in selected:
            current = source_connection.execute(
                """
                SELECT transaction_reference, step, transaction_type, amount,
                       origin_balance_before, origin_key
                FROM transactions WHERE transaction_reference = ?
                """,
                (candidate.transaction_reference,),
            ).fetchone()
            rows = source_connection.execute(
                """
                SELECT transaction_reference, step, transaction_type, amount,
                       origin_balance_before, origin_key
                FROM transactions
                WHERE origin_key = ? AND step < ?
                ORDER BY step, transaction_reference
                """,
                (current[5], current[1]),
            ).fetchall()
            output.executemany(
                "INSERT OR IGNORE INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
                [*rows, current],
            )
        output.execute(
            "CREATE INDEX transactions_origin_step_idx "
            "ON transactions (origin_key, step, transaction_reference)"
        )
        output.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        output.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            {
                "schema_version": "1",
                "source": "genuine_prepared_paysim_subset",
                "causal_boundary": "historical.step < current.step",
                "labels_stored": "false",
            }.items(),
        )


def _build_relationship_database(
    path: Path,
    source: Path,
    selected: list[ShowcaseCandidate],
    *,
    force: bool,
) -> None:
    _prepare_target(path, force=force)
    with _readonly_connection(source) as source_connection, sqlite3.connect(path) as output:
        output.execute(
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
        for candidate in selected:
            current = source_connection.execute(
                """
                SELECT transaction_reference, step, amount, origin_key, destination_key
                FROM relationship_transactions WHERE transaction_reference = ?
                """,
                (candidate.transaction_reference,),
            ).fetchone()
            rows = source_connection.execute(
                """
                SELECT transaction_reference, step, amount, origin_key, destination_key
                FROM relationship_transactions
                WHERE (origin_key = ? OR destination_key = ?) AND step < ?
                ORDER BY step, transaction_reference
                """,
                (current[3], current[4], current[1]),
            ).fetchall()
            output.executemany(
                "INSERT OR IGNORE INTO relationship_transactions VALUES (?, ?, ?, ?, ?)",
                [*rows, current],
            )
        output.execute(
            "CREATE INDEX relationship_pair_step_idx ON relationship_transactions "
            "(origin_key, destination_key, step, transaction_reference)"
        )
        output.execute(
            "CREATE INDEX relationship_origin_step_idx ON relationship_transactions "
            "(origin_key, step, destination_key)"
        )
        output.execute(
            "CREATE INDEX relationship_destination_step_idx ON relationship_transactions "
            "(destination_key, step, origin_key)"
        )
        output.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        output.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            {
                "schema_version": "1",
                "source": "genuine_prepared_paysim_subset",
                "causal_boundary": "historical.step < current.step",
                "labels_loaded": "false",
                "labels_stored": "false",
            }.items(),
        )


def seed_demo(
    *,
    demo_root: Path = DEMO_ROOT,
    force: bool = False,
    base_settings: Settings | None = None,
) -> dict:
    """Build a repeatable isolated showcase from genuine prepared PaySim indexes."""

    source_settings = base_settings or get_settings()
    behavior_database = demo_root / "behavior.sqlite"
    relationship_database = demo_root / "relationship.sqlite"
    case_database = demo_root / "cases.sqlite"
    _ensure_targets_available(
        (behavior_database, relationship_database, case_database), force=force
    )
    scenarios = _select_scenarios(source_settings)
    selected = list(scenarios.values())
    _build_behavior_database(
        behavior_database,
        source_settings.behavioral_history_db,
        selected,
        force=force,
    )
    _build_relationship_database(
        relationship_database,
        source_settings.relationship_history_db,
        selected,
        force=force,
    )
    _prepare_target(case_database, force=force)

    settings = replace(
        source_settings,
        behavioral_history_db=behavior_database,
        relationship_history_db=relationship_database,
        case_database=case_database,
        llm_enabled=False,
        llm_api_key=None,
        gemini_api_key=None,
    )
    repository = SQLiteCaseRepository(settings.case_database)
    created = []
    for name, candidate in scenarios.items():
        request = CaseCreateRequest(
            transaction_reference=candidate.transaction_reference
        )
        context, action = build_investigation(request, settings)
        snapshot = build_sanitized_context(context, action)
        detail = repository.create(
            source_type="REFERENCE",
            transaction_reference_available=True,
            model_version=context.model_output.model_version,
            snapshot=snapshot,
            priority=assign_case_priority(snapshot),
            evidence_summary=evidence_summary(snapshot),
            limitations=investigation_limitations(snapshot),
        )
        if name == "strong-investigation":
            report = create_copilot_service(settings).investigate(snapshot)
            detail = repository.save_copilot(detail.case.case_id, report)
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(
                    status="IN_REVIEW",
                    analyst_note="Grounded evidence and historical context queued for review.",
                ),
            )
        elif name == "lower-risk-resolution":
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(status="IN_REVIEW"),
            )
            detail = repository.update(
                detail.case.case_id,
                CaseUpdateRequest(
                    status="CLEARED",
                    analyst_note="Available evidence and historical context reviewed.",
                ),
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
                "behavioral_prior_transaction_count": (
                    detail.intelligence_snapshot.behavioral_context.prior_transaction_count
                ),
                "relationship_prior_interaction_count": (
                    detail.intelligence_snapshot.relationship_context.prior_interaction_count
                ),
                "origin_network_count": (
                    detail.intelligence_snapshot.relationship_context.origin_network
                    .prior_unique_counterparty_count
                ),
                "destination_network_count": (
                    detail.intelligence_snapshot.relationship_context.destination_network
                    .prior_unique_origin_count
                ),
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
        "disclosure": (
            "Genuine prepared PaySim rows selected deterministically; references and raw "
            "identities remain internal to ignored local indexes."
        ),
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(seed_demo(demo_root=args.output_dir, force=args.force), indent=2))


if __name__ == "__main__":
    main()
