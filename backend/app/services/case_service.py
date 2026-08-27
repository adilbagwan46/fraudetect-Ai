from __future__ import annotations

import json
import secrets
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from backend.app.schemas.case import (
    AnalystDecision,
    AnalystDisposition,
    CaseDetailResponse,
    CaseListResponse,
    CasePriority,
    CaseSourceType,
    CaseStatus,
    CaseStatusHistoryItem,
    CaseSummary,
    CaseUpdateRequest,
    EvidenceSummary,
)
from backend.app.schemas.copilot import (
    CopilotInvestigationResponse,
    SanitizedInvestigationContext,
)
from backend.app.schemas.risk import EvidenceSeverity, RiskLevel

SEVERITY_ORDER: tuple[EvidenceSeverity, ...] = (
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
    "INFO",
)
VALID_TRANSITIONS: dict[CaseStatus, frozenset[CaseStatus]] = {
    "OPEN": frozenset({"IN_REVIEW"}),
    "IN_REVIEW": frozenset({"ESCALATED", "CLEARED"}),
    "ESCALATED": frozenset({"CLOSED"}),
    "CLEARED": frozenset({"CLOSED"}),
    "CLOSED": frozenset(),
}
STRONG_DEVIATION_IDS = frozenset(
    {
        "behavior_amount_above_typical",
        "relationship_amount_deviation",
    }
)


class CaseStoreUnavailableError(RuntimeError):
    """Raised when the local case workflow database cannot be used."""


class CaseNotFoundError(LookupError):
    """Raised when an application-level case ID cannot be resolved."""


class InvalidCaseTransitionError(ValueError):
    """Raised when a requested workflow transition violates the state machine."""


class CaseRepository(Protocol):
    def create(
        self,
        *,
        source_type: CaseSourceType,
        transaction_reference_available: bool,
        model_version: str,
        snapshot: SanitizedInvestigationContext,
        priority: CasePriority,
        evidence_summary: EvidenceSummary,
        limitations: list[str],
    ) -> CaseDetailResponse: ...

    def get(self, case_id: str) -> CaseDetailResponse: ...

    def list(
        self,
        *,
        status: CaseStatus | None,
        risk_level: RiskLevel | None,
        priority: CasePriority | None,
        disposition: AnalystDisposition | None,
        limit: int,
        offset: int,
    ) -> CaseListResponse: ...

    def update(self, case_id: str, request: CaseUpdateRequest) -> CaseDetailResponse: ...

    def save_copilot(
        self, case_id: str, response: CopilotInvestigationResponse
    ) -> CaseDetailResponse: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def evidence_summary(context: SanitizedInvestigationContext) -> EvidenceSummary:
    severities = Counter(item.severity for item in context.evidence)
    categories = Counter(item.category for item in context.evidence)
    highest = next((value for value in SEVERITY_ORDER if severities[value]), None)
    return EvidenceSummary(
        total_count=len(context.evidence),
        highest_severity=highest,
        severity_counts={severity: severities[severity] for severity in SEVERITY_ORDER},
        category_counts=dict(sorted(categories.items())),
    )


def assign_case_priority(context: SanitizedInvestigationContext) -> CasePriority:
    """Deterministic workflow priority; never changes or replaces ML risk."""

    risk_level = context.model_output.risk_level
    evidence_ids = {item.evidence_id for item in context.evidence}
    severities = {item.severity for item in context.evidence}
    if risk_level == "HIGH":
        return "CRITICAL"
    if risk_level == "MEDIUM" or "CRITICAL" in severities:
        return "HIGH"
    if evidence_ids & STRONG_DEVIATION_IDS:
        return "HIGH"
    if severities & {"HIGH", "MEDIUM"}:
        return "MEDIUM"
    return "LOW"


def investigation_limitations(
    context: SanitizedInvestigationContext,
) -> list[str]:
    limitations = [
        "PaySim is synthetic; this case is a demonstration and not a production fraud decision."
    ]
    behavior = context.behavioral_context
    if not behavior.history_available:
        limitations.append("No prior behavioral history is available.")
    elif behavior.prior_transaction_count <= 2:
        limitations.append("The behavioral baseline is limited.")
    relationship = context.relationship_context
    if not relationship.context_available:
        limitations.append("Relationship context is unavailable for manual input.")
    elif not relationship.history_available:
        limitations.append("No prior origin-destination relationship was observed.")
    elif relationship.baseline_is_limited:
        limitations.append("The relationship baseline is limited.")
    return limitations


class SQLiteCaseRepository:
    """Small local workflow store containing only identifier-free case snapshots."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cases (
                        case_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        transaction_reference_available INTEGER NOT NULL,
                        model_version TEXT NOT NULL,
                        fraud_probability REAL NOT NULL,
                        risk_level TEXT NOT NULL,
                        operating_mode TEXT NOT NULL,
                        analyst_disposition TEXT NOT NULL,
                        analyst_note TEXT,
                        disposition_at TEXT,
                        evidence_summary_json TEXT NOT NULL,
                        snapshot_json TEXT NOT NULL,
                        limitations_json TEXT NOT NULL,
                        copilot_json TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS case_status_history (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_id TEXT NOT NULL REFERENCES cases(case_id),
                        occurred_at TEXT NOT NULL,
                        previous_status TEXT,
                        new_status TEXT NOT NULL,
                        disposition TEXT NOT NULL,
                        note_recorded INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS cases_queue_idx ON cases "
                    "(status, priority, risk_level, created_at)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS case_history_idx ON case_status_history "
                    "(case_id, sequence)"
                )
        except sqlite3.Error as error:
            raise CaseStoreUnavailableError("Case store could not be initialized") from error

    @staticmethod
    def _summary(row: sqlite3.Row) -> CaseSummary:
        return CaseSummary(
            case_id=str(row["case_id"]),
            status=str(row["status"]),
            priority=str(row["priority"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            source_type=str(row["source_type"]),
            transaction_reference_available=bool(
                row["transaction_reference_available"]
            ),
            model_version=str(row["model_version"]),
            fraud_probability=float(row["fraud_probability"]),
            risk_level=str(row["risk_level"]),
            operating_mode=str(row["operating_mode"]),
            evidence_summary=EvidenceSummary.model_validate_json(
                str(row["evidence_summary_json"])
            ),
            analyst_decision=AnalystDecision(
                disposition=str(row["analyst_disposition"]),
                note=(str(row["analyst_note"]) if row["analyst_note"] is not None else None),
                disposition_at=(
                    str(row["disposition_at"])
                    if row["disposition_at"] is not None
                    else None
                ),
            ),
        )

    @staticmethod
    def _history(connection: sqlite3.Connection, case_id: str) -> list[CaseStatusHistoryItem]:
        rows = connection.execute(
            """
            SELECT occurred_at, previous_status, new_status, disposition, note_recorded
            FROM case_status_history
            WHERE case_id = ?
            ORDER BY sequence
            """,
            (case_id,),
        ).fetchall()
        return [
            CaseStatusHistoryItem(
                occurred_at=str(row["occurred_at"]),
                previous_status=(
                    str(row["previous_status"])
                    if row["previous_status"] is not None
                    else None
                ),
                new_status=str(row["new_status"]),
                disposition=str(row["disposition"]),
                note_recorded=bool(row["note_recorded"]),
            )
            for row in rows
        ]

    def _detail_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> CaseDetailResponse:
        copilot_json = row["copilot_json"]
        return CaseDetailResponse(
            case=self._summary(row),
            intelligence_snapshot=SanitizedInvestigationContext.model_validate_json(
                str(row["snapshot_json"])
            ),
            investigation_limitations=json.loads(str(row["limitations_json"])),
            copilot=(
                CopilotInvestigationResponse.model_validate_json(str(copilot_json))
                if copilot_json is not None
                else None
            ),
            status_history=self._history(connection, str(row["case_id"])),
        )

    @staticmethod
    def _new_case_id() -> str:
        return f"CASE-{secrets.token_hex(8).upper()}"

    def create(
        self,
        *,
        source_type: CaseSourceType,
        transaction_reference_available: bool,
        model_version: str,
        snapshot: SanitizedInvestigationContext,
        priority: CasePriority,
        evidence_summary: EvidenceSummary,
        limitations: list[str],
    ) -> CaseDetailResponse:
        now = _utc_now().isoformat()
        try:
            with self._connect() as connection:
                for _ in range(3):
                    case_id = self._new_case_id()
                    try:
                        connection.execute(
                            """
                            INSERT INTO cases VALUES (
                                ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                'NONE', NULL, NULL, ?, ?, ?, NULL
                            )
                            """,
                            (
                                case_id,
                                priority,
                                now,
                                now,
                                source_type,
                                int(transaction_reference_available),
                                model_version,
                                snapshot.model_output.fraud_probability,
                                snapshot.model_output.risk_level,
                                snapshot.model_output.operating_mode,
                                evidence_summary.model_dump_json(),
                                snapshot.model_dump_json(),
                                json.dumps(limitations),
                            ),
                        )
                        break
                    except sqlite3.IntegrityError:
                        continue
                else:
                    raise CaseStoreUnavailableError("Could not allocate a safe case ID")
                connection.execute(
                    """
                    INSERT INTO case_status_history
                    (case_id, occurred_at, previous_status, new_status, disposition, note_recorded)
                    VALUES (?, ?, NULL, 'OPEN', 'NONE', 0)
                    """,
                    (case_id, now),
                )
                row = connection.execute(
                    "SELECT * FROM cases WHERE case_id = ?", (case_id,)
                ).fetchone()
                return self._detail_from_row(connection, row)
        except CaseStoreUnavailableError:
            raise
        except sqlite3.Error as error:
            raise CaseStoreUnavailableError("Case could not be created") from error

    def get(self, case_id: str) -> CaseDetailResponse:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM cases WHERE case_id = ?", (case_id,)
                ).fetchone()
                if row is None:
                    raise CaseNotFoundError(f"Unknown case ID: {case_id}")
                return self._detail_from_row(connection, row)
        except CaseNotFoundError:
            raise
        except sqlite3.Error as error:
            raise CaseStoreUnavailableError("Case store could not be read") from error

    def list(
        self,
        *,
        status: CaseStatus | None,
        risk_level: RiskLevel | None,
        priority: CasePriority | None,
        disposition: AnalystDisposition | None,
        limit: int,
        offset: int,
    ) -> CaseListResponse:
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("status", status),
            ("risk_level", risk_level),
            ("priority", priority),
            ("analyst_disposition", disposition),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = """
            ORDER BY
                CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END,
                CASE priority
                    WHEN 'CRITICAL' THEN 4 WHEN 'HIGH' THEN 3
                    WHEN 'MEDIUM' THEN 2 ELSE 1 END DESC,
                CASE risk_level
                    WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 2 ELSE 1 END DESC,
                created_at ASC,
                case_id ASC
        """
        try:
            with self._connect() as connection:
                total = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM cases {where}", parameters
                    ).fetchone()[0]
                )
                rows = connection.execute(
                    f"SELECT * FROM cases {where} {order} LIMIT ? OFFSET ?",
                    [*parameters, limit, offset],
                ).fetchall()
                return CaseListResponse(
                    items=[self._summary(row) for row in rows],
                    total=total,
                    limit=limit,
                    offset=offset,
                )
        except sqlite3.Error as error:
            raise CaseStoreUnavailableError("Case queue could not be read") from error

    def update(self, case_id: str, request: CaseUpdateRequest) -> CaseDetailResponse:
        now = _utc_now().isoformat()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM cases WHERE case_id = ?", (case_id,)
                ).fetchone()
                if row is None:
                    raise CaseNotFoundError(f"Unknown case ID: {case_id}")
                current_status: CaseStatus = str(row["status"])
                target_status = request.status or current_status
                if (
                    request.status is not None
                    and target_status not in VALID_TRANSITIONS[current_status]
                ):
                    raise InvalidCaseTransitionError(
                        f"Invalid case transition: {current_status} -> {target_status}"
                    )
                if current_status == "CLOSED":
                    raise InvalidCaseTransitionError("Closed cases are immutable")

                disposition: AnalystDisposition = str(row["analyst_disposition"])
                disposition_at = row["disposition_at"]
                if target_status == "ESCALATED":
                    disposition, disposition_at = "ESCALATED", now
                elif target_status == "CLEARED":
                    disposition, disposition_at = "CLEARED", now
                note = request.analyst_note
                if note is None:
                    note = row["analyst_note"]
                connection.execute(
                    """
                    UPDATE cases
                    SET status = ?, updated_at = ?, analyst_disposition = ?,
                        analyst_note = ?, disposition_at = ?
                    WHERE case_id = ?
                    """,
                    (target_status, now, disposition, note, disposition_at, case_id),
                )
                if target_status != current_status:
                    connection.execute(
                        """
                        INSERT INTO case_status_history
                        (case_id, occurred_at, previous_status, new_status,
                         disposition, note_recorded)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            case_id,
                            now,
                            current_status,
                            target_status,
                            disposition,
                            int(request.analyst_note is not None),
                        ),
                    )
                updated = connection.execute(
                    "SELECT * FROM cases WHERE case_id = ?", (case_id,)
                ).fetchone()
                return self._detail_from_row(connection, updated)
        except (CaseNotFoundError, InvalidCaseTransitionError):
            raise
        except sqlite3.Error as error:
            raise CaseStoreUnavailableError("Case could not be updated") from error

    def save_copilot(
        self, case_id: str, response: CopilotInvestigationResponse
    ) -> CaseDetailResponse:
        now = _utc_now().isoformat()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT status FROM cases WHERE case_id = ?", (case_id,)
                ).fetchone()
                if row is None:
                    raise CaseNotFoundError(f"Unknown case ID: {case_id}")
                if str(row["status"]) == "CLOSED":
                    raise InvalidCaseTransitionError("Closed cases are immutable")
                connection.execute(
                    "UPDATE cases SET copilot_json = ?, updated_at = ? WHERE case_id = ?",
                    (response.model_dump_json(), now, case_id),
                )
                row = connection.execute(
                    "SELECT * FROM cases WHERE case_id = ?", (case_id,)
                ).fetchone()
                return self._detail_from_row(connection, row)
        except (CaseNotFoundError, InvalidCaseTransitionError):
            raise
        except sqlite3.Error as error:
            raise CaseStoreUnavailableError("Copilot report could not be saved") from error
