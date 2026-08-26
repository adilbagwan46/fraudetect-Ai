from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from statistics import fmean, median
from typing import Protocol

from backend.app.schemas.risk import (
    BehavioralContext,
    CurrentAmountBehavior,
    PriorAmountContext,
    RecentActivityContext,
    RecentWindowActivity,
    RiskPredictionRequest,
    TransactionTypeBehavior,
)

DEFAULT_RECENT_WINDOWS = (1, 6, 24)


class BehaviorHistoryUnavailableError(RuntimeError):
    """Raised when the generated local history index is not available."""


class TransactionReferenceNotFoundError(LookupError):
    """Raised when a safe internal transaction reference cannot be resolved."""


@dataclass(frozen=True)
class HistoricalTransaction:
    """Internal-only representation; identity keys must never enter public DTOs."""

    transaction_reference: str
    step: int
    transaction_type: str
    amount: float
    origin_balance_before: float
    origin_key: str

    def scoring_request(self) -> RiskPredictionRequest:
        return RiskPredictionRequest(
            transaction_type=self.transaction_type,
            amount=self.amount,
            origin_balance_before=self.origin_balance_before,
            hour_of_day=self.step % 24,
        )


class BehaviorHistoryProvider(Protocol):
    """Boundary that can later be implemented by a production history store."""

    def resolve(self, transaction_reference: str) -> HistoricalTransaction: ...

    def strictly_prior_for(
        self, current: HistoricalTransaction
    ) -> Sequence[HistoricalTransaction]: ...


def unavailable_behavioral_context(
    explanation: str = "No prepared historical identity was supplied for this transaction.",
    *,
    windows: Sequence[int] = DEFAULT_RECENT_WINDOWS,
) -> BehavioralContext:
    return BehavioralContext(
        history_available=False,
        availability_explanation=explanation,
        prior_transaction_count=0,
        prior_total_amount=0.0,
        prior_amount=None,
        current_amount_context=None,
        recent_activity=RecentActivityContext(
            windows=[
                RecentWindowActivity(
                    window_steps=window,
                    prior_transaction_count=0,
                    prior_amount_total=0.0,
                )
                for window in windows
            ],
            steps_since_previous_transaction=None,
        ),
        transaction_type_context=TransactionTypeBehavior(
            prior_transaction_type_count=0,
            is_new_transaction_type_for_origin=True,
        ),
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def build_behavioral_context(
    current: HistoricalTransaction,
    historical: Iterable[HistoricalTransaction],
    *,
    windows: Sequence[int] = DEFAULT_RECENT_WINDOWS,
) -> BehavioralContext:
    """Aggregate strictly prior origin history into a bounded public context."""

    if not windows or any(window <= 0 for window in windows):
        raise ValueError("Recent windows must contain positive step counts")

    prior = sorted(
        (
            event
            for event in historical
            if event.origin_key == current.origin_key
            and event.step < current.step
            and event.amount >= 0
            and math.isfinite(event.amount)
        ),
        key=lambda event: (event.step, event.transaction_reference),
    )
    if not prior:
        return unavailable_behavioral_context(
            "No transactions for this origin were observed at an earlier PaySim step.",
            windows=windows,
        )

    amounts = [event.amount for event in prior]
    average = float(fmean(amounts))
    middle = float(median(amounts))
    maximum = float(max(amounts))
    total = float(math.fsum(amounts))
    empirical_percentile = sum(amount <= current.amount for amount in amounts) / len(amounts)
    previous_step = max(event.step for event in prior)
    type_count = sum(event.transaction_type == current.transaction_type for event in prior)
    recent_windows = []
    for window in windows:
        lower_step = current.step - window
        events = [event for event in prior if event.step >= lower_step]
        recent_windows.append(
            RecentWindowActivity(
                window_steps=window,
                prior_transaction_count=len(events),
                prior_amount_total=float(math.fsum(event.amount for event in events)),
            )
        )

    return BehavioralContext(
        history_available=True,
        availability_explanation=(
            "Aggregates use only transactions for this origin with step < current step."
        ),
        prior_transaction_count=len(prior),
        prior_total_amount=total,
        prior_amount=PriorAmountContext(
            average=average,
            median=middle,
            maximum=maximum,
        ),
        current_amount_context=CurrentAmountBehavior(
            amount_vs_prior_average=_safe_ratio(current.amount, average),
            amount_vs_prior_median=_safe_ratio(current.amount, middle),
            amount_vs_prior_maximum=_safe_ratio(current.amount, maximum),
            prior_empirical_percentile=empirical_percentile,
            exceeds_prior_maximum=current.amount > maximum,
        ),
        recent_activity=RecentActivityContext(
            windows=recent_windows,
            steps_since_previous_transaction=current.step - previous_step,
        ),
        transaction_type_context=TransactionTypeBehavior(
            prior_transaction_type_count=type_count,
            is_new_transaction_type_for_origin=type_count == 0,
        ),
    )


def iter_causal_behavioral_contexts(
    chronological_events: Iterable[HistoricalTransaction],
    *,
    windows: Sequence[int] = DEFAULT_RECENT_WINDOWS,
) -> Iterable[tuple[str, BehavioralContext]]:
    """Generate offline contexts in one pass, updating state only after each step."""

    origin_history: dict[str, list[HistoricalTransaction]] = {}
    previous_step = -1
    for step, step_events_iterator in groupby(
        chronological_events,
        key=lambda event: event.step,
    ):
        if step < previous_step:
            raise ValueError("Offline behavioral events must be ordered by nondecreasing step")
        step_events = tuple(step_events_iterator)
        for current in step_events:
            yield (
                current.transaction_reference,
                build_behavioral_context(
                    current,
                    origin_history.get(current.origin_key, ()),
                    windows=windows,
                ),
            )
        for current in step_events:
            origin_history.setdefault(current.origin_key, []).append(current)
        previous_step = step


class SQLitePaySimHistoryProvider:
    """Read-only indexed lookup over an ignored, generated PaySim history store."""

    def __init__(self, database_path: Path) -> None:
        if not database_path.is_file():
            raise BehaviorHistoryUnavailableError(
                "Behavioral history index is unavailable; run build-behavior-history first"
            )
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path.resolve()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _from_row(row: sqlite3.Row) -> HistoricalTransaction:
        return HistoricalTransaction(
            transaction_reference=str(row["transaction_reference"]),
            step=int(row["step"]),
            transaction_type=str(row["transaction_type"]),
            amount=float(row["amount"]),
            origin_balance_before=float(row["origin_balance_before"]),
            origin_key=str(row["origin_key"]),
        )

    def resolve(self, transaction_reference: str) -> HistoricalTransaction:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT transaction_reference, step, transaction_type, amount,
                           origin_balance_before, origin_key
                    FROM transactions
                    WHERE transaction_reference = ?
                    """,
                    (transaction_reference,),
                ).fetchone()
        except sqlite3.Error as error:
            raise BehaviorHistoryUnavailableError(
                "Behavioral history index could not be read"
            ) from error
        if row is None:
            raise TransactionReferenceNotFoundError(
                f"Unknown transaction reference: {transaction_reference}"
            )
        return self._from_row(row)

    def strictly_prior_for(
        self, current: HistoricalTransaction
    ) -> Sequence[HistoricalTransaction]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT transaction_reference, step, transaction_type, amount,
                           origin_balance_before, origin_key
                    FROM transactions
                    WHERE origin_key = ? AND step < ?
                    ORDER BY step, transaction_reference
                    """,
                    (current.origin_key, current.step),
                ).fetchall()
        except sqlite3.Error as error:
            raise BehaviorHistoryUnavailableError(
                "Behavioral history index could not be read"
            ) from error
        return tuple(self._from_row(row) for row in rows)

    def context_for(
        self, transaction_reference: str
    ) -> tuple[HistoricalTransaction, BehavioralContext]:
        current = self.resolve(transaction_reference)
        return current, build_behavioral_context(current, self.strictly_prior_for(current))
