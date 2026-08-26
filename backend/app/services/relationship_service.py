from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from statistics import fmean, median
from typing import Protocol

from backend.app.schemas.risk import (
    CurrentRelationshipAmountContext,
    DestinationNetworkContext,
    OriginNetworkContext,
    RelationshipAmountContext,
    RelationshipContext,
)


class RelationshipHistoryUnavailableError(RuntimeError):
    """Raised when the generated relationship index cannot be used."""


class RelationshipTransactionNotFoundError(LookupError):
    """Raised when an internal transaction reference is absent from the index."""


@dataclass(frozen=True)
class RelationshipTransaction:
    """Internal-only event. Entity keys must never cross the provider boundary."""

    transaction_reference: str
    step: int
    amount: float
    origin_key: str
    destination_key: str


class RelationshipHistoryProvider(Protocol):
    def context_for(self, transaction_reference: str) -> RelationshipContext: ...


def unavailable_relationship_context(
    explanation: str = (
        "Relationship context requires a prepared transaction reference and is unavailable "
        "for manual input."
    ),
) -> RelationshipContext:
    return RelationshipContext(
        context_available=False,
        history_available=False,
        availability_explanation=explanation,
        relationship_seen_before=None,
        relationship_first_seen=None,
        prior_interaction_count=0,
        prior_total_amount=0.0,
        prior_amount=None,
        current_amount_context=None,
        steps_since_previous_interaction=None,
        baseline_is_limited=True,
        origin_network=OriginNetworkContext(
            prior_unique_counterparty_count=0,
            prior_transaction_count=0,
            current_destination_is_new=None,
        ),
        destination_network=DestinationNetworkContext(
            prior_unique_origin_count=0,
            prior_transaction_count=0,
            current_origin_is_new_for_destination=None,
        ),
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _context_from_prior(
    current: RelationshipTransaction,
    prior: Iterable[RelationshipTransaction],
) -> RelationshipContext:
    eligible = tuple(
        event
        for event in prior
        if event.step < current.step and event.amount >= 0 and math.isfinite(event.amount)
    )
    relationship = tuple(
        event
        for event in eligible
        if event.origin_key == current.origin_key
        and event.destination_key == current.destination_key
    )
    origin_events = tuple(event for event in eligible if event.origin_key == current.origin_key)
    destination_events = tuple(
        event for event in eligible if event.destination_key == current.destination_key
    )
    pair_count = len(relationship)
    relationship_seen = pair_count > 0
    origin_destinations = {event.destination_key for event in origin_events}
    destination_origins = {event.origin_key for event in destination_events}

    if not relationship:
        return RelationshipContext(
            context_available=True,
            history_available=False,
            availability_explanation=(
                "No prior interaction between this origin and destination was observed at "
                "an earlier PaySim step."
            ),
            relationship_seen_before=False,
            relationship_first_seen=True,
            prior_interaction_count=0,
            prior_total_amount=0.0,
            prior_amount=None,
            current_amount_context=None,
            steps_since_previous_interaction=None,
            baseline_is_limited=True,
            origin_network=OriginNetworkContext(
                prior_unique_counterparty_count=len(origin_destinations),
                prior_transaction_count=len(origin_events),
                current_destination_is_new=True,
            ),
            destination_network=DestinationNetworkContext(
                prior_unique_origin_count=len(destination_origins),
                prior_transaction_count=len(destination_events),
                current_origin_is_new_for_destination=True,
            ),
        )

    amounts = [event.amount for event in relationship]
    average = float(fmean(amounts))
    middle = float(median(amounts))
    maximum = float(max(amounts))
    return RelationshipContext(
        context_available=True,
        history_available=True,
        availability_explanation=(
            "Aggregates use only origin-destination interactions with step < current step."
        ),
        relationship_seen_before=relationship_seen,
        relationship_first_seen=False,
        prior_interaction_count=pair_count,
        prior_total_amount=float(math.fsum(amounts)),
        prior_amount=RelationshipAmountContext(
            average=average,
            median=middle,
            maximum=maximum,
        ),
        current_amount_context=CurrentRelationshipAmountContext(
            amount_vs_prior_average=_safe_ratio(current.amount, average),
            amount_vs_prior_median=_safe_ratio(current.amount, middle),
            amount_vs_prior_maximum=_safe_ratio(current.amount, maximum),
            prior_empirical_percentile=(
                sum(amount <= current.amount for amount in amounts) / pair_count
            ),
            exceeds_prior_relationship_maximum=current.amount > maximum,
        ),
        steps_since_previous_interaction=(
            current.step - max(event.step for event in relationship)
        ),
        baseline_is_limited=pair_count <= 2,
        origin_network=OriginNetworkContext(
            prior_unique_counterparty_count=len(origin_destinations),
            prior_transaction_count=len(origin_events),
            current_destination_is_new=False,
        ),
        destination_network=DestinationNetworkContext(
            prior_unique_origin_count=len(destination_origins),
            prior_transaction_count=len(destination_events),
            current_origin_is_new_for_destination=False,
        ),
    )


def build_relationship_context(
    current: RelationshipTransaction,
    historical: Iterable[RelationshipTransaction],
) -> RelationshipContext:
    """Build identifier-free aggregates from strictly earlier events only."""

    return _context_from_prior(current, historical)


def iter_causal_relationship_contexts(
    chronological_events: Iterable[RelationshipTransaction],
) -> Iterable[tuple[str, RelationshipContext]]:
    """Yield contexts before adding any event from the current step to state."""

    prior: list[RelationshipTransaction] = []
    previous_step = -1
    for step, step_events_iterator in groupby(
        chronological_events,
        key=lambda event: event.step,
    ):
        if step < previous_step:
            raise ValueError("Relationship events must be ordered by nondecreasing step")
        step_events = tuple(step_events_iterator)
        for current in step_events:
            yield current.transaction_reference, _context_from_prior(current, prior)
        prior.extend(step_events)
        previous_step = step


class SQLiteRelationshipHistoryProvider:
    """Read-only indexed provider over the ignored label-free relationship artifact."""

    def __init__(self, database_path: Path) -> None:
        if not database_path.is_file():
            raise RelationshipHistoryUnavailableError(
                "Relationship index is unavailable; run build-relationship-history first"
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
    def _transaction(row: sqlite3.Row) -> RelationshipTransaction:
        return RelationshipTransaction(
            transaction_reference=str(row["transaction_reference"]),
            step=int(row["step"]),
            amount=float(row["amount"]),
            origin_key=str(row["origin_key"]),
            destination_key=str(row["destination_key"]),
        )

    def _resolve(
        self,
        connection: sqlite3.Connection,
        transaction_reference: str,
    ) -> RelationshipTransaction:
        row = connection.execute(
            """
            SELECT transaction_reference, step, amount, origin_key, destination_key
            FROM relationship_transactions
            WHERE transaction_reference = ?
            """,
            (transaction_reference,),
        ).fetchone()
        if row is None:
            raise RelationshipTransactionNotFoundError(
                f"Unknown transaction reference: {transaction_reference}"
            )
        return self._transaction(row)

    def context_for(self, transaction_reference: str) -> RelationshipContext:
        try:
            with self._connect() as connection:
                current = self._resolve(connection, transaction_reference)
                pair_rows = connection.execute(
                    """
                    SELECT step, amount
                    FROM relationship_transactions
                    WHERE origin_key = ? AND destination_key = ? AND step < ?
                    ORDER BY step, transaction_reference
                    """,
                    (current.origin_key, current.destination_key, current.step),
                ).fetchall()
                origin_row = connection.execute(
                    """
                    SELECT COUNT(*) AS transaction_count,
                           COUNT(DISTINCT destination_key) AS unique_count
                    FROM relationship_transactions
                    WHERE origin_key = ? AND step < ?
                    """,
                    (current.origin_key, current.step),
                ).fetchone()
                destination_row = connection.execute(
                    """
                    SELECT COUNT(*) AS transaction_count,
                           COUNT(DISTINCT origin_key) AS unique_count
                    FROM relationship_transactions
                    WHERE destination_key = ? AND step < ?
                    """,
                    (current.destination_key, current.step),
                ).fetchone()
        except RelationshipTransactionNotFoundError:
            raise
        except sqlite3.Error as error:
            raise RelationshipHistoryUnavailableError(
                "Relationship index could not be read"
            ) from error

        prior = tuple(
            RelationshipTransaction(
                transaction_reference="internal-prior-event",
                step=int(row["step"]),
                amount=float(row["amount"]),
                origin_key=current.origin_key,
                destination_key=current.destination_key,
            )
            for row in pair_rows
        )
        base = _context_from_prior(current, prior)
        origin_count = int(origin_row["transaction_count"])
        origin_unique = int(origin_row["unique_count"])
        destination_count = int(destination_row["transaction_count"])
        destination_unique = int(destination_row["unique_count"])
        return base.model_copy(
            update={
                "origin_network": OriginNetworkContext(
                    prior_unique_counterparty_count=origin_unique,
                    prior_transaction_count=origin_count,
                    current_destination_is_new=not base.history_available,
                ),
                "destination_network": DestinationNetworkContext(
                    prior_unique_origin_count=destination_unique,
                    prior_transaction_count=destination_count,
                    current_origin_is_new_for_destination=not base.history_available,
                ),
            }
        )
