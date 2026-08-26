import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from backend.app.services.evidence_service import generate_relationship_evidence
from backend.app.services.relationship_service import (
    RelationshipTransaction,
    SQLiteRelationshipHistoryProvider,
    build_relationship_context,
    iter_causal_relationship_contexts,
    unavailable_relationship_context,
)
from ml.fraudetect_ml.data.relationship_index import build_relationship_index


def event(
    reference: str,
    step: int,
    amount: float,
    *,
    origin: str = "origin-private-1",
    destination: str = "destination-private-1",
) -> RelationshipTransaction:
    return RelationshipTransaction(
        transaction_reference=reference,
        step=step,
        amount=amount,
        origin_key=origin,
        destination_key=destination,
    )


def assert_finite_tree(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite_tree(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_relationship_context_excludes_current_same_step_and_future_events() -> None:
    current = event("TX-current", 10, 100)
    context = build_relationship_context(
        current,
        [
            event("TX-prior", 9, 20),
            current,
            event("TX-same-step", 10, 900),
            event("TX-future", 11, 10_000),
        ],
    )

    assert context.prior_interaction_count == 1
    assert context.prior_total_amount == 20
    assert context.prior_amount is not None
    assert context.prior_amount.maximum == 20
    assert context.current_amount_context is not None
    assert context.current_amount_context.amount_vs_prior_average == 5


def test_future_mutation_cannot_change_earlier_relationship_context() -> None:
    current = event("TX-current", 10, 100)
    base = [event("TX-prior", 9, 20), event("TX-future", 11, 50)]
    mutated = [event("TX-prior", 9, 20), event("TX-future", 11, 9_000_000)]

    assert build_relationship_context(current, base) == build_relationship_context(
        current, mutated
    )


def test_offline_contexts_update_state_after_the_complete_step() -> None:
    contexts = dict(
        iter_causal_relationship_contexts(
            [
                event("TX-train", 1, 10),
                event("TX-validation-a", 2, 20),
                event("TX-validation-b", 2, 30),
                event("TX-test", 3, 40),
            ]
        )
    )

    assert contexts["TX-train"].prior_interaction_count == 0
    assert contexts["TX-validation-a"].prior_interaction_count == 1
    assert contexts["TX-validation-b"].prior_interaction_count == 1
    assert contexts["TX-test"].prior_interaction_count == 3


def test_new_relationship_and_network_novelty_are_explicit() -> None:
    current = event("TX-current", 3, 25, destination="destination-new")
    context = build_relationship_context(
        current,
        [
            event("TX-origin-prior", 1, 10, destination="destination-old"),
            event(
                "TX-destination-prior",
                2,
                20,
                origin="origin-other",
                destination="destination-new",
            ),
        ],
    )

    assert context.context_available is True
    assert context.history_available is False
    assert context.relationship_first_seen is True
    assert context.relationship_seen_before is False
    assert context.origin_network.prior_unique_counterparty_count == 1
    assert context.origin_network.current_destination_is_new is True
    assert context.destination_network.prior_unique_origin_count == 1
    assert context.destination_network.current_origin_is_new_for_destination is True


def test_repeated_relationship_amounts_and_destination_origins_are_aggregated() -> None:
    current = event("TX-current", 5, 30)
    context = build_relationship_context(
        current,
        [
            event("TX-pair-1", 1, 10),
            event("TX-pair-2", 3, 20),
            event("TX-origin-other-destination", 2, 7, destination="destination-other"),
            event("TX-other-origin", 4, 9, origin="origin-private-2"),
        ],
    )

    assert context.relationship_seen_before is True
    assert context.prior_interaction_count == 2
    assert context.prior_total_amount == 30
    assert context.prior_amount is not None
    assert context.prior_amount.average == 15
    assert context.prior_amount.median == 15
    assert context.prior_amount.maximum == 20
    assert context.steps_since_previous_interaction == 2
    assert context.origin_network.prior_unique_counterparty_count == 2
    assert context.destination_network.prior_unique_origin_count == 2
    assert context.baseline_is_limited is True


def test_strong_deviation_and_normal_amount_evidence_are_factual() -> None:
    prior = [
        event("TX-1", 1, 10),
        event("TX-2", 2, 20),
        event("TX-3", 3, 30),
    ]
    deviation = build_relationship_context(event("TX-current", 4, 300), prior)
    normal = build_relationship_context(event("TX-normal", 4, 20), prior)
    deviation_ids = {item.id for item in generate_relationship_evidence(deviation)}
    normal_ids = {item.id for item in generate_relationship_evidence(normal)}

    assert "relationship_amount_deviation" in deviation_ids
    assert "relationship_exceeds_prior_maximum" in deviation_ids
    assert "relationship_amount_deviation" not in normal_ids
    assert "relationship_exceeds_prior_maximum" not in normal_ids
    assert "relationship_previously_observed" in normal_ids


def test_zero_relationship_baseline_uses_null_ratios_and_never_nonfinite_values() -> None:
    context = build_relationship_context(
        event("TX-current", 3, 10),
        [event("TX-zero-1", 1, 0), event("TX-zero-2", 2, 0)],
    )

    assert context.current_amount_context is not None
    assert context.current_amount_context.amount_vs_prior_average is None
    assert context.current_amount_context.amount_vs_prior_median is None
    assert context.current_amount_context.amount_vs_prior_maximum is None
    assert_finite_tree(context.model_dump())
    assert "NaN" not in context.model_dump_json()
    assert "Infinity" not in context.model_dump_json()


def test_public_context_and_unavailable_context_contain_no_internal_identity() -> None:
    context = build_relationship_context(
        event("TX-current-private", 2, 20),
        [event("TX-prior-private", 1, 10)],
    )
    serialized = context.model_dump_json()

    for forbidden in (
        "TX-current-private",
        "TX-prior-private",
        "origin-private-1",
        "destination-private-1",
        "origin_key",
        "destination_key",
    ):
        assert forbidden not in serialized
    assert unavailable_relationship_context().context_available is False


def test_generated_relationship_index_is_label_free_indexed_and_cross_split_causal(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    frame = pd.DataFrame(
        {
            "transaction_id": ["TX-1", "TX-2", "TX-3", "TX-4"],
            "step": [1, 2, 2, 3],
            "amount": [10.0, 50.0, 999.0, 70.0],
            "customer_id": ["C1", "C1", "C1", "C1"],
            "counterparty_id": ["M1", "M1", "M1", "M1"],
            "is_fraud": [1, 0, 1, 0],
            "is_flagged_fraud": [1, 0, 1, 0],
        }
    )
    paths = {}
    for split, subset in (
        ("train", frame.iloc[:1]),
        ("validation", frame.iloc[1:3]),
        ("test", frame.iloc[3:]),
    ):
        path = processed / f"{split}.csv"
        subset.to_csv(path, index=False)
        paths[split] = {"path": str(path)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"source": {"sha256": "dataset-test"}, "splits": paths}
        ),
        encoding="utf-8",
    )
    database = tmp_path / "relationship.sqlite"

    result = build_relationship_index(manifest, database, chunksize=1)
    provider = SQLiteRelationshipHistoryProvider(database)
    validation = provider.context_for("TX-2")
    same_step = provider.context_for("TX-3")
    test = provider.context_for("TX-4")

    assert result["rows"] == 4
    assert result["labels_loaded"] is False
    assert result["labels_stored"] is False
    assert validation.prior_interaction_count == 1
    assert same_step.prior_interaction_count == 1
    assert test.prior_interaction_count == 3
    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(relationship_transactions)"
            )
        }
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        pair_plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT step, amount FROM relationship_transactions "
            "WHERE origin_key = ? AND destination_key = ? AND step < ?",
            ("C1", "M1", 3),
        ).fetchall()
    assert "is_fraud" not in columns
    assert "is_flagged_fraud" not in columns
    assert metadata["causal_boundary"] == "historical.step < current.step"
    assert metadata["labels_loaded"] == "false"
    assert any("relationship_pair_step_idx" in str(row) for row in pair_plan)
