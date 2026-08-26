import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from backend.app.schemas.risk import DerivedFeatures, ModelOutputContext
from backend.app.services.behavioral_service import (
    HistoricalTransaction,
    SQLitePaySimHistoryProvider,
    build_behavioral_context,
    iter_causal_behavioral_contexts,
)
from backend.app.services.evidence_service import generate_evidence
from ml.fraudetect_ml.data.behavioral_index import build_behavioral_index
from tests.helpers import reference_profile


def event(
    reference: str,
    step: int,
    amount: float,
    *,
    transaction_type: str = "PAYMENT",
    origin: str = "C-internal-1",
) -> HistoricalTransaction:
    return HistoricalTransaction(
        transaction_reference=reference,
        step=step,
        transaction_type=transaction_type,
        amount=amount,
        origin_balance_before=1_000.0,
        origin_key=origin,
    )


def model_output() -> ModelOutputContext:
    return ModelOutputContext(
        fraud_probability=0.2,
        risk_score=20,
        risk_level="LOW",
        fraud_prediction=False,
        classification_threshold=0.5,
        operating_mode="BALANCED",
        model_version="frozen-test",
    )


def test_context_excludes_current_same_step_and_future_transactions() -> None:
    current = event("TX-current", 10, 30)
    historical = [
        event("TX-prior-1", 7, 10),
        event("TX-prior-2", 9, 20),
        current,
        event("TX-same-step", 10, 900),
        event("TX-future", 11, 10_000),
    ]

    context = build_behavioral_context(current, historical)

    assert context.prior_transaction_count == 2
    assert context.prior_total_amount == 30
    assert context.prior_amount is not None
    assert context.prior_amount.average == 15
    assert context.prior_amount.median == 15
    assert context.prior_amount.maximum == 20


def test_future_mutations_do_not_change_earlier_context() -> None:
    current = event("TX-current", 10, 30)
    base = [event("TX-prior", 9, 10), event("TX-future", 11, 100)]
    mutated = [event("TX-prior", 9, 10), event("TX-future", 11, 1_000_000)]

    assert build_behavioral_context(current, base) == build_behavioral_context(
        current, mutated
    )


def test_offline_generation_updates_state_only_after_complete_step() -> None:
    contexts = dict(
        iter_causal_behavioral_contexts(
            [
                event("TX-step-1", 1, 10),
                event("TX-step-2-a", 2, 20),
                event("TX-step-2-b", 2, 30),
                event("TX-step-3", 3, 40),
            ]
        )
    )

    assert contexts["TX-step-1"].history_available is False
    assert contexts["TX-step-2-a"].prior_transaction_count == 1
    assert contexts["TX-step-2-b"].prior_transaction_count == 1
    assert contexts["TX-step-3"].prior_transaction_count == 3


def test_no_prior_history_is_explicit_and_does_not_invent_baselines() -> None:
    context = build_behavioral_context(event("TX-first", 1, 10), [])

    assert context.history_available is False
    assert context.prior_transaction_count == 0
    assert context.prior_amount is None
    assert context.current_amount_context is None
    assert context.recent_activity.steps_since_previous_transaction is None


def test_zero_historical_amounts_stay_finite_without_invented_ratios() -> None:
    context = build_behavioral_context(
        event("TX-current", 3, 10),
        [event("TX-zero-1", 1, 0), event("TX-zero-2", 2, 0)],
    )

    assert context.current_amount_context is not None
    assert context.current_amount_context.amount_vs_prior_average is None
    assert context.current_amount_context.amount_vs_prior_median is None
    assert context.current_amount_context.amount_vs_prior_maximum is None
    numeric_values = [
        value
        for value in context.model_dump().values()
        if isinstance(value, (int, float))
    ]
    assert all(math.isfinite(value) for value in numeric_values)


def test_amount_statistics_ignore_invalid_prior_amounts() -> None:
    current = event("TX-current", 5, 30)
    context = build_behavioral_context(
        current,
        [
            event("TX-valid-1", 1, 10),
            event("TX-invalid-negative", 2, -10),
            event("TX-invalid-infinite", 3, float("inf")),
            event("TX-valid-2", 4, 20),
        ],
    )

    assert context.prior_transaction_count == 2
    assert context.prior_amount is not None
    assert context.prior_amount.average == 15
    assert context.prior_amount.median == 15


def test_type_novelty_and_recent_windows_use_only_prior_steps() -> None:
    current = event("TX-current", 10, 30, transaction_type="TRANSFER")
    historical = [
        event("TX-old", 3, 5),
        event("TX-recent-1", 9, 10),
        event("TX-recent-2", 9, 20),
        event("TX-same", 10, 50, transaction_type="TRANSFER"),
    ]

    context = build_behavioral_context(current, historical, windows=(1, 6))

    assert context.transaction_type_context.prior_transaction_type_count == 0
    assert context.transaction_type_context.is_new_transaction_type_for_origin is True
    assert context.recent_activity.steps_since_previous_transaction == 1
    assert context.recent_activity.windows[0].prior_transaction_count == 2
    assert context.recent_activity.windows[0].prior_amount_total == 30
    assert context.recent_activity.windows[1].prior_transaction_count == 2


def test_public_behavioral_context_contains_no_internal_identifiers() -> None:
    context = build_behavioral_context(
        event("TX-current-secret", 3, 10),
        [event("TX-prior-secret", 1, 5)],
    )
    serialized = context.model_dump_json()

    assert "C-internal-1" not in serialized
    assert "TX-current-secret" not in serialized
    assert "TX-prior-secret" not in serialized
    assert "origin_key" not in serialized


def test_behavioral_evidence_is_deterministic_and_existing_evidence_is_unchanged() -> None:
    current = event("TX-current", 5, 1_000, transaction_type="TRANSFER")
    context = build_behavioral_context(current, [event("TX-prior", 1, 10)])
    request = current.scoring_request()
    derived = DerivedFeatures(
        log_amount=math.log1p(request.amount),
        amount_to_origin_balance=1.0,
    )
    kwargs = {
        "transaction": request,
        "derived_features": derived,
        "model_output": model_output(),
        "reference_profile": reference_profile(),
    }

    original_first = generate_evidence(**kwargs)
    original_second = generate_evidence(**kwargs)
    behavioral_first = generate_evidence(**kwargs, behavioral_context=context)
    behavioral_second = generate_evidence(**kwargs, behavioral_context=context)

    assert original_first == original_second
    assert behavioral_first == behavioral_second
    assert original_first == generate_evidence(**kwargs, behavioral_context=None)
    assert any(item.category == "BEHAVIORAL_CONTEXT" for item in behavioral_first)


def test_no_history_and_recent_activity_create_factual_behavioral_evidence() -> None:
    current = event("TX-current", 10, 30)
    unavailable = build_behavioral_context(current, [])
    recent = build_behavioral_context(
        current,
        [event("TX-prior", 9, 10)],
    )
    request = current.scoring_request()
    kwargs = {
        "transaction": request,
        "derived_features": DerivedFeatures(
            log_amount=math.log1p(request.amount),
            amount_to_origin_balance=0.03,
        ),
        "model_output": model_output(),
        "reference_profile": reference_profile(),
    }

    unavailable_ids = {
        item.id for item in generate_evidence(**kwargs, behavioral_context=unavailable)
    }
    recent_ids = {item.id for item in generate_evidence(**kwargs, behavioral_context=recent)}

    assert "behavior_history_unavailable" in unavailable_ids
    assert "behavior_recent_activity" in recent_ids


def test_generated_history_index_is_label_free_and_uses_indexed_causal_lookup(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    columns = {
        "transaction_id": ["TX-1", "TX-2", "TX-3"],
        "step": [1, 2, 2],
        "transaction_type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [10.0, 50.0, 999.0],
        "origin_balance_before": [100.0, 100.0, 100.0],
        "customer_id": ["C1", "C1", "C1"],
        "is_fraud": [1, 0, 1],
        "is_flagged_fraud": [1, 0, 1],
    }
    paths = {}
    for split, frame in (
        ("train", pd.DataFrame(columns).iloc[:1]),
        ("validation", pd.DataFrame(columns).iloc[1:2]),
        ("test", pd.DataFrame(columns).iloc[2:]),
    ):
        path = processed / f"{split}.csv"
        frame.to_csv(path, index=False)
        paths[split] = {"path": str(path)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {"sha256": "dataset-test"},
                "splits": paths,
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "history.sqlite"

    result = build_behavioral_index(manifest, database, chunksize=1)
    provider = SQLitePaySimHistoryProvider(database)
    _, context = provider.context_for("TX-2")

    assert result["rows"] == 3
    assert result["labels_stored"] is False
    assert context.prior_transaction_count == 1
    assert context.prior_total_amount == 10
    with sqlite3.connect(database) as connection:
        table_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(transactions)")
        }
        query_plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM transactions "
            "WHERE origin_key = ? AND step < ? ORDER BY step, transaction_reference",
            ("C1", 2),
        ).fetchall()
    assert "is_fraud" not in table_columns
    assert "is_flagged_fraud" not in table_columns
    assert any("transactions_origin_step_idx" in str(row) for row in query_plan)
