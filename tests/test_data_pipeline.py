import json
from pathlib import Path

import pandas as pd

from ml.fraudetect_ml.data.contracts import ML_EXCLUDED_COLUMNS, ML_FEATURE_COLUMNS
from ml.fraudetect_ml.data.enrichment import add_demo_relationship_fields
from ml.fraudetect_ml.data.features import add_foundation_features, build_model_matrix
from ml.fraudetect_ml.data.historical import strictly_prior_events
from ml.fraudetect_ml.data.pipeline import prepare_dataset
from ml.fraudetect_ml.data.splitting import chronological_split
from scripts.generate_demo_data import generate_rows


def canonical_demo_frame(row_count: int = 100) -> pd.DataFrame:
    raw = pd.DataFrame(generate_rows(row_count, seed=42)).rename(
        columns={
            "type": "transaction_type",
            "nameOrig": "customer_id",
            "oldbalanceOrg": "origin_balance_before",
            "newbalanceOrig": "origin_balance_after",
            "nameDest": "counterparty_id",
            "oldbalanceDest": "destination_balance_before",
            "newbalanceDest": "destination_balance_after",
            "isFraud": "is_fraud",
            "isFlaggedFraud": "is_flagged_fraud",
        }
    )
    raw["transaction_id"] = [f"TX-{index:04d}" for index in range(len(raw))]
    return raw


def test_demo_enrichment_is_reproducible_and_label_independent() -> None:
    frame = canonical_demo_frame()
    changed_labels = frame.copy()
    changed_labels["is_fraud"] = 1 - changed_labels["is_fraud"]

    kwargs = {
        "seed": "fixed",
        "device_buckets": 18,
        "ip_buckets": 12,
        "source_kind": "generated_demo_only",
    }
    first = add_demo_relationship_fields(frame, **kwargs)
    second = add_demo_relationship_fields(changed_labels, **kwargs)

    assert first["device_id"].equals(second["device_id"])
    assert first["ip_id"].equals(second["ip_id"])
    assert first["device_id"].nunique() < first["customer_id"].nunique()


def test_foundation_features_have_expected_invariants() -> None:
    featured = add_foundation_features(canonical_demo_frame())

    assert featured["hour_of_day"].between(0, 23).all()
    assert (featured["log_amount"] >= 0).all()
    assert (featured["origin_balance_error"] >= 0).all()


def test_temporal_splits_are_ordered_and_disjoint() -> None:
    frame = canonical_demo_frame(120).sample(frac=1, random_state=3)
    splits = chronological_split(frame)

    ids = [set(split["transaction_id"]) for split in (splits.train, splits.validation, splits.test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert splits.train["step"].max() <= splits.validation["step"].min()
    assert splits.validation["step"].max() <= splits.test["step"].min()
    assert sum(map(len, (splits.train, splits.validation, splits.test))) == len(frame)


def test_complete_steps_never_cross_split_boundaries() -> None:
    frame = canonical_demo_frame(121)
    frame.loc[frame.index[-17:], "step"] = 500

    splits = chronological_split(frame)
    step_sets = [set(split["step"]) for split in (splits.train, splits.validation, splits.test)]

    assert step_sets[0].isdisjoint(step_sets[1])
    assert step_sets[0].isdisjoint(step_sets[2])
    assert step_sets[1].isdisjoint(step_sets[2])
    assert sum(len(split) for split in (splits.train, splits.validation, splits.test)) == len(frame)


def test_model_matrix_uses_only_explicit_safe_allowlist() -> None:
    prepared = add_foundation_features(canonical_demo_frame())
    prepared = add_demo_relationship_fields(
        prepared,
        seed="fixed",
        device_buckets=18,
        ip_buckets=12,
        source_kind="generated_demo_only",
    )

    model_matrix = build_model_matrix(prepared)

    assert tuple(model_matrix.columns) == ML_FEATURE_COLUMNS
    assert set(model_matrix.columns).isdisjoint(ML_EXCLUDED_COLUMNS)
    assert "is_fraud" not in model_matrix
    assert "origin_balance_after" not in model_matrix
    assert "device_id" not in model_matrix


def test_causal_history_excludes_same_step_and_future_events() -> None:
    frame = canonical_demo_frame(60)

    history = strictly_prior_events(frame, current_step=5)

    assert (history["step"] < 5).all()
    assert not (history["step"] == 5).any()


def test_prepare_dataset_writes_provenance_manifest(tmp_path: Path) -> None:
    source = tmp_path / "demo.csv"
    output = tmp_path / "prepared"
    pd.DataFrame(generate_rows(60, seed=9)).to_csv(source, index=False)

    manifest = prepare_dataset(
        source,
        output,
        enrichment_seed="test-seed",
        source_kind="generated_demo_only",
    )
    saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["dataset"]["rows"] == 60
    assert saved_manifest["source"]["enrichment_uses_label"] is False
    assert saved_manifest["source"]["enrichment_configuration"][
        "depends_on_dataset_cardinality"
    ] is False
    assert saved_manifest["source"]["kind"] == "generated_demo_only"
    assert saved_manifest["split_strategy"]["complete_steps_are_atomic"] is True
    assert tuple(saved_manifest["ml_feature_contract"]["features"]) == ML_FEATURE_COLUMNS
    assert {path.name for path in output.glob("*.csv")} == {
        "train.csv",
        "validation.csv",
        "test.csv",
    }
