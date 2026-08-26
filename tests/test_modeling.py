import numpy as np
import pandas as pd

from ml.fraudetect_ml.data.contracts import ML_FEATURE_COLUMNS
from ml.fraudetect_ml.modeling.candidates import CandidateSpec, build_candidate_pipeline
from ml.fraudetect_ml.modeling.data import split_fit_calibration
from ml.fraudetect_ml.modeling.evaluation import ranking_metrics_at_capacity
from ml.fraudetect_ml.modeling.thresholds import select_threshold_policies


def model_frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_type": ["PAYMENT", "TRANSFER"] * (rows // 2),
            "amount": np.arange(1, rows + 1, dtype=float),
            "origin_balance_before": np.arange(101, 101 + rows, dtype=float),
            "hour_of_day": np.arange(rows) % 24,
            "log_amount": np.log1p(np.arange(1, rows + 1, dtype=float)),
            "amount_to_origin_balance": np.arange(1, rows + 1) / np.arange(101, 101 + rows),
            "is_fraud": ([0] * 9 + [1]) * (rows // 10),
            "step": np.repeat(np.arange(1, rows // 2 + 1), 2),
        }
    )


def test_fit_calibration_partition_is_strictly_chronological() -> None:
    partition = split_fit_calibration(model_frame(), fit_fraction=0.75)

    assert partition.fit_max_step < partition.calibration_min_step
    assert tuple(partition.fit_features.columns) == ML_FEATURE_COLUMNS
    assert len(partition.fit_target) + len(partition.calibration_target) == 40


def test_preprocessing_statistics_are_fit_from_training_only() -> None:
    fit = model_frame(40)
    validation = model_frame(40)
    validation["amount"] = 1_000_000
    spec = CandidateSpec("test_logistic", "logistic_regression", False)
    pipeline = build_candidate_pipeline(spec, class_weights=None, random_state=7)

    pipeline.fit(fit.loc[:, ML_FEATURE_COLUMNS], fit["is_fraud"])
    pipeline.predict_proba(validation.loc[:, ML_FEATURE_COLUMNS])
    numeric_pipeline = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]

    assert numeric_pipeline.named_steps["imputer"].statistics_[0] == 20.5
    assert numeric_pipeline.named_steps["scaler"].mean_[0] == 20.5


def test_threshold_policies_have_documented_capacity_ordering() -> None:
    target = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0] * 200)
    probabilities = np.linspace(1, 0, len(target), endpoint=False)

    policies = select_threshold_policies(target, probabilities)

    assert set(policies) == {"HIGH_PRECISION", "BALANCED", "HIGH_RECALL"}
    assert policies["HIGH_PRECISION"]["review_rate"] <= 0.001
    assert policies["HIGH_RECALL"]["review_rate"] <= 0.01
    assert policies["HIGH_RECALL"]["recall"] >= policies["HIGH_PRECISION"]["recall"]


def test_ranking_operating_point_uses_exact_top_k_capacity() -> None:
    target = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    metrics = ranking_metrics_at_capacity(target, probabilities, 0.5)

    assert metrics["review_count"] == 5
    assert metrics["precision"] == 0.4
    assert metrics["recall"] == 1.0

