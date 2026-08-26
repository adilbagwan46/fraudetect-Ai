from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.fraudetect_ml.data.contracts import ML_FEATURE_COLUMNS

NUMERIC_FEATURES = (
    "amount",
    "origin_balance_before",
    "hour_of_day",
    "log_amount",
    "amount_to_origin_balance",
)
CATEGORICAL_FEATURES = ("transaction_type",)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: Literal["logistic_regression", "hist_gradient_boosting"]
    weighted: bool


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("logistic_unweighted", "logistic_regression", False),
        CandidateSpec("logistic_balanced", "logistic_regression", True),
        CandidateSpec("hist_gradient_boosting_unweighted", "hist_gradient_boosting", False),
        CandidateSpec("hist_gradient_boosting_balanced", "hist_gradient_boosting", True),
    )


def balanced_class_weights(target: np.ndarray) -> dict[int, float]:
    counts = np.bincount(target.astype("int64"), minlength=2)
    if (counts == 0).any():
        raise ValueError("Both classes are required to calculate balanced weights")
    total = int(counts.sum())
    return {0: total / (2 * int(counts[0])), 1: total / (2 * int(counts[1]))}


def build_candidate_pipeline(
    spec: CandidateSpec,
    *,
    class_weights: dict[int, float] | None,
    random_state: int,
) -> Pipeline:
    if tuple((*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)) != ML_FEATURE_COLUMNS:
        raise RuntimeError("Candidate preprocessing no longer matches the ML feature contract")
    if spec.weighted and class_weights is None:
        raise ValueError("Weighted candidate requires training-derived class weights")
    selected_weights = class_weights if spec.weighted else None

    if spec.family == "logistic_regression":
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessing = ColumnTransformer(
            [
                ("categorical", categorical, list(CATEGORICAL_FEATURES)),
                ("numeric", numeric, list(NUMERIC_FEATURES)),
            ],
            sparse_threshold=1.0,
        )
        estimator = LogisticRegression(
            class_weight=selected_weights,
            max_iter=200,
            random_state=random_state,
            solver="lbfgs",
            tol=1e-4,
        )
    else:
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        preprocessing = ColumnTransformer(
            [
                ("categorical", categorical, list(CATEGORICAL_FEATURES)),
                ("numeric", numeric, list(NUMERIC_FEATURES)),
            ],
            sparse_threshold=0.0,
        )
        estimator = HistGradientBoostingClassifier(
            class_weight=selected_weights,
            early_stopping=False,
            l2_regularization=1.0,
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=31,
            random_state=random_state,
        )

    return Pipeline([("preprocessing", preprocessing), ("estimator", estimator)])
