from __future__ import annotations

import numpy as np
import pandas as pd

from ml.fraudetect_ml.data.contracts import ML_FEATURE_COLUMNS, ML_TARGET_COLUMN


class FeatureContractError(ValueError):
    """Raised when prepared data cannot satisfy the Phase 2 feature contract."""


def add_safe_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive row-local fields available when a transaction is submitted."""

    featured = frame.copy()
    featured["hour_of_day"] = featured["step"] % 24
    featured["log_amount"] = np.log1p(featured["amount"])
    denominator = featured["origin_balance_before"].replace(0, np.nan)
    featured["amount_to_origin_balance"] = (featured["amount"] / denominator).fillna(0.0)
    return featured


def add_investigation_only_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive post-event evidence that is forbidden in the Phase 2 model."""

    featured = frame.copy()
    featured["day_index"] = featured["step"] // 24
    featured["origin_balance_error"] = (
        featured["origin_balance_before"]
        - featured["amount"]
        - featured["origin_balance_after"]
    ).abs()
    featured["destination_balance_error"] = (
        featured["destination_balance_before"]
        + featured["amount"]
        - featured["destination_balance_after"]
    ).abs()
    featured["origin_emptied"] = (
        (featured["origin_balance_before"] > 0) & (featured["origin_balance_after"] == 0)
    ).astype("int8")
    return featured


def add_foundation_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper for prepared investigation datasets.

    Model code must use ``build_model_matrix`` rather than consuming this output
    wholesale because it includes post-event investigation fields.
    """

    return add_investigation_only_features(add_safe_model_features(frame))


def build_model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the explicit, scoring-time-safe Phase 2 feature columns."""

    missing = sorted(set(ML_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise FeatureContractError(f"Missing required ML features: {', '.join(missing)}")
    return frame.loc[:, ML_FEATURE_COLUMNS].copy()


def extract_target(frame: pd.DataFrame) -> pd.Series:
    if ML_TARGET_COLUMN not in frame:
        raise FeatureContractError(f"Missing ML target: {ML_TARGET_COLUMN}")
    return frame[ML_TARGET_COLUMN].copy()
