from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.fraudetect_ml.data.contracts import ML_FEATURE_COLUMNS, ML_TARGET_COLUMN
from ml.fraudetect_ml.data.features import build_model_matrix, extract_target

TRAINING_COLUMNS = (*ML_FEATURE_COLUMNS, ML_TARGET_COLUMN, "step")


@dataclass(frozen=True)
class FitCalibrationData:
    fit_features: pd.DataFrame
    fit_target: pd.Series
    calibration_features: pd.DataFrame
    calibration_target: pd.Series
    fit_min_step: int
    fit_max_step: int
    calibration_min_step: int
    calibration_max_step: int


def load_prepared_split(path: Path) -> pd.DataFrame:
    """Load only the feature contract, target, and temporal boundary field."""

    frame = pd.read_csv(
        path,
        usecols=TRAINING_COLUMNS,
        dtype={
            "transaction_type": "category",
            "amount": "float64",
            "origin_balance_before": "float64",
            "hour_of_day": "int8",
            "log_amount": "float64",
            "amount_to_origin_balance": "float64",
            ML_TARGET_COLUMN: "int8",
            "step": "int32",
        },
    )
    if frame.empty:
        raise ValueError(f"Prepared split is empty: {path}")
    if frame.loc[:, TRAINING_COLUMNS].isna().any().any():
        raise ValueError(f"Prepared split contains missing model values: {path}")
    if not set(frame[ML_TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Prepared split contains non-binary targets: {path}")
    return frame


def split_fit_calibration(
    training_frame: pd.DataFrame,
    *,
    fit_fraction: float = 0.90,
) -> FitCalibrationData:
    """Create a whole-step, chronological calibration tail inside training."""

    if not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction must be between zero and one")
    ordered = training_frame.sort_values(["step"], kind="stable").reset_index(drop=True)
    step_counts = ordered.groupby("step", sort=True).size()
    if len(step_counts) < 2:
        raise ValueError("Training data needs at least two steps for calibration")
    target_rows = len(ordered) * fit_fraction
    cumulative = step_counts.cumsum()
    candidate_steps = cumulative.iloc[:-1]
    fit_max_step = int((candidate_steps - target_rows).abs().idxmin())
    fit_mask = ordered["step"] <= fit_max_step
    fit_frame = ordered.loc[fit_mask]
    calibration_frame = ordered.loc[~fit_mask]
    if fit_frame.empty or calibration_frame.empty:
        raise ValueError("Chronological fit/calibration split produced an empty partition")
    if fit_frame["step"].max() >= calibration_frame["step"].min():
        raise ValueError("Calibration events must occur strictly after model-fit events")

    return FitCalibrationData(
        fit_features=build_model_matrix(fit_frame),
        fit_target=extract_target(fit_frame),
        calibration_features=build_model_matrix(calibration_frame),
        calibration_target=extract_target(calibration_frame),
        fit_min_step=int(fit_frame["step"].min()),
        fit_max_step=int(fit_frame["step"].max()),
        calibration_min_step=int(calibration_frame["step"].min()),
        calibration_max_step=int(calibration_frame["step"].max()),
    )


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_model_matrix(frame), extract_target(frame)

