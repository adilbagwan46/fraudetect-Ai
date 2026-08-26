from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.fraudetect_ml.data.contracts import CANONICAL_RENAME, PAYSim_REQUIRED_COLUMNS


class DatasetValidationError(ValueError):
    """Raised when a source file cannot satisfy the transaction contract."""


def load_paysim(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DatasetValidationError(f"Dataset file does not exist: {path}")

    frame = pd.read_csv(path)
    missing = sorted(set(PAYSim_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetValidationError(f"Missing required PaySim columns: {', '.join(missing)}")
    if frame.empty:
        raise DatasetValidationError("Dataset contains no rows")

    frame = frame.loc[:, PAYSim_REQUIRED_COLUMNS].rename(columns=CANONICAL_RENAME).copy()
    numeric_columns = [
        "step",
        "amount",
        "origin_balance_before",
        "origin_balance_after",
        "destination_balance_before",
        "destination_balance_after",
        "is_fraud",
        "is_flagged_fraud",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_numeric = frame[numeric_columns].isna().any(axis=1)
    if invalid_numeric.any():
        raise DatasetValidationError(
            f"Found {int(invalid_numeric.sum())} rows with invalid required numeric values"
        )
    if (frame["step"] < 0).any() or (frame["amount"] < 0).any():
        raise DatasetValidationError("step and amount must be non-negative")
    for label in ("is_fraud", "is_flagged_fraud"):
        values = set(frame[label].astype(int).unique())
        if not values.issubset({0, 1}):
            raise DatasetValidationError(f"{label} must contain binary values only")

    frame["step"] = frame["step"].astype("int64")
    frame["is_fraud"] = frame["is_fraud"].astype("int8")
    frame["is_flagged_fraud"] = frame["is_flagged_fraud"].astype("int8")
    frame["transaction_id"] = [f"TX-{index + 1:09d}" for index in range(len(frame))]
    return frame

