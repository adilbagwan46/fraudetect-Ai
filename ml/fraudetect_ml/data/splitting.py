from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplits:
    if frame.empty:
        raise ValueError("Cannot split an empty dataset")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test split")

    ordered = frame.sort_values(["step", "transaction_id"], kind="stable").reset_index(drop=True)
    step_counts = ordered.groupby("step", sort=True).size()
    if len(step_counts) < 3:
        raise ValueError("Dataset needs at least three distinct steps for temporal splits")

    cumulative_rows = step_counts.cumsum().tolist()
    row_count = len(ordered)

    def nearest_boundary(target_rows: float, minimum_groups: int, maximum_groups: int) -> int:
        """Return number of complete step groups closest to the target row count."""

        return min(
            range(minimum_groups, maximum_groups + 1),
            key=lambda group_count: (
                abs(cumulative_rows[group_count - 1] - target_rows),
                group_count,
            ),
        )

    train_groups = nearest_boundary(row_count * train_fraction, 1, len(step_counts) - 2)
    validation_groups = nearest_boundary(
        row_count * (train_fraction + validation_fraction),
        train_groups + 1,
        len(step_counts) - 1,
    )
    train_end = cumulative_rows[train_groups - 1]
    validation_end = cumulative_rows[validation_groups - 1]

    splits = TemporalSplits(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )
    if splits.train["step"].max() > splits.validation["step"].min():
        raise ValueError("Training split occurs after validation data")
    if splits.validation["step"].max() > splits.test["step"].min():
        raise ValueError("Validation split occurs after test data")
    step_sets = [set(split["step"]) for split in (splits.train, splits.validation, splits.test)]
    if any(step_sets[left] & step_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("A time step was assigned to more than one split")
    return splits
