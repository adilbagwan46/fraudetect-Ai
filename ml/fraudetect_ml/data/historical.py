from __future__ import annotations

from typing import Protocol

import pandas as pd


class CausalHistoricalFeatureBuilder(Protocol):
    """Extension boundary for history features derived from strictly prior events.

    Implementations may calculate velocity, rolling amount statistics, elapsed
    time, unique counterparties, and first-time-counterparty indicators. They
    must fit state on training data only and must never expose same-step or future
    transactions to the transaction being transformed.
    """

    def fit(self, frame: pd.DataFrame) -> CausalHistoricalFeatureBuilder: ...

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...


def strictly_prior_events(frame: pd.DataFrame, *, current_step: int) -> pd.DataFrame:
    """Return events observable before a step; same-step events are intentionally excluded."""

    return frame.loc[frame["step"] < current_step].copy()
