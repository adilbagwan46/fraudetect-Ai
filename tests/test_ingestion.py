from pathlib import Path

import pandas as pd
import pytest

from ml.fraudetect_ml.data.ingestion import DatasetValidationError, load_paysim


def test_load_paysim_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"step": [1], "amount": [20]}).to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="Missing required"):
        load_paysim(path)


def test_load_paysim_normalizes_contract(tmp_path: Path) -> None:
    path = tmp_path / "valid.csv"
    pd.DataFrame(
        [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 20.5,
                "nameOrig": "C1",
                "oldbalanceOrg": 100,
                "newbalanceOrig": 79.5,
                "nameDest": "M1",
                "oldbalanceDest": 0,
                "newbalanceDest": 20.5,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            }
        ]
    ).to_csv(path, index=False)

    loaded = load_paysim(path)

    assert loaded.loc[0, "transaction_id"] == "TX-000000001"
    assert loaded.loc[0, "customer_id"] == "C1"
    assert loaded.loc[0, "is_fraud"] == 0

