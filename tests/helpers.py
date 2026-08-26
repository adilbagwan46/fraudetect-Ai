from typing import Any

from ml.fraudetect_ml.data.contracts import ML_FEATURE_COLUMNS


def reference_profile() -> dict[str, Any]:
    percentiles = {
        "0.010": 1.0,
        "0.050": 5.0,
        "0.250": 25.0,
        "0.500": 50.0,
        "0.750": 75.0,
        "0.900": 90.0,
        "0.950": 95.0,
        "0.990": 99.0,
        "0.995": 99.5,
        "0.999": 99.9,
    }
    transaction_types = {}
    type_hours = {}
    for transaction_type in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"):
        prevalence = 0.003 if transaction_type == "TRANSFER" else 0.0001
        transaction_types[transaction_type] = {
            "rows": 1_000,
            "share": 0.2,
            "fraud_rows": 20 if transaction_type == "TRANSFER" else 1,
            "historical_fraud_prevalence": prevalence,
            "amount": {"count": 1_000, "mean": 50.0, "percentiles": percentiles},
            "amount_to_origin_balance": {
                "count": 1_000,
                "mean": 0.2,
                "percentiles": percentiles,
            },
        }
        type_hours[transaction_type] = {
            str(hour): {
                "rows": 40,
                "share_within_type": 1 / 24,
                "fraud_rows": 1,
                "historical_fraud_prevalence": 0.001,
            }
            for hour in range(24)
        }
    return {
        "reference_profile_version": "reference-test",
        "source_boundary": {
            "split": "train",
            "min_step": 1,
            "max_step": 10,
            "validation_used": False,
            "test_used": False,
        },
        "statistics": {
            "historical_fraud_prevalence": 0.001,
            "amount": {"count": 5_000, "mean": 50.0, "percentiles": percentiles},
            "origin_balance_before": {
                "count": 5_000,
                "mean": 100.0,
                "percentiles": percentiles,
            },
            "amount_to_origin_balance": {
                "count": 5_000,
                "mean": 0.2,
                "percentiles": percentiles,
            },
            "transaction_types": transaction_types,
            "hours": {
                str(hour): {
                    "rows": 200,
                    "share": 1 / 24,
                    "fraud_rows": 1,
                    "historical_fraud_prevalence": 0.001,
                }
                for hour in range(24)
            },
            "transaction_type_hours": type_hours,
        },
        "global_model_importance": {
            "features": {
                feature: {"normalized_positive_importance": 1 / len(ML_FEATURE_COLUMNS)}
                for feature in ML_FEATURE_COLUMNS
            }
        },
    }
