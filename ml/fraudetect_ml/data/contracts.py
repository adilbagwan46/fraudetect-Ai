from __future__ import annotations

PAYSim_REQUIRED_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

CANONICAL_RENAME = {
    "nameOrig": "customer_id",
    "oldbalanceOrg": "origin_balance_before",
    "newbalanceOrig": "origin_balance_after",
    "nameDest": "counterparty_id",
    "oldbalanceDest": "destination_balance_before",
    "newbalanceDest": "destination_balance_after",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
    "type": "transaction_type",
}

# Phase 2 deployable baseline. Training code must select these columns explicitly;
# prepared datasets intentionally retain additional investigation and audit fields.
ML_FEATURE_COLUMNS = (
    "transaction_type",
    "amount",
    "origin_balance_before",
    "hour_of_day",
    "log_amount",
    "amount_to_origin_balance",
)

ML_TARGET_COLUMN = "is_fraud"

ML_EXCLUDED_COLUMNS = (
    "is_fraud",
    "is_flagged_fraud",
    "transaction_id",
    "customer_id",
    "counterparty_id",
    "origin_balance_after",
    "destination_balance_before",
    "destination_balance_after",
    "origin_balance_error",
    "destination_balance_error",
    "origin_emptied",
    "device_id",
    "ip_id",
    "data_provenance",
    "day_index",
    "step",
)
