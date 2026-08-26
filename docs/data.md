# Data Card

## Primary source

PaySim is the selected public transaction dataset. It is simulator-generated mobile-money data informed by aggregated real transaction patterns; it is not raw Razorpay or merchant data. Required source fields are validated before any processing.

The expected local filename is `data/raw/paysim.csv`. The repository does not download, bundle, or redistribute it. The source dataset/publication terms remain applicable; the repository MIT license covers code only.

| Source field | Canonical meaning |
|---|---|
| `step` | simulated time step (one hour in PaySim) |
| `type` | transaction type |
| `amount` | transaction amount |
| `nameOrig` | source/customer account identifier |
| `oldbalanceOrg`, `newbalanceOrig` | source balances before/after |
| `nameDest` | destination/counterparty identifier |
| `oldbalanceDest`, `newbalanceDest` | destination balances before/after |
| `isFraud` | supervised fraud label |
| `isFlaggedFraud` | PaySim rule flag; not the model target |

## Generated fields

- `transaction_id`: stable sequential ID for one prepared source file.
- `device_id` and `ip_id`: deterministic demo-only hash buckets based on customer ID, namespace, and `FRAUDETECT_ENRICHMENT_SEED`.
- `data_provenance`: explicit enrichment marker.
- time, logarithmic amount, balance-error, amount-ratio, and emptied-account foundation features.

The device/IP generator never reads `is_fraud`. It uses fixed configured bucket counts (`FRAUDETECT_DEVICE_BUCKETS` and `FRAUDETECT_IP_BUCKETS`) rather than whole-dataset customer cardinality. Enrichment occurs after splitting and is excluded from the Phase 2 model. Hash collisions only demonstrate graph mechanics and cannot support claims about genuine device/IP data.

## Phase 2 feature contract

Only these scoring-time fields may enter the initial model:

- `transaction_type`
- `amount`
- `origin_balance_before`
- `hour_of_day`
- `log_amount`
- `amount_to_origin_balance`

The target, rule flag, identifiers, destination balance, post-event balances, balance errors, account-empty indicator, synthetic relationships, provenance, absolute day, and raw step are explicitly excluded. `destination_balance_before` remains excluded because its availability is not justified for the primary merchant scoring workflow.

## Split policy

Rows are stably ordered by `step` and transaction ID. Boundaries closest to the target 70/15/15 row proportions are selected only between complete steps. Every transaction from one step belongs to exactly one split. Because steps are atomic, actual fractions may differ slightly; the manifest records exact rows, fractions, fraud counts, step counts, and time bounds.

The test split is not used for preprocessing decisions, model choice, hyperparameter selection, or thresholds.

## Causal history requirement

Future behavioral features must be computed from events with `historical.step < current.step`. Same-step and future events are unavailable when a transaction is scored and are forbidden. Preprocessors and stateful historical builders fit on training data only; frozen state transforms validation and test data in chronological order. Candidate features include prior-window counts, past-only amount statistics, elapsed time, prior unique counterparties, and first-time-counterparty status.

## Reproducibility

Use `.venv/bin/python scripts/prepare_data.py`. The output manifest records a source hash, source kind, field provenance, explicit ML contract, split configuration, row and fraud counts, time bounds, and produced columns. Large/raw/prepared data is ignored by Git.

`scripts/generate_demo_data.py` produces a deterministic PaySim-compatible development fixture. It is labeled `generated_demo_only` at preparation time and is excluded from final evaluation claims.
