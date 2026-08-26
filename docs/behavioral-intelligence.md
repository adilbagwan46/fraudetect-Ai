# Behavioral Intelligence

## Purpose

Phase 3 adds a deterministic Behavioral Context Engine beside the frozen Phase 2A model. It
answers a narrow investigation question: what had been observed for this PaySim origin before
the transaction's step? Behavioral deviation is investigation context, not a fraud prediction,
and it never changes model inputs, probability, calibration, thresholds, or operating policy.

## Causal boundary

For a current transaction at step `S`, eligible history is exactly:

```text
historical.step < S
```

Transactions at step `S` are excluded because PaySim provides no trustworthy within-step event
ordering. Future steps are excluded. The current transaction is therefore excluded from its own
baselines. Tests also mutate future values and prove that earlier contexts remain identical.

Fraud labels (`is_fraud` and `is_flagged_fraud`) are neither loaded into nor stored by the
behavioral index. No previous-fraud aggregates exist. This keeps the context computable without
knowing transaction outcomes.

## BehavioralContext contract

The public typed context contains only aggregate facts:

- `history_available` and an explicit availability explanation;
- prior transaction count and total amount;
- prior average, median, and maximum amount when history exists;
- current amount ratios to those baselines, an empirical prior percentile, and whether it exceeds
  the prior maximum;
- counts and total amounts in the previous 1, 6, and 24 PaySim steps;
- the step gap since the most recent eligible transaction;
- count of prior transactions with the current type and whether the type is new in available
  prior history.

Ratios whose historical denominator is zero are `null`; they are never represented as infinity.
When there is no history, amount baselines and amount comparisons are `null` rather than invented
zeros. All numeric outputs reject NaN and infinity.

### Window semantics

For a window of `N` steps, the engine includes events satisfying:

```text
current.step - N <= historical.step < current.step
```

The 1-step window captures the immediately preceding PaySim step, 6 steps captures short-range
simulation activity, and 24 steps captures a broader simulation cycle. Although a PaySim step is
often described as an hour in the simulator, these fields deliberately use “steps,” not real-world
clock-time claims.

## Architecture and efficiency

`BehaviorHistoryProvider` separates aggregation from storage. Phase 3 supplies a
`SQLitePaySimHistoryProvider`; a later deployment could implement the same boundary with a
production transaction-history store.

For offline tests and representative examples, `iter_causal_behavioral_contexts` consumes events
already ordered by nondecreasing step. It calculates every context in a step before adding any
event from that step to origin state, then updates state once for the complete step. This is a
single chronological pass and mechanically preserves same-step exclusion.

Run:

```bash
.venv/bin/python scripts/build_behavior_history.py
```

The builder makes one chronological, chunked pass over the prepared train, validation, and test
CSVs and writes `artifacts/behavioral/history.sqlite`. The generated database is Git-ignored. It
stores only the internal transaction reference, step, type, amount, pre-transaction origin
balance, and internal origin key. It does not store labels, destination identity, post-event
balances, device/IP enrichment, or raw rows.

The database has a primary-key lookup for transaction reference and an index on
`(origin_key, step, transaction_reference)`. Building is `O(N log N)` because of index creation and
uses bounded CSV chunks. Each investigation is an indexed reference lookup plus an indexed scan
of that origin's strictly prior history: approximately `O(log N + H)`, where `H` is the eligible
history for one origin. It never rescans all 6,362,620 rows per request and does not precompute a
massive context snapshot for every transaction.

## API behavior

`POST /api/v1/risk/predict` is unchanged. It accepts manual scoring fields and returns the frozen
model output with Phase 2B evidence; it does not load behavioral history.

`POST /api/v1/risk/investigate` supports two mutually exclusive request modes:

```json
{"transaction_reference": "TX-000000002"}
```

or the existing flat manual transaction fields. A reference is accepted only by the investigation
endpoint. It is resolved internally and is never echoed. Unknown references return `404`; a
missing/unreadable generated index returns `503`. Manual transactions are not assigned fabricated
identity or history and return `history_available: false`.

The investigation response includes `behavioral_context` and may include deterministic behavioral
evidence. Model prediction, reference-population evidence, and origin-specific behavioral
deviation remain explicitly separate concepts.

## Privacy and future LLM boundary

Raw PaySim customer/destination identifiers and transaction references are internal lookup keys.
Public behavioral responses contain no identifiers and no raw history. A future LLM may receive
only model output, deterministic evidence, approved aggregate reference context, and this aggregate
behavioral context. It must not receive `nameOrig`, `nameDest`, raw transaction IDs, raw history, or
unrestricted dataset access.

## Limitations

- PaySim is synthetic; observed behavior is not evidence of real customer behavior.
- `is_new_transaction_type_for_origin` means new within available earlier PaySim history, not the
  customer's first-ever real-world transaction.
- Same-step ordering is intentionally unavailable, so potentially earlier same-step rows are
  excluded.
- The local generated SQLite index is a practical buildathon lookup, not a proposed production
  database architecture.
- Destination/counterparty aggregates were intentionally deferred. They would enlarge the index
  and introduce ambiguity without materially improving this focused first behavioral layer.
- Behavioral evidence is descriptive. A deviation does not prove fraud and does not alter the
  frozen Phase 2A model.
