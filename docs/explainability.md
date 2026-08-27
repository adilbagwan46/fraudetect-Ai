# Evidence & Explainability Engine

## Responsibility boundary

Fraudetect separates three responsibilities:

1. The frozen Phase 2A model produces fraud probability, classification, threshold, operating mode, and risk level.
2. The deterministic Phase 2B engine produces factual evidence around that output.
3. The optional LLM Copilot may summarize the sanitized typed investigation context but does not calculate fraud risk or receive authority to invent evidence.

Evidence is associative context, not causal attribution. The engine deliberately uses phrases such as “risk signal” and “historical training prevalence,” never “this caused fraud.”

## Methodology

The selected hybrid approach combines:

- **Global model context:** average-precision permutation importance on a deterministic 250,000-row sample from the approved training split. Three fixed-seed repeats measure global predictive reliance on the original six model inputs.
- **Local factual context:** deterministic rules compare the submitted amount, amount-to-balance ratio, transaction type, and hour with stored training-reference statistics.
- **Model-risk context:** reports whether the unchanged calibrated probability exceeds the unchanged BALANCED threshold and by what margin.

SHAP was not added. It would add a heavy dependency and more complex behavior around the calibrated preprocessing/model wrapper, while the product currently needs stable factual evidence rather than a potentially overinterpreted local attribution decomposition.

## Reference profile

`artifacts/models/<model-version>/reference-profile.json` is produced by `scripts/build_reference_profile.py`.

The active profile is `reference-c528b8642a96` and contains:

- PaySim dataset SHA-256 and frozen model version
- explicit source boundary: training steps 1–323
- 4,463,587 reference rows and 3,643 fraud labels
- overall amount, origin-balance, and amount-to-balance distributions
- percentiles from 1% through 99.9%
- amount and ratio distributions by transaction type
- training row share and historical fraud prevalence by transaction type
- hourly row share and historical fraud prevalence
- transaction-type/hour activity distributions
- deterministic global permutation importance

Validation and test data are rejected as reference-profile sources. The profile explicitly records `validation_used: false` and `test_used: false`. It stores aggregates only, not raw training transactions.

Global normalized positive permutation importance on the training sample:

| Feature | Importance |
|---|---:|
| amount_to_origin_balance | 0.432228 |
| transaction_type | 0.267853 |
| amount | 0.181422 |
| origin_balance_before | 0.076846 |
| hour_of_day | 0.041651 |
| log_amount | 0.000000 |

These values describe global model reliance on the sampled synthetic training reference. They do not prove causality and are not direct per-transaction contributions.

## Evidence types

- `MODEL_RISK`: frozen score versus active threshold
- `AMOUNT_CONTEXT`: estimated global and transaction-type amount percentiles
- `BALANCE_CONTEXT`: amount as a fraction of recorded pre-transaction balance, with safe zero handling
- `TRANSACTION_TYPE_CONTEXT`: historical fraud prevalence in training data
- `TIME_CONTEXT`: hourly training prevalence and transaction-type activity share

Historical prevalence is always labeled as population context and never presented as the transaction’s probability.

## Deterministic prioritization

The engine creates one valid candidate for each evidence type, assigns a severity and relevance score, then sorts by:

1. severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`)
2. relevance score
3. stable evidence ID

At most five items are returned. Amount and `log_amount` are intentionally combined to avoid redundant evidence. Low-risk transactions receive proportional INFO evidence and are not forced into suspicious narratives.

## Risk and action policy

Phase 2B preserves Phase 2A behavior:

| Risk level | Definition | Simulated recommendation |
|---|---|---|
| LOW | probability below BALANCED threshold | NORMAL_PROCESSING |
| MEDIUM | probability at/above BALANCED but below HIGH_PRECISION threshold | MANUAL_REVIEW |
| HIGH | probability at/above HIGH_PRECISION threshold | HOLD_FOR_INVESTIGATION |

`fraud_probability` is the calibrated frozen model score. `fraud_prediction` is the thresholded BALANCED classification. `risk_level` is a presentation/policy band. Evidence does not change any of them.

Referenced Phase 3 investigations may add `BEHAVIORAL_CONTEXT` evidence derived from aggregate
origin history at strictly earlier PaySim steps. This evidence describes deviation from prior
observations; it is separate from model prediction and training-reference population context. The
prediction endpoint and Phase 2B evidence behavior remain unchanged when no behavioral context is
provided. See [behavioral-intelligence.md](behavioral-intelligence.md).

Referenced Phase 5 investigations also return a separate list of `RELATIONSHIP_CONTEXT` evidence.
It describes strictly earlier origin-destination history, amount comparison, novelty, and sparse
baselines without changing or competing with Phase 2B model evidence. See
[relationship-intelligence.md](relationship-intelligence.md).

## Investigation context

`POST /api/v1/risk/investigate` returns a typed context containing:

- validated raw scoring input
- server-derived features
- frozen model output
- active threshold and operating mode
- deterministic evidence
- reference-profile identity and approved aggregate context

This is the sanitizer's source context. The Copilot receives a smaller positive-selection projection with no unrestricted dataset access, arbitrary model internals, identifiers, raw history, or application state.

## Limitations

- PaySim and the reference profile are synthetic, not merchant production data.
- Percentiles are interpolated from stored reference quantiles rather than raw rows at request time.
- Permutation importance can distribute importance unpredictably between correlated `amount` and `log_amount`.
- Rules identify associations and unusual characteristics, not causal reasons.
- PaySim relationship history is synthetic and does not establish real identity or network risk.
