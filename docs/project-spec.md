# Fraudetect AI — Project Specification

## Product scope

Fraudetect AI is a payment-fraud risk detection and investigation platform for merchant risk analysts. It evaluates a payment with a measurable ML model, adds behavioral and relationship context, and lets an evidence-grounded AI investigator explain what deserves human attention.

**Product principle:** ML detects the risk. AI investigates the evidence. Humans retain control.

The one-week build covers payment fraud only. It does not attempt chargeback management, AML, account takeover, SIEM, real payment blocking, or autonomous financial decisions.

## Primary analyst workflow

1. An analyst opens the risk overview and filters suspicious transactions.
2. The analyst selects one transaction and sees its model probability, configured risk level, and contributing factors.
3. Deterministic behavioral and relationship intelligence summarizes strictly earlier aggregate history without exposing identities.
4. The Copilot summarizes a positive-selection sanitized snapshot and returns grounded structured findings, uncertainty, and reversible advisory actions; deterministic fallback remains available offline.
5. The analyst owns the lifecycle decision, while immutable snapshots and server-generated append-only events preserve the factual audit trail.

## Success criteria

- A complete transaction-to-investigation demo works locally without cloud infrastructure.
- Fraud prediction is evaluated on an untouched, time-ordered held-out test set.
- Precision, recall, F1, confusion matrix, false positives, false negatives, and an assumption-based false-positive cost are reproducible.
- Factual AI claims can be traced to retrieved evidence.
- The risk system still works when no LLM is configured.
- Public versus generated data is labeled unambiguously.

## Data strategy

### Primary dataset: PaySim

PaySim is selected because its transaction records include time steps, transaction type, amount, source/destination account identifiers, balances, and fraud labels. That supports real supervised evaluation and customer-level behavioral features better than the commonly used anonymized European card dataset.

PaySim is itself a published simulator-derived dataset rather than raw merchant production data. It must be described as public synthetic financial transaction data. The raw CSV will not be committed. A preparation command will accept a locally downloaded PaySim CSV.

### Deterministic enrichment

PaySim does not provide device IDs, IP addresses, or physical location. Phase 1 adds optional deterministic demo-only device and IP identifiers derived from a documented seed and account identifier. The mapping does **not** use the fraud label. Collisions are deliberate so graph behavior can be demonstrated, but conclusions based on those relationships are demo evidence, not measured real-world performance.

For repository tests and the UI smoke demo, small deterministic generated fixtures are provided by script. They are not used for final reported model claims.

## Architecture

```text
Transaction
    -> frozen ML fraud model
    -> fraud probability + ML risk level
    -> deterministic evidence
    -> causal behavioral intelligence
    -> causal relationship intelligence
    -> privacy-sanitized immutable case snapshot
    -> grounded LLM report or deterministic fallback
    -> human analyst lifecycle
    -> append-only audit timeline
```

This is a modular monolith. Python modules isolate data, risk, relationships, evidence tools, LLM providers, and persistence without deployment-heavy microservices.

## Component boundaries

- **Data pipeline:** validates PaySim-compatible inputs, normalizes names/types, enriches demo identifiers, engineers non-model-specific features, and creates chronological train/validation/test manifests.
- **ML risk engine:** uses a validation-selected, frozen HistGradientBoosting pipeline and exposes calibrated probability separately from presentation risk level.
- **Evidence layer:** creates deterministic, citation-addressable model, amount, balance, type, and time evidence without changing the model output.
- **Behavioral and relationship engines:** query separate local indexes and return bounded identifier-free aggregates using only `historical.step < current.step`.
- **AI investigator:** receives only the positive-selection sanitized context, validates structured output and grounding, and degrades to an explicitly labeled deterministic report without affecting ML results.
- **API:** versioned FastAPI routes with Pydantic request/response contracts and centralized exception handling.
- **Storage:** separate ignored SQLite files support local historical indexes, immutable cases, lifecycle history, and append-only audit events. Model artifacts and evaluation reports remain ignored versioned files.
- **Frontend:** React/Vite/TypeScript analyst workspace for readiness, operational metrics, case queue, evidence, causal context, Copilot output, human decisions, and timeline.

## Implemented API surface

- `GET /api/v1/health`
- `GET /api/v1/dataset/status`
- `POST /api/v1/risk/predict`
- `POST /api/v1/risk/investigate`
- `POST /api/v1/risk/investigate/copilot`
- `POST /api/v1/cases`
- `GET /api/v1/cases` and `GET /api/v1/cases/{case_id}`
- `PATCH /api/v1/cases/{case_id}`
- `POST /api/v1/cases/{case_id}/copilot`
- `GET /api/v1/model/status`
- `GET /api/v1/model/evaluation`
- `GET /api/v1/system/readiness`
- `GET /api/v1/system/metrics`

These routes never execute payment actions. Referenced investigation inputs are resolved internally and omitted from all returned and stored case payloads.

## Risk and action semantics

Model probability is the estimator output. Risk score is its 0–100 presentation form. Runtime LOW/MEDIUM/HIGH risk bands use validation-selected thresholds stored in the frozen model artifact. `FRAUDETECT_LOW_RISK_MAX` and `FRAUDETECT_HIGH_RISK_MIN` are parsed configuration fields but do not control those frozen scoring bands. Recommended actions are simulated policy suggestions: normal processing, manual review, or hold for investigation. They do not trigger payment actions.

## Reliability and security

- Validate input columns, numeric ranges, IDs, and response schemas.
- Keep secrets in environment variables and provide `.env.example`.
- Never send raw bulk datasets to an LLM.
- Persist evidence snapshots used for an investigation.
- Mark absent history or graph data explicitly.
- Treat LLM timeout, provider failure, and malformed output as isolated degradation.
- Use genuine prepared rows from the public synthetic PaySim dataset for final evaluation and the public showcase; generated fixtures support tests and smoke/demo workflows only. No real merchant or Razorpay data is used, and there is no live payment integration.

## Explicit non-goals for the buildathon

- Multiple agents, vector databases, RAG infrastructure, microservices, Kubernetes, live blocking, graph databases, online model learning, and production-scale streaming.
- Claims that synthetic enrichment proves device/IP fraud detection performance.
- Tuning against the held-out test set.
