# Fraudetect AI — Project Specification

## Product scope

Fraudetect AI is a payment-fraud risk detection and investigation platform for merchant risk analysts. It evaluates a payment with a measurable ML model, adds behavioral and relationship context, and lets an evidence-grounded AI investigator explain what deserves human attention.

**Product principle:** ML detects the risk. AI investigates the evidence. Humans retain control.

The one-week build covers payment fraud only. It does not attempt chargeback management, AML, account takeover, SIEM, real payment blocking, or autonomous financial decisions.

## Primary analyst workflow

1. An analyst opens the risk overview and filters suspicious transactions.
2. The analyst selects one transaction and sees its model probability, configured risk level, and contributing factors.
3. Behavioral and local graph evidence reveals relevant history, shared identifiers, and connected risky entities.
4. The AI investigator retrieves evidence through bounded tools and returns validated structured findings, uncertainty, and a policy-constrained recommendation.
5. The analyst makes the decision; the investigation inputs and outputs are retained in an audit record.

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

For repository tests and the UI smoke demo, a small deterministic generated dataset is provided by script. It is not used for final reported model claims.

## Architecture

```text
PaySim CSV / generated demo events
             |
       validated ingestion
             |
   normalized transaction contract
             |
      feature engineering
         /          \
 ML risk engine   relationship engine
         \          /
         unified evidence context
                  |
       tool-bounded AI investigator
                  |
       structured investigation result
                  |
        FastAPI + SQLite audit store
                  |
       React analyst workspace
```

This is a modular monolith. Python modules isolate data, risk, relationships, evidence tools, LLM providers, and persistence without deployment-heavy microservices.

## Component boundaries

- **Data pipeline:** validates PaySim-compatible inputs, normalizes names/types, enriches demo identifiers, engineers non-model-specific features, and creates chronological train/validation/test manifests.
- **ML risk engine (Phase 2):** compares logistic regression with HistGradientBoosting, chooses thresholds on validation data, serializes the full preprocessing/model pipeline, and exposes calibrated probability separately from risk level.
- **Relationship engine (Phase 3):** builds an in-memory/local NetworkX graph and returns bounded neighborhoods and aggregate signals. No graph database is needed.
- **Evidence layer (Phase 3/4):** turns repository results into typed, citation-addressable evidence objects.
- **AI investigator (Phase 4):** invokes only allow-listed tools, validates structured output, rejects unsupported evidence references, and degrades to an unavailable state without affecting ML results.
- **API:** versioned FastAPI routes with Pydantic request/response contracts and centralized exception handling.
- **Storage:** SQLite stores normalized demo transactions, investigations, evidence snapshots, and audit events. Model artifacts and evaluation reports remain versioned files.
- **Frontend:** React/Vite/TypeScript analyst workspace with focused views for overview, transactions, investigation, local graph, and model evaluation.

## Planned API surface

- `GET /api/v1/health`
- `GET /api/v1/dataset/status`
- `GET /api/v1/transactions` and `GET /api/v1/transactions/{id}`
- `POST /api/v1/risk/predict`
- `GET /api/v1/transactions/{id}/evidence`
- `GET /api/v1/entities/{id}/connections`
- `POST /api/v1/investigations`
- `GET /api/v1/investigations/{id}`
- `GET /api/v1/model/evaluation`

Only health and dataset status are implemented in Phase 1.

## Risk and action semantics

Model probability is the estimator output. Risk score is its 0–100 presentation form. Risk level is a configurable LOW/MEDIUM/HIGH band selected after validation analysis. Recommended actions are simulated policy suggestions: normal processing, manual review, or hold for investigation. They do not trigger payment actions.

## Reliability and security

- Validate input columns, numeric ranges, IDs, and response schemas.
- Keep secrets in environment variables and provide `.env.example`.
- Never send raw bulk datasets to an LLM.
- Persist evidence snapshots used for an investigation.
- Mark absent history or graph data explicitly.
- Treat LLM timeout, provider failure, and malformed output as isolated degradation.
- Use generated/demo data only; no offensive fraud guidance or live payment integration.

## Explicit non-goals for the buildathon

- Multiple agents, vector databases, RAG infrastructure, microservices, Kubernetes, live blocking, graph databases, online model learning, and production-scale streaming.
- Claims that synthetic enrichment proves device/IP fraud detection performance.
- Tuning against the held-out test set.

