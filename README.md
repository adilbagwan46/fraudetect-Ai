# Fraudetect AI

**AI-Powered Fraud Risk Detection & Investigation Platform**

> ML detects the risk. AI investigates the evidence. Humans retain control.

Fraudetect AI is a focused payment-fraud analyst workspace being built for the Razorpay AI Buildathon (Track 2 — AI Risk Manager). It combines measurable supervised fraud detection, behavioral signals, relationship context, and a tool-bounded AI investigation layer. It is not a transaction-blocking system and does not make autonomous financial decisions.

## Current status

The portfolio build is feature-complete through the Phase 10 validation checkpoint. It includes a frozen and held-out-tested fraud model, deterministic evidence, causal behavioral and relationship context, an optional grounded real-LLM provider, deterministic fallback, immutable analyst cases, an append-only audit timeline, operational metrics, and an isolated demo workflow. See [docs/demo-guide.md](docs/demo-guide.md), [docs/llm-copilot.md](docs/llm-copilot.md), [docs/analyst-workflow.md](docs/analyst-workflow.md), and [docs/auditability.md](docs/auditability.md).

The frozen Phase 2A model, Phase 2B evidence, Phase 3 causal behavior, and deterministic Phase 5 relationship provider remain the sources of truth. The Copilot only summarizes approved context and never participates in prediction. The current model is an honest baseline evaluated only on public synthetic PaySim data; it is not evidence of production merchant performance.

## Why this is not just a classifier

```text
Transaction
    -> frozen ML probability + ML risk level
    -> deterministic evidence
    -> causal behavior + relationship context
    -> privacy-sanitized immutable case snapshot
    -> grounded real-LLM brief or deterministic fallback
    -> human workflow decision
    -> append-only audit timeline
```

- **ML risk engine:** owns measurable fraud prediction.
- **Behavioral intelligence:** compares the event with prior customer behavior.
- **Relationship intelligence:** summarizes causal origin-destination history and aggregate network breadth without claiming hidden or risky identities.
- **AI investigator:** retrieves bounded evidence, distinguishes facts from interpretation, and reports uncertainty.
- **Human analyst:** owns the final action.

## Dataset and provenance

The primary model dataset is **PaySim**, public simulator-generated mobile-money transaction data. It provides transaction time steps, account IDs, balances, amounts, types, and fraud labels. Raw data is not committed to this repository.

PaySim does not contain device or IP fields. The pipeline can add deterministic synthetic device/IP identifiers for demonstrating relationship workflows. This enrichment:

- is generated from account IDs and a documented seed;
- never reads the fraud label;
- is reproducible;
- must not be interpreted as real merchant evidence or model performance.

The repository also includes a small deterministic data generator for local development. Its output is demo-only and must not be used for final evaluation claims.

## Repository structure

```text
backend/                 FastAPI application and typed API contracts
frontend/                React/Vite/TypeScript analyst UI
ml/fraudetect_ml/data/   ingestion, enrichment, features, splitting, pipeline
scripts/                 reproducible data commands
tests/                   regression, privacy, lifecycle, causality, and provider tests
docs/                    specification, plan, and decisions
data/                    ignored raw and prepared datasets
artifacts/               ignored model/evaluation artifacts
```

## Local setup

Prerequisites: Python 3.11+ and Node.js 20+.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,ml]"
cp .env.example .env
```

To enable the optional real OpenAI provider, install the additional server dependency:

```bash
.venv/bin/python -m pip install -e ".[dev,ml,llm]"
```

Generate a small local dataset and prepare it:

```bash
.venv/bin/python scripts/generate_demo_data.py
.venv/bin/python scripts/prepare_data.py \
  --input data/raw/demo_transactions.csv \
  --source-kind generated_demo_only
```

For genuine PaySim evaluation, place the downloaded CSV at exactly `data/raw/paysim.csv` and run:

```bash
.venv/bin/python scripts/prepare_data.py
```

The preparation command validates the original PaySim schema and writes `train.csv`, `validation.csv`, `test.csv`, and `manifest.json` under `data/processed/`. The manifest records the source SHA-256, provenance, exact row/fraud/step counts and fractions, feature contract, complete-step boundaries, and held-out-test policy.

The genuine PaySim file is not bundled or downloaded automatically. Its source terms apply independently of this repository's MIT-licensed code. Do not commit or redistribute the raw dataset here.

Start the API:

```bash
make normal
```

Use normal mode for everyday development, full-index PaySim case creation, and manual testing.
API documentation is available at `http://localhost:8000/docs`.

After training a fresh model bundle, generate its training-only evidence profile with:

```bash
.venv/bin/python scripts/build_reference_profile.py
```

The profile records deterministic training-reference distributions and global permutation importance without storing raw training rows. See [docs/explainability.md](docs/explainability.md).

Build the ignored, label-free behavioral history index after preparing genuine PaySim:

```bash
.venv/bin/python scripts/build_behavior_history.py
```

This performs one chunked pass over prepared splits and enables indexed, causal investigation-time lookup without per-request full-dataset scans. See [docs/behavioral-intelligence.md](docs/behavioral-intelligence.md).

Build the separate ignored, label-free relationship index:

```bash
.venv/bin/python scripts/build_relationship_history.py
```

It stores only the internal fields needed for indexed relationship aggregation, never loads fraud labels, and enforces `historical.step < current.step` at lookup. See [docs/relationship-intelligence.md](docs/relationship-intelligence.md).

Start the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

The development server proxies `/api` to `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` at build time only when the deployed API is hosted separately.

To populate a self-contained local showcase queue from deterministic genuine PaySim selections,
first build the full behavioral and relationship indexes, then run:

```bash
make demo-cases
```

The command selects three presentation scenarios without using labels, then creates an isolated
case store plus minimal PaySim-backed behavioral and relationship index subsets under ignored
`artifacts/demo/`. It refuses to overwrite an existing demo. Only use
`.venv/bin/python scripts/seed_demo_cases.py --force` when replacement is intentional. Launch the
API against the isolated showcase databases with `make demo`; the exact sequence is in
[docs/demo-guide.md](docs/demo-guide.md). Transaction references and raw identities remain internal
to the ignored local indexes and never enter public case snapshots. Use showcase mode for the
curated three-case presentation rather than general case creation.

## Environment variables

See `.env.example`. The Copilot defaults to deterministic fallback mode. Real mode requires
`FRAUDETECT_LLM_ENABLED=true`, an explicit `FRAUDETECT_LLM_PROVIDER` selection, and the matching
server-side `OPENAI_API_KEY` or `GEMINI_API_KEY`. A key alone never enables a provider. No API key is
needed for development, testing, prediction, evidence, behavior, or fallback reports. Provider
secrets are never sent to the frontend. Readiness reports configuration state only and does not
call either external provider.

## Testing

```bash
.venv/bin/pytest -q
.venv/bin/ruff check backend ml scripts tests
cd frontend && npm run build
```

Tests cover the data contract, frozen ML regression behavior, causal intelligence, Copilot grounding, case lifecycle and persistence, priority policy, status validation, and identifier/privacy boundaries.

## Technology stack

- Python 3.11+, FastAPI, Pydantic, pandas, NumPy, scikit-learn, and joblib
- SQLite for local behavioral, relationship, case, and audit demonstration stores
- React, TypeScript, and Vite for the analyst workspace
- Pytest and Ruff for regression and static verification
- Optional OpenAI Responses API and Google Gen AI Gemini adapters with Pydantic structured output

The Phase 2 baseline is restricted to `transaction_type`, `amount`, `origin_balance_before`, `hour_of_day`, `log_amount`, and `amount_to_origin_balance`. Labels, identifiers, post-event balances, balance-error fields, enrichment, absolute simulation day, and destination balance are rejected from the model matrix by construction.

## Model evaluation

Phase 2 compared weighted and unweighted logistic-regression and HistGradientBoosting candidates. Model and threshold selection used training and validation only. The frozen choice was evaluated once on the chronological held-out test set with:

- precision, recall, and F1;
- confusion matrix, false positives, and false negatives;
- actual class prevalence per split;
- estimated false-positive cost using explicitly labeled manual-review and friction assumptions.

The Phase 2A winner is the training-weighted HistGradientBoosting candidate, selected by validation BALANCED-policy F1. At its frozen threshold, held-out test precision is 95.38%, recall 82.89%, and F1 88.70%. PaySim's test period has a materially higher fraud prevalence than training and validation, so these values must be interpreted with that temporal shift and the dataset's synthetic origin in mind. Full candidate, operating-point, and confusion-matrix results are in [docs/evaluation.md](docs/evaluation.md).

No fabricated or demo-generator metrics are presented as final results.

## Engineering decisions

The rationale for model/LLM separation, PaySim, synthetic enrichment, chronological splitting, the modular monolith, SQLite, and evidence-addressable AI output is documented in [docs/engineering-decisions.md](docs/engineering-decisions.md).

## Evidence and explainability

`POST /api/v1/risk/predict` returns the frozen model output plus three to five deterministic evidence items. `POST /api/v1/risk/investigate` returns the reusable typed investigation context intended for later LLM summarization. Evidence reports factual associations and reference statistics; it does not claim causality or replace the model decision.

The investigation endpoint accepts either the existing manual transaction fields or an exclusive safe internal `transaction_reference`. Referenced investigations add aggregate `behavioral_context`, `relationship_context`, and separate typed relationship evidence; the reference and raw PaySim identities are never returned. Manual investigations do not fabricate identity and explicitly report unavailable history. Prediction requests remain history-free and inexpensive.

`POST /api/v1/risk/investigate/copilot` returns a typed advisory report containing a summary, frozen risk assessment, evidence-linked signals, behavioral and relationship analysis, uncertainties, reversible next steps, mode metadata, safe execution metadata, safe relationship aggregates, and synthetic-data disclosure. The allowlisted provider payload excludes identifiers and raw history. See [docs/llm-copilot.md](docs/llm-copilot.md).

The case API provides `POST/GET /api/v1/cases`, `GET/PATCH /api/v1/cases/{case_id}`, and `POST /api/v1/cases/{case_id}/copilot`. Case priority is a transparent workflow ordering separate from ML risk; analyst disposition is never treated as model ground truth. Case storage contains only a positively selected immutable snapshot and minimal workflow metadata. Case detail includes a decision trace assembled only from stored creation, Copilot-generation, and status-transition timestamps.

`GET /api/v1/system/readiness` exposes safe ready/unavailable states for the frozen model, deterministic evidence, reference profile, behavioral and relationship indexes, case store, and Copilot mode. It never exposes configured paths, credentials, transaction identities, or raw history. If the LLM is disabled or unavailable, it explicitly reports that deterministic fallback remains ready.

`GET /api/v1/system/metrics` returns aggregate case totals, active workload, status, workflow-priority, disposition, and saved Copilot-mode counts. It returns no case IDs, notes, transaction identifiers, raw history, paths, labels, or credentials. `CRITICAL` in this response is a workflow priority and is never an ML risk level.

## Limitations

- PaySim is synthetic and cannot establish real merchant performance.
- Device/IP enrichment demonstrates relationship mechanics, is applied after temporal splitting with fixed configured bucket counts, and is excluded from the Phase 2 model.
- The local deterministic fallback is not an LLM, and real-provider quality depends on external model access and configuration.
- Behavioral histories can be sparse, and PaySim's account structure limits the richness of relationship intelligence.
- Real-provider quality, latency, availability, and cost depend on the selected provider, model, and account configuration.
- Batch CSV preparation is appropriate for the buildathon but not production streaming scale.
- Recommendations are simulated and never execute payment actions.
- Local SQLite case storage has no production authentication, authorization, tenancy, retention, or distributed audit controls.

## Implementation history

See [docs/implementation-plan.md](docs/implementation-plan.md). The repository intentionally stops at final validation; production authentication, distributed storage, streaming, and deployment infrastructure are outside this portfolio build.

## License

Source code is available under the [MIT License](LICENSE). Dataset licensing and attribution remain governed by their original sources.
