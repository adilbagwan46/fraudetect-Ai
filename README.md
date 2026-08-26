# Fraudetect AI

**AI-Powered Fraud Risk Detection & Investigation Platform**

> ML detects the risk. AI investigates the evidence. Humans retain control.

Fraudetect AI is a focused payment-fraud analyst workspace being built for the Razorpay AI Buildathon (Track 2 — AI Risk Manager). It combines measurable supervised fraud detection, behavioral signals, relationship context, and a tool-bounded AI investigation layer. It is not a transaction-blocking system and does not make autonomous financial decisions.

## Current status

Phase 2A is complete: the leakage-aware PaySim foundation now powers a calibrated HistGradientBoosting risk model selected against three alternatives using validation data only. The winning model, three validation-selected operating modes, one-shot held-out evaluation, machine-readable artifacts, and typed prediction/evaluation APIs are available locally.

The relationship engine, AI investigator, and full analyst pages remain scheduled for later phases. The current model is an honest baseline evaluated only on public synthetic PaySim data; it is not evidence of production merchant performance.

## Why this is not just a classifier

```text
Transaction -> ML risk probability -> behavioral + graph evidence
            -> evidence-grounded AI investigation -> human decision
```

- **ML risk engine:** owns measurable fraud prediction.
- **Behavioral intelligence:** compares the event with prior customer behavior.
- **Relationship intelligence:** exposes relevant shared devices, IPs, accounts, and risky neighbors.
- **AI investigator:** retrieves bounded evidence, distinguishes facts from interpretation, and reports uncertainty.
- **Human analyst:** owns the final action.

## Dataset and provenance

The primary planned dataset is **PaySim**, public simulator-generated mobile-money transaction data. It provides transaction time steps, account IDs, balances, amounts, types, and fraud labels. Raw data is not committed to this repository.

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
tests/                   critical Phase 1 tests
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
.venv/bin/uvicorn backend.app.main:app --reload --port 8000
```

API documentation is available at `http://localhost:8000/docs`.

Start the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

## Environment variables

See `.env.example`. Risk thresholds are centralized but remain provisional until Phase 2 validation analysis. LLM settings default to disabled; no API key is required for the ML/evidence system.

## Testing

```bash
.venv/bin/pytest -q
.venv/bin/ruff check backend ml scripts tests
cd frontend && npm run build
```

Phase 1 tests cover invalid source schemas, normalized fields, enrichment reproducibility and label independence, feature invariants, chronological split isolation, API health, and missing-manifest fallback.

The Phase 2 baseline is restricted to `transaction_type`, `amount`, `origin_balance_before`, `hour_of_day`, `log_amount`, and `amount_to_origin_balance`. Labels, identifiers, post-event balances, balance-error fields, enrichment, absolute simulation day, and destination balance are rejected from the model matrix by construction.

## Evaluation plan

Phase 2 will compare a logistic-regression baseline with HistGradientBoosting. Model and threshold selection use training and validation only. The frozen choice will be evaluated once on the chronological held-out test set with:

- precision, recall, and F1;
- confusion matrix, false positives, and false negatives;
- actual class prevalence per split;
- estimated false-positive cost using explicitly labeled manual-review and friction assumptions.

The Phase 2A winner is the training-weighted HistGradientBoosting candidate, selected by validation BALANCED-policy F1. At its frozen threshold, held-out test precision is 95.38%, recall 82.89%, and F1 88.70%. PaySim's test period has a materially higher fraud prevalence than training and validation, so these values must be interpreted with that temporal shift and the dataset's synthetic origin in mind. Full candidate, operating-point, and confusion-matrix results are in [docs/evaluation.md](docs/evaluation.md).

No fabricated or demo-generator metrics are presented as final results.

## Engineering decisions

The rationale for model/LLM separation, PaySim, synthetic enrichment, chronological splitting, the modular monolith, SQLite, and evidence-addressable AI output is documented in [docs/engineering-decisions.md](docs/engineering-decisions.md).

## Limitations

- PaySim is synthetic and cannot establish real merchant performance.
- Device/IP enrichment demonstrates relationship mechanics, is applied after temporal splitting with fixed configured bucket counts, and is excluded from the Phase 2 model.
- The current phase has no trained model or investigation agent.
- Batch CSV preparation is appropriate for the buildathon but not production streaming scale.
- Recommendations are simulated and never execute payment actions.

## Roadmap

See [docs/implementation-plan.md](docs/implementation-plan.md). The recommended next step is Phase 2: build the leakage-aware preprocessing/model pipelines, perform validation-only model and threshold selection, serialize the winning artifact, and expose risk prediction through FastAPI.

## License

Source code is available under the [MIT License](LICENSE). Dataset licensing and attribution remain governed by their original sources.
