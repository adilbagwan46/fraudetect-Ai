# Implementation Plan

## Execution strategy

Each phase ends with runnable tests and a usable vertical increment. The order protects the measurable ML core before graph, LLM, and UI polish are added.

## Phase 1 — Foundation and data pipeline (current)

- Establish Python/FastAPI and React/Vite/TypeScript project boundaries.
- Define centralized settings and API health/data-readiness contracts.
- Implement PaySim column validation and normalized transaction identifiers.
- Add deterministic, label-independent demo enrichment for device and IP fields.
- Engineer reusable temporal, amount, and balance-consistency features.
- Create chronological 70/15/15 split metadata with train-only/validation/test separation.
- Add a deterministic small demo generator and a CLI preparation workflow.
- Test schema validation, reproducibility, feature invariants, and split isolation.

**Exit condition:** a raw PaySim-compatible or generated demo CSV can be validated and transformed into reproducible split files plus a machine-readable manifest.

## Phase 1.5 — Data foundation correction (complete)

- Require complete time steps to remain atomic across chronological splits.
- Define an explicit scoring-time-safe ML feature allowlist.
- Separate safe row-local features from post-event investigation evidence.
- Apply fixed-configuration device/IP enrichment only after splitting and keep it out of ML.
- Record source hash, exact split proportions, time-step counts, and feature contract in the manifest.
- Provide a causal historical-feature boundary that forbids same-step and future events.
- Keep generated data limited to smoke testing; genuine evaluation requires `data/raw/paysim.csv`.

**Exit condition:** the actual PaySim CSV can enter a leakage-aware preparation flow, while unsafe fields are mechanically excluded from the future model matrix.

## Phase 2 — ML risk engine

- Build a leakage-aware preprocessing pipeline.
- Train a logistic-regression baseline and HistGradientBoosting candidate.
- Handle imbalance using training-only class weights/sampling where supported; do not alter validation/test prevalence.
- Select model and LOW/MEDIUM/HIGH thresholds using validation results and documented operational trade-offs.
- Evaluate the frozen choice once on held-out test data.
- Save model pipeline, feature contract, threshold configuration, metrics, and confusion matrix.
- Add prediction service and API route.

**Exit condition:** transaction features produce a reproducible probability, risk score, risk level, and factor summary; final metrics come only from the held-out test split.

### Phase 2A completion note

Four approved candidates were compared on genuine prepared PaySim: weighted/unweighted Logistic Regression and weighted/unweighted HistGradientBoosting. Training-only chronological calibration, validation-only candidate/threshold selection, a frozen balanced HistGradientBoosting winner, one-shot held-out evaluation, versioned artifacts, and typed risk APIs are complete. See `docs/evaluation.md`.

## Phase 2B — Evidence and explainability engine (complete)

- Preserve the frozen Phase 2A model, threshold, calibration, and risk behavior.
- Build a reproducible aggregate reference profile from training steps 1–323 only.
- Measure global model reliance with fixed-seed training-only permutation importance.
- Generate deterministic local model-risk, amount, balance, type, and time evidence.
- Prioritize three to five proportional items without causal claims.
- Extend prediction responses and expose a typed investigation context for future LLM use.

**Exit condition:** identical model input and artifact produce identical structured evidence, low-risk results remain proportional, and future investigation components can consume a controlled factual context.

## Phase 3A — Causal behavioral intelligence (complete)

- Add origin history using the strict `historical.step < current.step` boundary.
- Compute focused prior-amount, amount-deviation, recent-step, step-gap, and type-novelty context.
- Build a label-free generated SQLite index through a provider abstraction for efficient lookup.
- Keep raw PaySim identities internal and return aggregate behavioral facts only.
- Extend investigation context and deterministic evidence without changing prediction behavior.

**Exit condition:** referenced investigations return deterministic aggregate context, manual inputs report unavailable history, future/same-step mutations cannot affect earlier context, and frozen Phase 2A/2B behavior remains intact.

## Phase 3B — Relationship intelligence (future)

- Compute new-device and new-IP indicators without changing the frozen model.
- Build a lightweight NetworkX graph for Customer–Device–IP–Transaction–Merchant links.
- Return a bounded local neighborhood and interpretable relationship aggregates.
- Add evidence DTOs and transaction/evidence/connection APIs.

**Exit condition:** every investigated transaction returns risk plus behavioral and relationship context, including explicit missing-data states.

## Phase 4 — Evidence-grounded AI investigation

- Define provider-neutral `LLMProvider` and OpenAI-compatible implementation.
- Implement allow-listed evidence tools: transaction history, behavior profile, entity connections, risk factors, policy, and optionally similar cases.
- Require JSON-schema-compatible structured results and evidence IDs.
- Validate output, strip/reject unsupported factual claims, capture uncertainty, and persist tool/audit events.
- Provide a deterministic no-LLM fallback status while retaining all risk/evidence output.

**Exit condition:** an investigation produces traceable evidence, interpretation, uncertainty, and a bounded recommendation; simulated provider failure is tested.

## Phase 5 — Analyst frontend

- Build risk overview, filterable transaction list, investigation workspace, local graph, and evaluation page.
- Emphasize the investigation page, evidence timeline, risk semantics, and human decision controls.
- Connect loading, empty, error, and LLM-unavailable states.

**Exit condition:** a five-minute end-to-end analyst demo works without developer tools.

## Phase 6 — Verification and documentation

- Run the final held-out evaluation once after model/threshold freeze.
- Document false-positive cost assumptions: `FP × (review cost + friction estimate)`.
- Add integration tests and audit/reliability checks.
- Complete README, data card, architecture, evaluation, and engineering decisions.
- Verify fresh-clone setup and remove generated/large artifacts from Git.

## Phase 7 — Demo polish

- Fix bugs and accessibility/layout problems; add no major features.
- Prepare stable demo cases covering low, medium, high, graph context, and LLM fallback.
- Capture screenshots and rehearse the five-minute narrative.

## Major implementation choices

| Area | Choice | Reason |
|---|---|---|
| Dataset | PaySim + transparent deterministic enrichment | Labeled fraud plus customer/account history; missing graph identifiers are clearly handled |
| Split | Chronological 70/15/15 by time step | Better reflects deployment and reduces future-to-past leakage |
| ML | Logistic regression baseline vs HistGradientBoosting | Honest interpretable baseline and strong dependency-light candidate |
| Backend | FastAPI modular monolith | Typed API and fast build without operational overhead |
| Frontend | React + Vite + TypeScript | Fast, polished SPA workflow and type-safe API integration |
| Graph | NetworkX bounded neighborhoods | Explainable and sufficient at demo scale |
| Database | SQLite | Zero-ops local durability; repository abstraction preserves upgrade path |
| AI | Provider interface + evidence tools + validated result | Meaningful tool use, grounding, fallback, and vendor portability |
