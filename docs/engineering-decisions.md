# Engineering Decisions

## ED-001 — Keep classification and investigation separate

**Decision:** a supervised ML pipeline owns fraud probability. The LLM only investigates supplied evidence and explains uncertainty.

**Why:** model performance can be measured and reproduced; generative prose cannot substitute for a calibrated classifier. This also keeps risk scoring available during LLM failure.

**Alternative rejected:** asking an LLM to classify raw transactions. It is less reproducible, harder to evaluate, costly, and likely to invent causal explanations.

## ED-002 — Use PaySim as primary data

**Decision:** accept the public PaySim CSV as the primary model dataset and keep it out of Git.

**Why:** unlike the anonymized European card dataset, PaySim retains time steps, transaction types, account identifiers, and balances, enabling customer history and temporal behavior.

**Caveat:** PaySim is simulator-generated public data. Results show performance on that dataset, not expected merchant production performance.

## ED-003 — Add transparent deterministic graph enrichment

**Decision:** generate demo device/IP identifiers from account IDs and a configured seed, without consulting the fraud label.

**Why:** the selected dataset has no device/IP fields, while relationship investigation is central to the product demo. Determinism makes it testable and reproducible.

**Alternative rejected:** pretending account IDs are device IDs or sourcing an unrelated graph dataset. Both would obscure provenance.

**Consequence:** device/IP relationship findings demonstrate product mechanics only. They are excluded from claims about real-world model lift unless separately evaluated and labeled.

## ED-004 — Split chronologically

**Decision:** allocate ordered time steps to approximately 70% train, 15% validation, and 15% held-out test.

**Why:** random row splits allow future behavioral patterns to inform the past and poorly simulate deployment. Model/threshold selection uses only train and validation.

**Consequence:** fraud prevalence may vary between splits; metrics will report each split's actual prevalence.

## ED-005 — Compare a simple baseline with a practical nonlinear model

**Decision:** Phase 2 compared logistic regression and scikit-learn HistGradientBoosting before selecting the frozen weighted HistGradientBoosting model.

**Why:** logistic regression gives a transparent baseline; histogram gradient boosting captures nonlinear interactions without adding XGBoost installation and deployment complexity.

**Alternative deferred:** XGBoost may improve metrics, but an extra dependency is not justified until the dependency-light candidate is measured.

## ED-006 — Modular monolith over microservices

**Decision:** one FastAPI deployment with internal module boundaries and one React client.

**Why:** a seven-day build benefits from simple local operation, atomic changes, and direct debugging. The boundaries still permit later extraction.

## ED-007 — SQLite and file-based model artifacts

**Decision:** SQLite stores transactions, investigations, and audit records; serialized model/evaluation artifacts remain files.

**Why:** this is adequate for a single-node public demo, requires no service setup, and supports auditable snapshots.

## ED-008 — Bounded, evidence-addressable AI output

**Decision:** tools return typed evidence with stable IDs. The agent returns a validated schema that cites those IDs, separates evidence from interpretation, and includes uncertainties.

**Why:** schema validation alone prevents malformed JSON but not hallucinated facts. Evidence references make factual claims mechanically checkable.

**Fallback:** if the provider is disabled, absent, slow, invalid, or rejected by grounding checks, the API returns a typed report explicitly marked `deterministic_fallback`; the frozen ML and deterministic intelligence remain available.

## ED-009 — Human approval is the terminal action

**Decision:** recommendations are policy-bounded and labeled simulated. No endpoint blocks, refunds, or approves a real payment.

**Why:** false positives create customer and operational harm, and a competition demo has neither authority nor production controls for autonomous action.

## ED-010 — Make cost assumptions explicit

**Decision:** report an illustrative false-positive cost as false positives multiplied by configurable manual-review and customer-friction assumptions.

**Why:** it turns precision trade-offs into product language without presenting invented amounts as merchant facts.

## ED-011 — Treat complete time steps as atomic

**Decision:** chronological split boundaries may approximate 70/15/15 but cannot divide one PaySim `step` across splits.

**Why:** row-count cutoffs can expose simultaneous-event context across boundaries and become unsafe when causal history features are introduced.

## ED-012 — Enforce a scoring-time feature contract

**Decision:** Phase 2 training selects six allow-listed pre-decision features. Post-event balances, labels, identifiers, rules, synthetic relationships, and investigation evidence remain in prepared records but cannot enter the model matrix.

**Why:** explicit selection prevents accidental target, post-event, and identifier leakage as the dataset evolves.
