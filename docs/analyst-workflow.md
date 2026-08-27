# Analyst Investigation Workflow

Phase 6 turns the existing risk, evidence, behavioral, relationship, and Copilot services into a durable local analyst workflow. It is a demonstration/research system built on synthetic data, not production case-management infrastructure.

## Case boundary

A case is an immutable, identifier-free snapshot of the intelligence available when an investigation is created. Creating a case invokes the existing investigation composition service; reopening a case reads the stored snapshot and does not rescore it. Analyst actions and Copilot generation can update workflow metadata, but cannot change the model output, threshold, evidence, behavioral context, or relationship context.

Application case IDs are random `CASE-` identifiers independent of PaySim. The SQLite store contains only:

- safe case metadata, workflow status, priority, latest note, and disposition;
- the positively selected sanitized investigation snapshot;
- evidence counts, explicit limitations, status history, append-only safe audit events, and optional Copilot output.

It does not store transaction references, origin/destination identifiers, raw history, fraud labels, model internals, API keys, or copies of the large history indexes.

## Lifecycle and human decisions

The enforced state machine is:

```text
OPEN -> IN_REVIEW -> ESCALATED -> CLOSED
                  \-> CLEARED  -> CLOSED
```

Invalid transitions return HTTP `409`. Closed cases are immutable. A note can be recorded during an allowed workflow state, with a maximum length of 2,000 characters. Moving to `ESCALATED` records the `ESCALATED` disposition; moving to `CLEARED` records `CLEARED`. Closing retains that disposition.

An analyst disposition is simulated workflow metadata and explicitly has `is_model_ground_truth: false`. It does not retrain the model and must not be interpreted as a confirmed fraud label.

## Deterministic priority policy

Case priority orders the local analyst queue; it is not an ML risk band or a second fraud score. The policy is evaluated once from the immutable snapshot:

1. `HIGH` ML risk becomes `CRITICAL` case priority.
2. `MEDIUM` ML risk or any `CRITICAL` evidence becomes `HIGH` priority.
3. A strong behavioral or relationship amount-deviation evidence item becomes `HIGH` priority.
4. Any `HIGH` or `MEDIUM` evidence becomes `MEDIUM` priority.
5. All other cases are `LOW` priority.

This policy never modifies probability, risk level, classification threshold, or evidence. Default queue ordering is unresolved before closed, then priority descending, ML risk descending, oldest first, and case ID as a stable tie-breaker.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/cases` | Create an immutable case from one internal reference or all four manual scoring fields |
| `GET` | `/api/v1/cases` | List cases with pagination and optional status, risk, priority, or disposition filters |
| `GET` | `/api/v1/cases/{case_id}` | Retrieve the complete analyst-safe snapshot and workflow history |
| `PATCH` | `/api/v1/cases/{case_id}` | Apply a valid status transition and/or record an analyst note |
| `POST` | `/api/v1/cases/{case_id}/copilot` | Generate and store an advisory report from only the saved sanitized snapshot |
| `GET` | `/api/v1/system/metrics` | Read privacy-safe aggregate operational workload counts |

List pagination uses `limit` (1–100) and `offset`. Existing risk and investigation endpoints remain backward compatible.

The Copilot cannot read analyst notes or database internals, change status, clear or escalate a case, or alter deterministic intelligence. Its provider, mode, AI availability, model, and fallback reason remain explicit. Provider failures use the existing labeled deterministic fallback.

## Local storage limitations

The default database is `artifacts/cases/cases.sqlite` and is ignored by Git. SQLite is suitable for a single-machine demonstration, not multi-user production operations. This implementation has no authentication, authorization, tenant isolation, encryption-at-rest policy, distributed locking, retention policy, or external audit sink. Those controls would be mandatory before using a similar workflow with real financial data.

## Deterministic showcase

With the frozen model bundle available, generate five local showcase cases:

```bash
make demo-cases
```

The command creates ignored files under `artifacts/demo/` and prints the three environment variables needed to run the API against them. It demonstrates:

- low ML risk;
- high ML risk with `CRITICAL` case priority;
- strong behavioral amount deviation from strictly earlier synthetic events;
- a first-observed relationship with prior origin network activity;
- manual input with explicitly unavailable behavioral and relationship history.

The high-risk case also receives a clearly labeled deterministic Copilot fallback report. The demo indexes use invented local keys and contain no PaySim rows or labels. Existing artifacts are not overwritten unless `--force` is passed directly to the script.

## Analyst workspace

The React workspace provides a filterable queue, case creation, immutable ML assessment, deterministic evidence cards, behavioral and relationship aggregates, limitations, Copilot generation, controlled status actions, analyst notes, and an investigation decision trace. It deliberately labels case priority separately from ML risk and human disposition separately from prediction.

The case timeline is exposed through the backward-compatible `decision_trace` field. New cases use server-generated append-only audit events for creation, snapshot capture, notes, Copilot generation, and lifecycle actions. Status history remains authoritative for allowed transitions. Existing Phase 6/7 cases are not backfilled; their timeline is projected only from factual timestamps and flags already stored. Missing historical actions are omitted.

The workspace also shows aggregate operational metrics for workload, lifecycle states, workflow priority, analyst disposition, and saved Copilot report modes. It never receives the internal reference used at creation, raw note content in metrics, or hidden database identifiers. See [auditability.md](auditability.md) for the event, migration, metrics, and privacy contracts.
