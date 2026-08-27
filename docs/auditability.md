# Auditability and Operational Intelligence

Phase 8 observes the analyst workflow; it does not alter fraud scoring or become a second source of fraud intelligence. The frozen probability, ML risk, threshold, operating mode, deterministic evidence, causal behavioral and relationship context, immutable case snapshot, priority policy, and Copilot grounding contract are unchanged.

## Append-only audit events

New cases persist server-generated events in `case_audit_events`. The public case timeline remains the existing `decision_trace` field, so clients have one chronological timeline rather than competing lifecycle and audit displays.

Supported meaningful events are:

- `CASE_CREATED` and `INTELLIGENCE_CAPTURED`, stored atomically with a new case;
- `NOTE_ADDED`, which records only that a note was added and never its content;
- `COPILOT_GENERATED`, including the factual `real_llm` or `deterministic_fallback` mode;
- `ANALYST_REVIEWED`, `CASE_ESCALATED`, `CASE_CLEARED`, and `CASE_CLOSED`, with safe previous and new statuses.

Each event stores a server timestamp, actor, optional lifecycle statuses, optional Copilot mode, and a note-recorded boolean. It does not store transaction references, origin/destination identifiers, raw history, fraud labels, note text, API keys, filesystem paths, or public database row IDs. SQLite triggers reject updates and deletes to audit rows. Invalid transitions and rejected closed-case actions roll back without events.

Lifecycle status history remains the source of truth for state transitions. Audit events describe meaningful actions and cannot modify cases, snapshots, model output, priority, evidence, histories, or Copilot reports.

## Timeline and deterministic ordering

Audit rows are ordered by their factual timestamp and an internal insertion sequence used only as a tie-breaker. The sequence is never returned by the API. This makes actions sharing one transaction timestamp deterministic without fabricating time.

Phase 6/7 databases are migrated by creating an empty audit table, indexes, and append-only triggers. Existing cases are not backfilled. When a case has no audit rows, its timeline remains a read-only projection of already stored creation, Copilot-generation, and lifecycle timestamps. Events lacking factual historical support are omitted. Initialization does not change snapshots, status history, case timestamps, or Copilot output.

## Operational metrics

`GET /api/v1/system/metrics` derives aggregate counts from the case store:

- `total_cases` and `active_cases` (`status != CLOSED`);
- exact `OPEN`, `IN_REVIEW`, `ESCALATED`, `CLEARED`, and `CLOSED` status counts;
- `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW` case-priority counts;
- analyst disposition counts;
- total saved Copilot reports split into `deterministic_fallback` and `real_llm` modes.

Metrics retrieval is read-only. It returns no case IDs, note content, intelligence snapshots, transaction identities, raw histories, fraud labels, paths, secrets, stack traces, or internal SQLite IDs. `CRITICAL` is a case priority only; ML risk remains `LOW`, `MEDIUM`, or `HIGH`. Copilot counts use the persisted report mode and never label fallback output as LLM-generated.

## Limitations

The local SQLite audit log is suitable for a single-machine demonstration. It is not an external tamper-evident ledger and does not provide authentication, authorization, multi-user actor identity, cryptographic event signing, retention governance, or a production audit export. The current `ANALYST` actor is a role label, not a verified user identity. Human workflow remains authoritative and Copilot remains advisory.
