# Relationship Intelligence and Causal Network Context

## Purpose and responsibility boundary

Phase 5 adds deterministic origin-destination relationship context for investigation. It answers
whether a pair was observed previously, summarizes strictly prior amounts, and describes aggregate
origin and destination network breadth. It never changes the frozen Phase 2A probability,
thresholds, classification, calibration, features, or simulated action.

Relationship novelty and amount deviation are descriptive facts, not evidence that fraud, money
laundering, shared identity, or coordinated activity occurred.

## Architecture

```text
prepared train -> validation -> test rows (five selected columns only)
                              |
                              v
             ignored label-free SQLite relationship index
                              |
             transaction reference + indexed read-only queries
                              |
                              v
                 identifier-free RelationshipContext
                    + typed relationship evidence
                              |
              deterministic API + Copilot sanitizer + UI
```

`RelationshipHistoryProvider` is the application boundary. The demonstration implementation,
`SQLiteRelationshipHistoryProvider`, opens the generated database in read-only mode. The model and
`POST /risk/predict` never instantiate this provider.

## Strict causal boundary

Every lookup uses `historical.step < current.step`.

The current row, every same-step row, and every future row are excluded. The offline reference
iterator computes every context for a complete step before adding that step to state. Because the
index contains all chronological splits but all queries enforce the strict predicate, validation
and test transactions may use earlier history while information never flows backward.

Tests prove current-event exclusion, same-step exclusion, future invariance, and causal history
across train, validation, and test boundaries.

## Storage and indexes

Run:

```bash
.venv/bin/python scripts/build_relationship_history.py
```

The generated artifact defaults to `artifacts/relationship/history.sqlite` and is ignored by Git.
The builder makes one chunked pass over train, validation, and test and selects only internal
transaction reference, PaySim step, amount, internal origin key, and internal destination key. It
never loads or stores `is_fraud` or `is_flagged_fraud`. Internal keys exist only inside the local
artifact and never enter public DTOs.

SQLite indexes cover transaction-reference resolution, origin/destination pair history, prior
origin network queries, and prior destination network queries. A lookup is an indexed reference
read plus bounded pair and aggregate queries. Pair median and empirical percentile require reading
the earlier amounts for that pair, which is practical for PaySim's generally sparse pairs. This
avoids a 6.3-million-row scan per investigation and avoids retaining a large Python graph in memory.

A full in-memory graph or NetworkX was intentionally not added: Phase 5 requires deterministic
local aggregates, not traversal, community detection, or centrality algorithms.

For the prepared 6,362,620-row PaySim dataset, the generated SQLite artifact is approximately
1.1 GB. A local build completed in about 31 seconds and sampled warm/cold lookups completed in
approximately 1.3–5.1 ms on the development machine; these are observations, not portable
benchmarks. An aggregate audit found no repeated directed origin-destination pair in this PaySim
dataset. Genuine-data investigations therefore report first-seen pair history, while repeated-pair
and amount-deviation behavior is retained and verified with deterministic fixtures. This limitation
is reported rather than manufacturing relationship history.

## Aggregate schema

`RelationshipContext` contains no identifiers and distinguishes context availability from actual
pair history. It includes:

- prior pair count, total, average, median, and maximum;
- step gap and first-seen/previously-seen state;
- ratios against prior average, median, and maximum;
- empirical prior percentile and maximum exceedance;
- origin prior transaction and unique-counterparty counts;
- destination prior transaction and unique-origin counts;
- new-destination and new-origin-for-destination flags;
- explicit unavailable and limited-baseline states.

Undefined ratios use `null`. Public schemas reject NaN and infinity.

## Evidence methodology

Relationship evidence is returned separately from model and behavioral evidence under the typed
`RELATIONSHIP_CONTEXT` category:

- `relationship_context_unavailable`
- `relationship_new_counterparty`
- `relationship_previously_observed`
- `relationship_limited_history`
- `relationship_amount_deviation`
- `relationship_exceeds_prior_maximum`

A substantial amount deviation means at least five times the prior relationship average. Limited
history means no more than two prior interactions. Neither condition changes model probability or
proves fraud.

## API, Copilot, and privacy boundary

Referenced `POST /api/v1/risk/investigate` requests return aggregate relationship context and typed
relationship evidence. Manual requests explicitly mark relationship context unavailable. A missing
generated relationship index returns HTTP 503; an unknown reference returns HTTP 404.

The Copilot sanitizer reconstructs relationship context through positive selection and admits only
approved aggregate facts and evidence IDs. It excludes transaction references, origin/destination
keys, raw relationship history, database content, and arbitrary metadata. Grounding validation
requires relationship claims to cite supplied relationship evidence and rejects invented hidden
relationships, shared identities, or network connections. Deterministic fallback preserves the
same unavailable and sparse-history limitations.

## Complexity and limitations

- Index construction is `O(n)` input ingestion plus SQLite index construction over `n` rows.
- Reference resolution is indexed; pair work is proportional to the strictly prior interaction
  count for the selected pair.
- Origin/destination aggregate work uses covering indexes but can grow with entity degree.
- PaySim identities and relationships are synthetic simulation artifacts, not verified people,
  merchants, devices, or real payment networks.
- No centrality, community, ring, multi-hop, shared-device, or shared-IP claim is made.
- This local SQLite design demonstrates causal context, not production streaming graph analytics.
