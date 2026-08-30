# Deterministic Genuine-PaySim Showcase

This workflow demonstrates the product with existing prepared PaySim transactions. It does not
fabricate transactions, history, relationships, probabilities, or evidence. The frozen Phase 2A
model performs every score, and the existing behavioral and relationship providers calculate all
context using `historical.step < current.step`.

## Prerequisites

Prepare genuine PaySim and build the full ignored history indexes before generating the showcase:

```bash
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/build_behavior_history.py
.venv/bin/python scripts/build_relationship_history.py
```

The frozen model bundle must also be available under `artifacts/models/`.

## Generate the showcase

```bash
make demo
```

The command deterministically selects three existing PaySim transactions and creates three ignored
databases under `artifacts/demo/`:

- `strong-investigation`: High ML risk with earlier origin behavior, origin-network context,
  destination-network context, deterministic evidence, a saved deterministic Copilot fallback,
  and a persisted analyst-review event.
- `limited-context`: Medium ML risk from the earliest PaySim step, proving that unavailable
  history is stated explicitly rather than inferred.
- `lower-risk-resolution`: Low ML risk with richer earlier behavioral and network context,
  progressed through analyst review to a human `CLEARED` disposition.

The generator copies only each selected transaction and the strictly earlier rows required for its
aggregates. It does not copy fraud labels. Raw identities and internal transaction references remain
inside ignored local indexes and are excluded from case snapshots and public API responses.

The generator does not overwrite an existing showcase. Replacement requires explicit intent:

```bash
.venv/bin/python scripts/seed_demo_cases.py --force
```

## Start the product

Terminal 1:

```bash
FRAUDETECT_CASE_DATABASE=artifacts/demo/cases.sqlite \
FRAUDETECT_BEHAVIORAL_HISTORY_DB=artifacts/demo/behavior.sqlite \
FRAUDETECT_RELATIONSHIP_HISTORY_DB=artifacts/demo/relationship.sqlite \
.venv/bin/uvicorn backend.app.main:app --port 8000
```

Terminal 2:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173/`.

## Presentation narrative

1. Open `strong-investigation`. Contrast frozen ML risk with workflow priority, then review its
   deterministic evidence, non-zero behavioral history, origin and destination network context,
   labeled deterministic Copilot brief, analyst note, and append-only timeline.
2. Open `limited-context`. Show that both intelligence panels explicitly report zero eligible
   earlier-step history instead of inventing context.
3. Open `lower-risk-resolution`. Contrast Low ML risk with its independent workflow priority,
   review the richer historical context, and show the human `CLEARED` outcome.

Scenario names are printed by the generator for the presenter but are not stored in public case
payloads. Case IDs are generated application identifiers.

## Boundaries to state aloud

- PaySim is public synthetic research data; this is not production merchant evidence.
- The model probability, threshold, risk level, and operating mode are frozen.
- Behavioral, pair, origin-network, and destination-network queries require
  `historical.step < current.step`.
- Genuine PaySim has no eligible repeated exact origin-destination pairs in the prepared index.
  Network context is presented honestly without fabricating pair history.
- The Copilot receives only approved identifier-free context and remains advisory.
- The analyst owns lifecycle and disposition; no decision is fed back as model truth.
- Audit events and operational metrics describe workflow actions only.
