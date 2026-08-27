# Deterministic Demo Guide

This workflow demonstrates the product without copying raw PaySim rows into the repository or presenting fixture output as genuine evaluation data. The frozen Phase 2A model performs every score. The local behavior and relationship events are deterministic synthetic fixtures used only to demonstrate investigation mechanics.

## Prepare the showcase

The frozen ignored model bundle must already exist under `artifacts/models/`. From the repository root:

```bash
make demo
```

This creates three ignored databases under `artifacts/demo/` and prints the generated case IDs. It does not touch the default analyst case database. If those demo files already exist, the command stops safely. Replace them only with explicit intent:

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

## Five-minute narrative

1. Expand system readiness. Explain that the model, deterministic evidence, local historical indexes, case store, and Copilot mode are independently observable without exposing secrets or data paths.
2. Review the operational overview: the five seeded cases intentionally cover `OPEN`, `IN_REVIEW`, `ESCALATED`, `CLEARED`, and `CLOSED`, and show multiple workflow priorities.
3. Select the `high-risk` seeded case. Contrast `CRITICAL PRIORITY` (queue ordering) with `HIGH ML RISK` (frozen prediction), then show its saved deterministic fallback report and analyst escalation.
4. Select `strong-behavioral-deviation`. Its `IN_REVIEW` timeline includes a real persisted note event; behavioral comparisons still use only earlier synthetic events.
5. Select `new-relationship`. It is `CLEARED`; show prior origin network breadth alongside the explicitly first-observed pair and its factual lifecycle events.
6. Select `history-unavailable`. It is `CLOSED`; show honest unavailable-state language, the closure timeline, and disabled controls.
7. Select `low-risk` to demonstrate the remaining `OPEN` workload and create additional actions if desired.

The displayed case IDs are generated application identifiers. Scenario names appear in the `make demo` output and are not stored in public case payloads.

## Boundaries to state aloud

- PaySim and the showcase history are synthetic; this is not production merchant evidence.
- The model probability, threshold, risk level, and operating mode are frozen.
- Behavioral and relationship context use `historical.step < current.step`.
- The Copilot receives only approved identifier-free context and remains advisory.
- The analyst owns lifecycle and disposition; no decision is fed back as model truth.
- Audit events and operational metrics describe workflow actions only; they are not a second fraud-intelligence source.
