# LLM Investigation Copilot

## Purpose and separation of responsibilities

Phase 4 transforms the approved deterministic `InvestigationContext` into a concise, typed analyst
report. It does not calculate fraud probability, change model output, override thresholds, alter the
simulated recommendation, or participate in model features.

```text
Frozen ML output + deterministic evidence + causal behavior + relationship context
                         |
                         v
             positive-selection sanitizer
                         |
                         v
             SanitizedInvestigationContext
                         |
              +----------+-----------+
              |                      |
        OpenAI provider       deterministic fallback
              |                      |
              +----------+-----------+
                         |
              validated InvestigationReport
```

The OpenAI adapter follows the official Structured Outputs pattern: the server calls the Responses
API through `client.responses.parse` with a Pydantic response model. See the
[official OpenAI Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

## Approved context boundary

`build_sanitized_context` constructs a new payload through positive field selection. It never
serializes `InvestigationContext` wholesale.

Allowed data is limited to:

- transaction type, amount, pre-transaction origin balance, and step-derived hour;
- frozen probability, risk score/level, fraud classification, threshold, operating mode, and the
  deterministic simulated action;
- deterministic evidence ID, title, category, severity, and evidence-ID-specific scalar facts;
- training split boundary and aggregate training fraud prevalence;
- aggregate `BehavioralContext` values;
- aggregate, identifier-free `RelationshipContext` values and approved relationship evidence.

The Copilot never receives transaction references, customer/origin identifiers, destination
identifiers, raw transaction history, model version, derived feature internals, unrestricted
reference metadata, datasets, files, or database access. Unknown evidence fact keys and unapproved
string values are discarded. Behavioral and relationship availability explanations are
reconstructed from fixed server text.

## Prompt and injection boundary

The system prompt is maintained in `backend/app/services/copilot/prompts.py`. It defines the Copilot
as advisory, forbids probability changes and unsupported facts, distinguishes the four evidence
sources, requires limited-history qualification, and permits only reversible review steps.

The sanitized JSON is placed in a separate user-role message inside an explicit `DATA_CONTEXT`
boundary and declared inert data. No free-form transaction text is accepted by the request schema.
The OpenAI request sets `store=False`; all calls remain server-side.

## Provider architecture

`InvestigationLLMProvider` is the provider protocol. `OpenAIInvestigationProvider` supplies the real
server-side implementation. Tests and future providers can inject the same interface without
changing routing, sanitization, or report validation.

Real mode uses:

```text
FRAUDETECT_LLM_ENABLED=true
FRAUDETECT_LLM_PROVIDER=openai
FRAUDETECT_LLM_MODEL=gpt-5.6
OPENAI_API_KEY=<server-side secret>
```

Install the optional provider dependency with:

```bash
.venv/bin/python -m pip install -e ".[dev,ml,llm]"
```

Never put API keys in frontend environment variables or commit `.env`.

The LLM is disabled by default. No API key is needed for development, demos, or the automated test
suite; those paths use the deterministic fallback. When enabled, the API key is read only by the
backend and is excluded from settings representations, responses, readiness output, stored reports,
audit events, logs, and frontend bundles. Readiness reports whether the provider is enabled and
configured, but deliberately does not make a paid network request or claim that the external service
is available.

## Structured report and validation

`InvestigationReport` has these frontend-ready sections:

- concise summary;
- risk assessment with an enum risk level;
- up to five key signals, each citing approved evidence IDs;
- behavioral summary and explicit history limitation;
- relationship summary, evidence IDs, and explicit unavailable/sparse-history limitation;
- one to five uncertainties;
- one to four advisory recommended actions;
- analyst note;
- fixed mode-appropriate disclaimer.

Pydantic rejects missing, extra, incorrectly typed, or oversized fields. A grounding validator then
requires the report risk level to equal frozen model output, restricts citations to supplied
evidence IDs, enforces no/limited behavioral and relationship-history language, rejects invented
network connections or hidden/shared identities, rejects raw identifier patterns, rejects known
unsupported claims and irreversible actions, and rejects percentages absent from the supplied
context. The application replaces provider-written disclaimer text with the approved disclosure.

## Fallback and failure behavior

Fallback is the default. It requires no SDK credentials and is explicitly returned as:

```json
{
  "provider": "deterministic_fallback",
  "mode": "deterministic_fallback",
  "ai_available": false
}
```

It deterministically summarizes the same allowlisted context and never pretends to be LLM output.
Disabled configuration, missing keys, unsupported providers, SDK absence, timeouts, provider errors,
refusals/missing parsed output, malformed schemas, unsupported claims, and grounding failures all
degrade to this controlled mode. Internal exception details are not returned.

New reports also include safe `execution` metadata: whether a real provider was attempted and
succeeded, locally measured end-to-end generation elapsed time when a provider was attempted, and a
bounded failure category such as
`provider_timeout`, `invalid_output`, or `grounding_rejected`. It contains no prompts, provider
payloads, request IDs, exception text, credentials, or identifiers. The field is optional so reports
stored by Phase 6–8 load unchanged; the application does not invent metadata for them.

## API

`POST /api/v1/risk/investigate/copilot` accepts the same mutually exclusive request modes as the
deterministic investigation endpoint:

```json
{"transaction_reference": "TX-000000001"}
```

or all four manual scoring fields. Manual input receives no fabricated history. The endpoint first
builds the existing deterministic context, sanitizes it, runs the configured provider, validates the
report, and returns mode metadata, safe aggregate relationship context, and
`InvestigationReport`.

`POST /api/v1/risk/predict` and `POST /api/v1/risk/investigate` remain independent of provider
availability and retain their previous behavior.

## Privacy and limitations

- PaySim is synthetic; reports do not establish production fraud behavior.
- Natural-language grounding cannot prove every sentence semantically correct. Structured evidence
  citations, strong prompting, post-validation, and fallback reduce but cannot eliminate model risk.
- No external provider call was made during automated verification because no real credential is
  stored in the repository. The adapter, structured request, configuration, and failure behavior are
  tested with injected SDK-shaped clients.
- Provider availability, model access, latency, cost, and rate limits depend on the configured
  OpenAI account.
- Automated readiness is configuration-only. A configured provider can still fail at generation
  time; that request safely falls back and records only its non-sensitive failure category.
- The Copilot is advisory. Human analysts retain responsibility and organizational policy governs
  any action.
