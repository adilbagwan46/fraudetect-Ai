# Showcase deployment

The first public deployment is intentionally limited to the three-case showcase. It uses one Free
Render Python web service and one process. FastAPI serves the compiled React application and
`/api/v1` from the same origin, so production does not require a second frontend service or
cross-origin API access.

This deployment does not include raw or prepared PaySim CSVs or the 1.66 GB full behavioral and
relationship indexes. It does not retrain or recalibrate the frozen model. Its writable case store
uses Render's ephemeral filesystem: analyst changes can survive only for the life of the current
service instance and can reset after a restart, spin-down, or redeploy.

## Public Free showcase versus local normal mode

| | Public Free showcase | Local normal mode |
|---|---|---|
| Cases | Three curated genuine PaySim cases | Normal local case database |
| Case storage | Ephemeral `/tmp/fraudetect/cases.sqlite` | Ignored local `artifacts/cases/cases.sqlite` |
| History | Small showcase-only behavioral and relationship subsets | Full prepared PaySim behavioral and relationship indexes |
| Case creation | Curated showcase workflow | Arbitrary valid prepared PaySim transaction references |
| Persistence | Resets when Render replaces the instance filesystem | Persists in the local ignored database |

The public deployment is a bounded portfolio demonstration. `make normal` remains the full local
PaySim workflow and is not changed by this deployment configuration.

## Runtime bundle

The ignored local artifacts are packaged outside the repository:

```bash
.venv/bin/python scripts/showcase_runtime.py package \
  --output /tmp/fraudetect-showcase-runtime.zip
```

The package command validates and includes only:

- the active frozen model pointer and the eight files read by the runtime (about 2.55 MB);
- `artifacts/demo/behavior.sqlite` and `artifacts/demo/relationship.sqlite` as read-only indexes;
- `artifacts/demo/cases.sqlite` as the initial three-case seed.

It rejects a missing model file, an unexpected database schema, or a case seed that does not
contain exactly three cases. The generated ZIP is deterministic for unchanged inputs. The current
showcase publishes this bounded bundle as a GitHub Release asset; it contains no raw PaySim CSV,
full history index, credentials, or `.env` content. Private HTTPS object storage remains supported.

Record the SHA-256 printed by the package command. Render downloads the archive during its build,
limits its size, verifies the checksum before extraction, rejects unexpected archive members, and
fails the deployment if validation does not pass. The URL and any optional bearer token belong in
Render environment settings, never in Git.

## Render Blueprint setup

Create a Blueprint from `render.yaml`. During the initial Blueprint flow, enter:

- `FRAUDETECT_RUNTIME_ARTIFACT_URL`: the HTTPS download URL;
- `FRAUDETECT_RUNTIME_ARTIFACT_SHA256`: the exact SHA-256 printed by the package command.

If the object host requires bearer authentication, add
`FRAUDETECT_RUNTIME_ARTIFACT_TOKEN` manually in the service's Environment page. It is optional and
must not be added to `render.yaml` or `.env.example`. Public downloads may follow HTTPS redirects,
including redirects from GitHub Releases. HTTP redirects are rejected, and authenticated downloads
never forward the bearer token when the redirect origin changes.

The Blueprint configures the Free Render plan, Python 3.12.13 to match the frozen model environment,
Node 24 for the frontend build, deterministic Copilot fallback, and the three ignored runtime
artifact locations. It does not request a persistent disk.

At process startup, `/tmp/fraudetect/cases.sqlite` is created atomically from the validated
three-case seed only when the destination is absent. If the file already exists in the running
instance, startup validates and reuses it instead of overwriting analyst workflow state. When Free
Render replaces the ephemeral filesystem, the next startup creates a fresh three-case store from
the unchanged seed. The seed, frozen model, and two showcase history subsets stay in the deployment
build; the behavioral and relationship providers open their SQLite indexes read-only.

Gemini and OpenAI remain disabled by default. No provider key is required. To opt in later, set the
existing server-side provider variables in Render; never expose provider keys through a Vite
variable or frontend source.

## Commands run by Render

Build:

```bash
python -m pip install -r requirements-deploy.txt && \
  npm --prefix frontend ci && \
  npm --prefix frontend run build && \
  python scripts/showcase_runtime.py install
```

Start:

```bash
python scripts/showcase_runtime.py prepare-case-store && \
  uvicorn backend.app.deployment:app --host 0.0.0.0 --port $PORT
```

The health check is `/api/v1/health`; readiness is `/api/v1/system/readiness`. The frontend uses
its existing default `/api/v1`, so the public Render URL serves both the UI and API with no API key
or backend URL embedded in JavaScript.

## Rebuild and recovery

For a new artifact version, regenerate the ZIP, upload it, update the URL and SHA-256 in
Render, and redeploy. A Free Render restart, spin-down, or redeploy can discard the ephemeral case
store; startup then restores the original three curated cases from the verified seed. This reset is
intentional for the public showcase and does not affect the normal local case database.

The public showcase has no authentication or multi-tenant isolation. Keep one instance, do not use
it for real payment data, and treat it as a portfolio demonstration rather than a production fraud
decision system.
