# Showcase deployment

The first public deployment is intentionally limited to the three-case showcase. It uses one
Render Python web service, one process, and one persistent disk. FastAPI serves the compiled React
application and `/api/v1` from the same origin, so production does not require a second frontend
service or cross-origin API access.

This deployment does not include raw or prepared PaySim CSVs or the 1.66 GB full behavioral and
relationship indexes. It does not retrain or recalibrate the frozen model.

## Private runtime bundle

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
contain exactly three cases. The generated ZIP is deterministic for unchanged inputs. Keep it
private and upload it to private HTTPS object storage. Do not attach it to a public release.

Record the SHA-256 printed by the package command. Render downloads the archive during its build,
limits its size, verifies the checksum before extraction, rejects unexpected archive members, and
fails the deployment if validation does not pass. The URL and any optional bearer token belong in
Render environment settings, never in Git.

## Render Blueprint setup

Create a Blueprint from `render.yaml`. During the initial Blueprint flow, enter:

- `FRAUDETECT_RUNTIME_ARTIFACT_URL`: the private HTTPS download URL;
- `FRAUDETECT_RUNTIME_ARTIFACT_SHA256`: the exact SHA-256 printed by the package command.

If the object host requires bearer authentication, add
`FRAUDETECT_RUNTIME_ARTIFACT_TOKEN` manually in the service's Environment page. It is optional and
must not be added to `render.yaml` or `.env.example`. Use a direct object URL: authenticated
downloads intentionally reject redirects so a bearer token cannot be forwarded to another host.

The Blueprint configures Python 3.12.13 to match the frozen model environment, Node 24 for the
frontend build, deterministic Copilot fallback, the three ignored runtime artifact locations, a
Singapore service, and a 1 GB persistent disk mounted at `/var/data/fraudetect`. A paid Render web
service is required because Render persistent disks are not available on free web services.

The disk-backed `/var/data/fraudetect/cases.sqlite` is created from the curated seed only when it
does not already exist. A restart or deploy validates and reuses an existing case store; it never
silently overwrites analyst workflow state. The model and the two history subsets remain in the
read-only deployment build. A disk-backed service is restricted to one instance, which matches the
current SQLite architecture.

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

For a new artifact version, regenerate the private ZIP, upload it, update the URL and SHA-256 in
Render, and redeploy. Existing disk-backed case state is preserved. To restore the original curated
queue, take a disk backup first and replace the case store through an explicit maintenance action;
normal deploys intentionally do not reset it.

The public showcase has no authentication or multi-tenant isolation. Keep one instance, do not use
it for real payment data, and treat it as a portfolio demonstration rather than a production fraud
decision system.
