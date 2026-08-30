from __future__ import annotations

import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from backend.app.main import app

frontend_dist = Path(os.getenv("FRAUDETECT_FRONTEND_DIST", "frontend/dist"))
if not (frontend_dist / "index.html").is_file():
    raise RuntimeError("The production frontend build is unavailable.")

# API and documentation routes are registered by backend.app.main before this
# catch-all mount. The production UI therefore shares the API's public origin.
app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
