from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.risk import router as risk_router
from backend.app.api.routes.system import router as system_router
from backend.app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Fraudetect AI API",
    description="Payment fraud risk detection and evidence-grounded investigation API.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(system_router, prefix=settings.api_prefix)
app.include_router(risk_router, prefix=settings.api_prefix)
