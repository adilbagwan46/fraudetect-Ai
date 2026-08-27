from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes.cases import router as cases_router
from backend.app.api.routes.risk import router as risk_router
from backend.app.api.routes.system import router as system_router
from backend.app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Fraudetect AI API",
    description="Payment fraud risk detection and evidence-grounded investigation API.",
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, error: RequestValidationError
) -> JSONResponse:
    """Return useful validation details without echoing rejected request values."""

    detail = [
        {
            "type": item["type"],
            "loc": item["loc"],
            "msg": item["msg"],
        }
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(system_router, prefix=settings.api_prefix)
app.include_router(risk_router, prefix=settings.api_prefix)
app.include_router(cases_router, prefix=settings.api_prefix)
