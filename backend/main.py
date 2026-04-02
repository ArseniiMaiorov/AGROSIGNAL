from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text
from starlette.responses import Response

from api.auth import router as auth_router
from api.admin_ml import router as admin_ml_router
from api.archive import router as archive_router
from api.bootstrap import router as bootstrap_router
from api.crops import router as crops_router
from api.data_imports import router as data_imports_router
from api.fields import router as fields_router
from api.detect import router as detect_router
from api.runs import router as runs_router
from api.debug_tiles import router as debug_tiles_router
from api.labeling import router as labeling_router
from api.layers import router as layers_router
from api.manual_markup import router as manual_markup_router
from api.modeling import router as modeling_router
from api.predictions import router as predictions_router
from api.satellite import router as satellite_router
from api.storage import router as storage_router
from api.status import router as status_router
from api.weather import router as weather_router
from core.logging import configure_logging
from core.rate_limit import limiter
from core.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)

    from storage.db import init_db, seed_defaults
    await init_db()
    await seed_defaults()

    yield


app = FastAPI(
    title="AgroMap API",
    version="1.0.0",
    description="Agricultural field auto-detection service using Sentinel-2 imagery",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(bootstrap_router, prefix="/api/v1")
app.include_router(fields_router, prefix="/api/v1")
app.include_router(detect_router, prefix="/api/v1")
app.include_router(runs_router, prefix="/api/v1")
app.include_router(debug_tiles_router, prefix="/api/v1")
app.include_router(layers_router, prefix="/api/v1")
app.include_router(weather_router, prefix="/api/v1")
app.include_router(status_router, prefix="/api/v1")
app.include_router(storage_router, prefix="/api/v1")
app.include_router(crops_router, prefix="/api/v1")
app.include_router(predictions_router, prefix="/api/v1")
app.include_router(satellite_router, prefix="/api/v1")
app.include_router(modeling_router, prefix="/api/v1")
app.include_router(archive_router, prefix="/api/v1")
app.include_router(manual_markup_router, prefix="/api/v1")
app.include_router(labeling_router, prefix="/api/v1")
app.include_router(data_imports_router, prefix="/api/v1")
app.include_router(admin_ml_router, prefix="/api/v1")


@app.get("/health")
async def health():
    checks: dict[str, str] = {}
    try:
        from storage.db import get_engine
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
    try:
        from core.celery_app import celery as celery_app
        inspect_result = celery_app.control.ping(timeout=2.0)
        checks["celery"] = "ok" if inspect_result else "no_workers"
    except Exception:
        checks["celery"] = "unavailable"
    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain; charset=utf-8")
