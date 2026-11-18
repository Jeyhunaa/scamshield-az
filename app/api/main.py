"""FastAPI backend.

Run:  uvicorn app.api.main:app --reload
Docs: http://127.0.0.1:8000/docs
App:  http://127.0.0.1:8000/app   (customer banking app, served from web/dist when built)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.schemas import AssessRequest, AssessResponse, HealthResponse, Scenario
from scamshield import __version__
from scamshield.config import PROJECT_ROOT
from scamshield.service import AssessmentService, journey_features_from_input, load_scenarios

log = logging.getLogger("scamshield.api")
STATE: dict[str, object] = {"service": None, "error": None}
WEB_DIST = PROJECT_ROOT / "web" / "dist"


def _artifacts_dir() -> Path:
    return Path(os.environ.get("SCAMSHIELD_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))


def load_service() -> None:
    try:
        STATE["service"] = AssessmentService.load(_artifacts_dir())
        STATE["error"] = None
    except Exception as exc:  # keep the API up so /health can report the problem
        STATE["service"], STATE["error"] = None, str(exc)
        log.error("Failed to load models: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_service()
    yield


app = FastAPI(
    title="ScamShield AZ API",
    version=__version__,
    description="APP-scam risk scoring and cost-aware intervention recommendation (synthetic-data MVP).",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service: AssessmentService | None = STATE["service"]  # type: ignore[assignment]
    if service is None:
        return HealthResponse(status="degraded", models_loaded=False, model_version=None, trained_on=None, baseline_thresholds=None, detail=str(STATE["error"]))
    meta = service.predictor.metadata
    th = dict(zip(("warn", "verify", "hold"), (None if v == float("inf") else v for v in service.baseline_thresholds)))
    return HealthResponse(
        status="ok",
        models_loaded=True,
        model_version=f"scamshield-{__version__}",
        trained_on=str(meta.get("note", "synthetic data")),
        baseline_thresholds=th,
    )


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest) -> AssessResponse:
    service: AssessmentService | None = STATE["service"]  # type: ignore[assignment]
    if service is None:
        raise HTTPException(status_code=503, detail=f"Models not loaded: {STATE['error']}")
    try:
        journey = journey_features_from_input(
            events=[e.model_dump() for e in req.journey_events] if req.journey_events else None,
            features=req.journey_features.model_dump() if req.journey_features else None,
        )
        result = service.assess(req.transaction.model_dump(), journey, hold_allowed=req.hold_allowed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AssessResponse(**result)


@app.get("/scenarios", response_model=list[Scenario])
def scenarios() -> list[Scenario]:
    """Demo scenario inputs. Outcomes are never included: they come from /assess."""
    return [Scenario(id=key, **sc) for key, sc in load_scenarios().items()]


# Customer banking app (React build). Same origin as the API, so no CORS is needed.
if WEB_DIST.exists():
    app.mount("/app", StaticFiles(directory=WEB_DIST, html=True), name="customer-app")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/app/" if WEB_DIST.exists() else "/docs")
