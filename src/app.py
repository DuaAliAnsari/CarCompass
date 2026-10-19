"""FastAPI service: /health, /metrics, /predict, /explain (docs at /docs).

Run:  uvicorn src.app:app --reload
"""
from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.model_utils import Artifacts, load_artifacts, max_year, predict_price

APP_VERSION = os.getenv("APP_VERSION", "dev")  # set to the image tag when deploying
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "models")


class PredictRequest(BaseModel):
    make: str = Field(min_length=1, examples=["Toyota"])
    model: str = Field(min_length=1, examples=["Corolla"])
    year: int = Field(ge=1980, le=max_year(), examples=[2016])
    mileage: int = Field(ge=0, le=1_000_000, examples=[75000])
    engine_cc: Optional[int] = Field(default=None, ge=50, le=8000)
    transmission: Optional[str] = None
    fuel: Optional[str] = None
    city: Optional[str] = None


def create_app(model_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="CarCompass", version=APP_VERSION,
                  description="Used-car price estimation for Pakistan.")
    app.state.stats = {"requests": 0, "errors": 0, "total_latency": 0.0}
    app.state.artifacts = None
    try:
        app.state.artifacts = load_artifacts(model_dir or DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        pass  # the service still starts; /predict answers 503 until a model exists

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        stats = request.app.state.stats
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            stats["errors"] += 1
            raise
        finally:
            stats["requests"] += 1
            stats["total_latency"] += time.perf_counter() - start
        if response.status_code >= 500:
            stats["errors"] += 1
        return response

    def get_artifacts() -> Artifacts:
        if app.state.artifacts is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Run `make train`.")
        return app.state.artifacts

    @app.get("/health")
    def health():
        art = app.state.artifacts
        return {
            "status": "ok",
            "model_loaded": art is not None,
            "model_version": art.meta["version"] if art else None,
            "app_version": APP_VERSION,
        }

    @app.get("/metrics")
    def metrics():
        stats = app.state.stats
        count = stats["requests"]
        art = app.state.artifacts
        return {
            "request_count": count,
            "error_count": stats["errors"],
            "avg_latency_ms": round(1000 * stats["total_latency"] / count, 2) if count else 0.0,
            "app_version": APP_VERSION,
            "model_version": art.meta["version"] if art else None,
        }

    @app.post("/predict")
    def predict(req: PredictRequest):
        return predict_price(get_artifacts(), req.model_dump())

    @app.post("/explain")
    def explain(req: PredictRequest):
        art = get_artifacts()
        result = predict_price(art, req.model_dump())
        result["top_factors"] = art.meta["importances"][:5]
        result["note"] = "Factors are the model's overall feature importance, not per-car."
        return result

    return app


app = create_app()
