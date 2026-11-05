"""FastAPI service: /health, /metrics, /models, /predict, /explain (docs at /docs).

Each request is answered by the price model the router picks for the car's year
(see src/model_router.py and models/manifest.json).

Run:  uvicorn src.app:app --reload
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.model_router import ModelRouter
from src.model_utils import max_year, predict_price

logger = logging.getLogger("carcompass")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))  # message is already JSON
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

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
    assembly: Optional[str] = Field(default=None, examples=["Local"],
                                    description="Local or Imported")


def default_version(router) -> str | None:
    if router is None:
        return None
    return router.models[router.default_name].meta["version"]


def create_app(model_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="CarCompass", version=APP_VERSION,
                  description="Used-car price estimation for Pakistan.")
    app.state.stats = {"requests": 0, "errors": 0, "total_latency": 0.0}
    app.state.router = None
    try:
        app.state.router = ModelRouter.load(model_dir or DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        pass  # the service still starts; /predict answers 503 until a model exists

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        stats = request.app.state.stats
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            stats["errors"] += 1
            raise
        finally:
            elapsed = time.perf_counter() - start
            stats["requests"] += 1
            stats["total_latency"] += elapsed
            router = request.app.state.router
            logger.info(json.dumps({
                "event": "request",
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round(1000 * elapsed, 2),
                "app_version": APP_VERSION,
                "model_version": default_version(router),
            }))
        if response.status_code >= 500:
            stats["errors"] += 1
        return response

    def get_router() -> ModelRouter:
        if app.state.router is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Run `make train`.")
        return app.state.router

    def estimate(req: PredictRequest):
        name, art = get_router().choose(req.year)
        result = predict_price(art, req.model_dump())
        result["selected_model"] = name
        return result, art

    @app.get("/health")
    def health():
        router = app.state.router
        return {
            "status": "ok",
            "model_loaded": router is not None,
            "model_version": default_version(router),
            "models_served": sorted(router.models) if router else [],
            "app_version": APP_VERSION,
        }

    @app.get("/metrics")
    def metrics():
        stats = app.state.stats
        count = stats["requests"]
        return {
            "request_count": count,
            "error_count": stats["errors"],
            "avg_latency_ms": round(1000 * stats["total_latency"] / count, 2) if count else 0.0,
            "app_version": APP_VERSION,
            "model_version": default_version(app.state.router),
        }

    @app.get("/models")
    def models():
        return get_router().describe()

    @app.post("/predict")
    def predict(req: PredictRequest):
        return estimate(req)[0]

    @app.post("/explain")
    def explain(req: PredictRequest):
        result, art = estimate(req)
        result["top_factors"] = art.meta["importances"][:5]
        result["note"] = "Factors are the model's overall feature importance, not per-car."
        return result

    return app


app = create_app()
