"""FastAPI service: /health, /metrics, /models, /options, /predict, /explain.

Each request is answered by the price model the router picks for the car's year
(see src/model_router.py and models/manifest.json). A small test page is served at
/ (src/static/index.html) and the API docs at /docs.

Run:  uvicorn src.app:app --reload
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.model_router import ModelRouter
from src.model_utils import CATEGORICAL, max_year, predict_price

STATIC_DIR = Path(__file__).parent / "static"

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


def known_values(router) -> dict:
    """Categories the default model's one-hot encoder learned, plus models per make."""
    art = router.models[router.default_name]
    encoder = (art.pipeline.named_steps["pre"].named_transformers_["cat"]
               .named_steps["onehot"])
    columns = art.pipeline.named_steps["pre"].transformers_[1][2]
    values = {
        col: sorted(str(v) for v in cats if str(v) != "unknown")
        for col, cats in zip(columns, encoder.categories_)
        if col in CATEGORICAL
    }
    by_make: dict = {}
    for key in art.meta["defaults"]["by_model"]:
        make, model = key.split("|", 1)
        by_make.setdefault(make, []).append(model)
    values["models_by_make"] = {m: sorted(v) for m, v in sorted(by_make.items())}
    return values


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

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/models")
    def models():
        return get_router().describe()

    @app.get("/options")
    def options():
        """Values the default model was trained on, for form dropdowns."""
        return known_values(get_router())

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
