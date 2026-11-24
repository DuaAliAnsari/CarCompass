"""FastAPI service: /health, /metrics, /models, /options, /predict, /explain.

Each request is answered by the price model the router picks for the car's year
(see src/model_router.py and models/manifest.json). The site in src/static/ is
served at /, /experiments, /wsp, /system, /pipeline and /demo. API docs are at /docs.

Run:  uvicorn src.app:app --reload
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.model_router import ModelRouter
from src.model_utils import CATEGORICAL, max_year, predict_price
from src.wsp_serve import WspExperiment

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
    predictor: Literal["routed", "wsp"] = Field(
        default="routed",
        description="routed = production LightGBM router; wsp = WSP beat experiment",
    )


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
    app.state.wsp = None
    model_root = model_dir or DEFAULT_MODEL_DIR
    try:
        app.state.router = ModelRouter.load(model_root)
    except FileNotFoundError:
        pass  # the service still starts; /predict answers 503 until a model exists
    app.state.wsp = WspExperiment.load(model_root)

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
        payload = req.model_dump()
        predictor = payload.pop("predictor", "routed")
        if predictor == "wsp":
            wsp = app.state.wsp
            if wsp is None:
                raise HTTPException(
                    status_code=503,
                    detail="WSP experiment model not loaded. Run `make train-wsp`.",
                )
            defaults = get_router().models[get_router().default_name].meta["defaults"]
            result = wsp.predict(payload, defaults)
            return result, None
        name, art = get_router().choose(req.year)
        result = predict_price(art, payload)
        result["selected_model"] = name
        return result, art

    @app.get("/health")
    def health():
        router = app.state.router
        wsp = app.state.wsp
        return {
            "status": "ok",
            "model_loaded": router is not None,
            "model_version": default_version(router),
            "models_served": sorted(router.models) if router else [],
            "wsp_loaded": wsp is not None,
            "wsp": wsp.describe() if wsp else None,
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

    def page(name: str) -> FileResponse:
        return FileResponse(STATIC_DIR / name)

    @app.get("/", include_in_schema=False)
    def home():
        return page("index.html")

    @app.get("/experiments", include_in_schema=False)
    def experiments_page():
        return page("experiments.html")

    @app.get("/wsp", include_in_schema=False)
    def wsp_page():
        return page("wsp.html")

    @app.get("/system", include_in_schema=False)
    def system_page():
        return page("system.html")

    @app.get("/pipeline", include_in_schema=False)
    def pipeline_page():
        return page("pipeline.html")

    @app.get("/demo", include_in_schema=False)
    def demo_page():
        return page("demo.html")

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/models")
    def models():
        body = get_router().describe()
        body["wsp_experiment"] = app.state.wsp.describe() if app.state.wsp else None
        return body

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
        if art is None:
            exp = result.get("experiment") or {}
            result["top_factors"] = []
            result["note"] = (
                "WSP experiment predictor "
                f"({exp.get('program', 'wsp')}: {exp.get('note', 'blend')}). "
                "Factors are not attributed the same way as the LightGBM champion."
            )
            return result
        result["top_factors"] = art.meta["importances"][:5]
        result["note"] = "Factors are the model's overall feature importance, not per-car."
        return result

    return app


app = create_app()
