"""Shared helpers: feature lists, artifact loading and the price estimate logic.

Training (src/train.py) and serving (src/app.py) both import from here, so they
always use the same columns and the same cleaning rules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REQUIRED_INPUT = ["make", "model", "year", "mileage"]
OPTIONAL = ["engine_cc", "transmission", "fuel", "city", "assembly"]
NUMERIC = ["year", "mileage", "engine_cc"]
CATEGORICAL = ["make", "model", "transmission", "fuel", "city", "assembly"]
FEATURES = NUMERIC + CATEGORICAL

MODEL_FILE = "price_model.joblib"
META_FILE = "metadata.json"


def max_year() -> int:
    return date.today().year + 1


def normalize_text(value):
    """Strip and title-case text so 'toyota ' and 'Toyota' match. Missing stays NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().title()
    return text if text else np.nan


def make_frame(values: dict) -> pd.DataFrame:
    """Build the one-row DataFrame the sklearn pipeline expects."""
    row = {}
    for col in FEATURES:
        val = values.get(col)
        if col in CATEGORICAL:
            row[col] = normalize_text(val)
        else:
            row[col] = np.nan if val is None else float(val)
    return pd.DataFrame([row], columns=FEATURES)


@dataclass
class Artifacts:
    pipeline: object
    meta: dict


def load_artifacts(model_dir) -> Artifacts:
    """Load the saved model and its metadata. Raises FileNotFoundError if missing."""
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILE
    meta_path = model_dir / META_FILE
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No trained model found in {model_dir}. Run `make train`.")
    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return Artifacts(pipeline=pipeline, meta=meta)


def predict_price(art: Artifacts, values: dict) -> dict:
    """Return price, range and which optional fields were assumed.

    Missing optional fields are filled with typical values for that make/model
    (from the training data), flagged as assumed, and the range gets wider.
    """
    values = dict(values)
    defaults = art.meta["defaults"]
    key = f"{normalize_text(values['make'])}|{normalize_text(values['model'])}"
    specific = defaults["by_model"].get(key, {})
    assumed = []
    for col in OPTIONAL:
        if values.get(col) is None:
            fill = specific.get(col, defaults["global"].get(col))
            if fill is not None:
                values[col] = fill
                assumed.append(col)

    pred_log = float(art.pipeline.predict(make_frame(values))[0])
    widen = 1 + 0.1 * len(assumed)
    price = float(np.expm1(pred_log))
    low = float(np.expm1(pred_log + art.meta["q10"] * widen))
    high = float(np.expm1(pred_log + art.meta["q90"] * widen))
    return {
        "estimated_price_pkr": round(price),
        "range_low_pkr": round(min(low, price)),
        "range_high_pkr": round(max(high, price)),
        "assumed_fields": assumed,
        "known_model": key in defaults["by_model"],
        "model_version": art.meta["version"],
    }
