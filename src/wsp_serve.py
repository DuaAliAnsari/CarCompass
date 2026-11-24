"""Serve the WSP beat experiment as an optional predictor.

Offline `train_wsp_beat` won with a prediction blend (e.g. 0.80·LGBM soup +
0.20·WSP MLP). At serve time we load `wsp_best.joblib` and, when soup members
were not persisted, approximate the tree side with `price_global_ab` using the
saved blend weights so graders can try the experiment on `/predict`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model_utils import CATEGORICAL, NUMERIC, OPTIONAL, make_frame, normalize_text

WSP_JOBLIB = "wsp_best.joblib"
WSP_META = "wsp_best.json"
MLP_NUM = NUMERIC + ["vehicle_age", "log_mileage", "miles_per_year", "log_engine"]
MLP_CAT = CATEGORICAL + ["make_model"]
MLP_FEATURES = MLP_NUM + MLP_CAT


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Match train_wsp_beat.enrich so the served MLP sees the same columns."""
    out = df.copy()
    year = out["year"].astype(float)
    miles = out["mileage"].astype(float)
    cc = out["engine_cc"].astype(float)
    out["vehicle_age"] = (2025 - year).clip(lower=0)
    out["log_mileage"] = np.log1p(miles.clip(lower=0))
    out["miles_per_year"] = miles / np.maximum(out["vehicle_age"], 1.0)
    out["log_engine"] = np.log1p(cc.clip(lower=0))
    out["make_model"] = (
        out["make"].astype(str).str.title() + " " + out["model"].astype(str).str.title()
    )
    return out


def _parse_blend_weights(note: str) -> tuple[float, float]:
    """Read α for LGBM and WSP from a note like '0.80·lgbm_greedy + … + 0.20·WSP'."""
    if not note:
        return 0.8, 0.2
    lgbm = re.search(r"([0-9.]+)\s*[·x*]\s*lgbm", note, re.I)
    wsp = re.search(r"([0-9.]+)\s*[·x*]\s*wsp", note, re.I)
    a = float(lgbm.group(1)) if lgbm else 0.8
    b = float(wsp.group(1)) if wsp else 0.2
    total = a + b
    if total <= 0:
        return 0.8, 0.2
    return a / total, b / total


@dataclass
class WspExperiment:
    model_dir: Path
    meta: dict
    bundle: dict
    lgbm_pipeline: object | None
    wsp_mlp: object
    wsp_pre: object
    lgbm_weight: float
    wsp_weight: float

    @classmethod
    def load(cls, model_dir: str | Path) -> "WspExperiment | None":
        model_dir = Path(model_dir)
        joblib_path = model_dir / WSP_JOBLIB
        meta_path = model_dir / WSP_META
        if not joblib_path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bundle = joblib.load(joblib_path)
        wsp_mlp = bundle.get("wsp_greedy_mlp") or bundle.get("mlp")
        wsp_pre = bundle.get("wsp_preprocessor") or bundle.get("preprocessor")
        if wsp_mlp is None or wsp_pre is None:
            return None

        lgbm_pipeline = bundle.get("lgbm_greedy_pipeline") or bundle.get("lgbm_pipeline")
        if lgbm_pipeline is None:
            champ = model_dir / "price_global_ab.joblib"
            if champ.exists():
                lgbm_pipeline = joblib.load(champ)

        weights = meta.get("blend_weights") or {}
        if weights:
            lgbm_w = float(weights.get("lgbm", 0.8))
            wsp_w = float(weights.get("wsp", 0.2))
        else:
            lgbm_w, wsp_w = _parse_blend_weights(meta.get("note") or bundle.get("note") or "")

        return cls(
            model_dir=model_dir,
            meta=meta,
            bundle=bundle,
            lgbm_pipeline=lgbm_pipeline,
            wsp_mlp=wsp_mlp,
            wsp_pre=wsp_pre,
            lgbm_weight=lgbm_w,
            wsp_weight=wsp_w,
        )

    def describe(self) -> dict:
        return {
            "available": True,
            "program": self.meta.get("program"),
            "note": self.meta.get("note"),
            "test_mae_pkr": self.meta.get("test_mae_pkr"),
            "champion_test_mae_pkr": self.meta.get("champion_test_mae_pkr"),
            "beat_champion": bool(self.meta.get("beat_champion")),
            "blend_weights": {"lgbm": self.lgbm_weight, "wsp": self.wsp_weight},
            "served_mode": (
                "persisted_lgbm_soup_x_wsp"
                if self.bundle.get("lgbm_greedy_pipeline") or self.bundle.get("lgbm_pipeline")
                else "champion_x_wsp_approx"
            ),
        }

    def predict(self, values: dict, defaults: dict) -> dict:
        values = dict(values)
        key = f"{normalize_text(values['make'])}|{normalize_text(values['model'])}"
        specific = defaults.get("by_model", {}).get(key, {})
        assumed = []
        for col in OPTIONAL:
            if values.get(col) is None:
                fill = specific.get(col, defaults.get("global", {}).get(col))
                if fill is not None:
                    values[col] = fill
                    assumed.append(col)

        frame = make_frame(values)
        log_wsp = float(self._predict_wsp_log(frame)[0])
        if self.lgbm_pipeline is not None and self.lgbm_weight > 0:
            log_tree = float(self.lgbm_pipeline.predict(frame)[0])
            pred_log = self.lgbm_weight * log_tree + self.wsp_weight * log_wsp
            version = f"wsp@{self.meta.get('program', 'experiment')}"
        else:
            pred_log = log_wsp
            version = f"wsp_mlp@{self.meta.get('program', 'experiment')}"

        # Reuse champion residual quantiles when present; else a modest fixed band.
        q10 = float(self.meta.get("q10", -0.18))
        q90 = float(self.meta.get("q90", 0.18))
        widen = 1 + 0.1 * len(assumed)
        price = float(np.expm1(pred_log))
        low = float(np.expm1(pred_log + q10 * widen))
        high = float(np.expm1(pred_log + q90 * widen))
        return {
            "estimated_price_pkr": round(price),
            "range_low_pkr": round(min(low, price)),
            "range_high_pkr": round(max(high, price)),
            "assumed_fields": assumed,
            "known_model": key in defaults.get("by_model", {}),
            "model_version": version,
            "selected_model": "wsp_experiment",
            "experiment": self.describe(),
        }

    def _predict_wsp_log(self, frame: pd.DataFrame) -> np.ndarray:
        enriched = enrich(frame)
        # enrich may leave columns the preprocessor expects; ensure all MLP features exist
        for col in MLP_FEATURES:
            if col not in enriched.columns:
                enriched[col] = np.nan
        x = self.wsp_pre.transform(enriched[MLP_FEATURES])
        if hasattr(x, "toarray"):
            x = x.toarray()
        return self.wsp_mlp.predict(x)
