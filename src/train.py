"""Train several price models, log every run to MLflow, save the best one.

Run:  python -m src.train
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.model_utils import (
    CATEGORICAL,
    FEATURES,
    META_FILE,
    MODEL_FILE,
    NUMERIC,
    OPTIONAL,
)
from src.prepare_data import file_hash


def build_estimator(name: str, params: dict):
    if name == "baseline":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor  # imported lazily so it is optional

        return LGBMRegressor(random_state=42, verbose=-1, **params)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, params: dict) -> Pipeline:
    """One sklearn pipeline = preprocessing + model, so training and serving match."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("pre", pre), ("model", build_estimator(name, params))])


def typical_values(df: pd.DataFrame) -> dict:
    """Typical optional values per make/model, used to fill missing fields."""
    def summarize(group):
        out = {}
        for col in OPTIONAL:
            series = group[col].dropna()
            if series.empty:
                continue
            out[col] = float(series.median()) if col in NUMERIC else str(series.mode().iloc[0])
        return out

    by_model = {f"{mk}|{md}": summarize(g) for (mk, md), g in df.groupby(["make", "model"])}
    return {"global": summarize(df), "by_model": by_model}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _metrics(y_true, pred_log):
    pred = np.expm1(pred_log)
    return {
        "mae_pkr": float(mean_absolute_error(y_true, pred)),
        "mdape": float(np.median(np.abs(pred - y_true) / y_true)),
    }


def train_all(train_df, test_df, configs, out_dir, tracking_uri=None,
              experiment="carcompass-price", data_hash="unknown"):
    """Train every config, log to MLflow, save the one with the lowest test MAE."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    fit_df, cal_df = train_test_split(train_df, test_size=0.15, random_state=42)
    y_fit, y_cal = np.log1p(fit_df["price"]), np.log1p(cal_df["price"])
    y_test = test_df["price"].to_numpy()
    commit = git_commit()

    results = []
    for cfg in configs:
        name, params = cfg["model"], cfg.get("params", {})
        try:
            pipe = build_pipeline(name, params)
        except ImportError:
            print(f"Skipping {cfg['name']}: library for '{name}' is not installed.")
            continue
        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.log_params({"model": name, **{f"param_{k}": v for k, v in params.items()}})
            mlflow.log_param("data_hash", data_hash)
            mlflow.set_tag("git_commit", commit)

            pipe.fit(fit_df[FEATURES], y_fit)
            residuals = y_cal.to_numpy() - pipe.predict(cal_df[FEATURES])
            q10, q90 = float(np.quantile(residuals, 0.10)), float(np.quantile(residuals, 0.90))

            pred_log = pipe.predict(test_df[FEATURES])
            metrics = _metrics(y_test, pred_log)
            low, high = np.expm1(pred_log + q10), np.expm1(pred_log + q90)
            metrics["interval_coverage"] = float(np.mean((y_test >= low) & (y_test <= high)))
            mlflow.log_metrics(metrics)
            results.append({"cfg": cfg, "pipe": pipe, "metrics": metrics, "q10": q10,
                            "q90": q90, "run_id": run.info.run_id})
            print(f"{cfg['name']:<16} MAE={metrics['mae_pkr']:,.0f} "
                  f"MdAPE={metrics['mdape']:.1%} coverage={metrics['interval_coverage']:.0%}")

    if not results:
        raise RuntimeError("No model could be trained.")
    best = min(results, key=lambda r: r["metrics"]["mae_pkr"])
    mlflow.MlflowClient().set_tag(best["run_id"], "champion", "true")

    importance = permutation_importance(
        best["pipe"], test_df[FEATURES], np.log1p(test_df["price"]),
        n_repeats=5, random_state=42,
    )
    ranked = sorted(zip(FEATURES, importance.importances_mean), key=lambda x: -x[1])

    version = time.strftime("%Y%m%d-%H%M%S")
    meta = {
        "version": version,
        "run_id": best["run_id"],
        "model": best["cfg"]["model"],
        "run_name": best["cfg"]["name"],
        "params": best["cfg"].get("params", {}),
        "metrics": best["metrics"],
        "q10": best["q10"],
        "q90": best["q90"],
        "data_hash": data_hash,
        "git_commit": commit,
        "importances": [{"feature": f, "importance": round(float(i), 4)} for f, i in ranked],
        "defaults": typical_values(train_df),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    versioned = out_dir / f"price_model_{version}.joblib"
    joblib.dump(best["pipe"], versioned)
    shutil.copy(versioned, out_dir / MODEL_FILE)  # the file the API loads
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    print(f"Best run: {meta['run_name']} -> saved {versioned.name}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train price models.")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    configs = yaml.safe_load(Path(args.config).read_text())["experiments"]
    train_df, test_df = pd.read_csv(args.train), pd.read_csv(args.test)
    train_all(train_df, test_df, configs, args.out, data_hash=file_hash(args.train))


if __name__ == "__main__":
    main()
