"""Train price models, log every run to MLflow, save the champion.

What this script does, in order:
1. Trains every model in configs/train.yaml on the training data (one MLflow run each).
2. Picks the champion: the global model with the lowest test error (MAE).
3. Extras (on by default): trains separate pre-2010 and 2010+ "era" models with the
   champion's settings and compares them with the global champion, and checks
   LightGBM quantile models against the residual-based price range.
4. Saves the champion to models/ with metadata.json and VERSION.json, and registers
   it in the MLflow model registry with the alias "production" (needs a database
   tracking URI such as sqlite:///mlflow.db).

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

VERSION_FILE = "VERSION.json"
ERA_YEAR = 2010  # "pre-2010" is year < 2010, "2010plus" is year >= 2010
ERA_FILES = {"pre2010": "price_model_pre2010.joblib",
             "2010plus": "price_model_2010plus.joblib"}
REGISTERED_NAME = "carcompass-price"
MIN_ERA_ROWS = 30  # skip an era model if it would have fewer rows than this
SOURCE_COL = "source"  # "A" or "B"; only present when dataset B was added in prep
B_WEIGHT = 0.3  # a dataset B row counts 30% as much as a dataset A row when fitting


def build_estimator(name: str, params: dict):
    if name == "baseline":
        # strategy: median (default), mean or quantile (with a "quantile" param)
        return DummyRegressor(**{"strategy": "median", **params})
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


def fit_weights(df: pd.DataFrame):
    """Sample weights for fitting. Dataset B rows have older (2021) prices, so they
    count less. Returns None when the data has no 'source' column (A only)."""
    if SOURCE_COL not in df.columns:
        return None
    return np.where(df[SOURCE_COL] == "B", B_WEIGHT, 1.0)


def fit_pipeline(pipe, df, y):
    """pipe.fit that applies fit_weights when they exist."""
    weights = fit_weights(df)
    if weights is None:
        pipe.fit(df[FEATURES], y)
    else:
        pipe.fit(df[FEATURES], y, model__sample_weight=weights)


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


def fit_and_evaluate(pipe, fit_df, cal_df, test_df):
    """Fit on fit_df, set the price range from cal_df, score on test_df.

    The model predicts log(price). The 10th and 90th percentiles of its errors on
    cal_df (q10, q90) give the low/high price range; coverage says how often the
    real test price falls inside that range (the target is about 80%).
    """
    if SOURCE_COL in cal_df.columns:
        cal_df = cal_df[cal_df[SOURCE_COL] == "A"]  # range from current-price rows only
    y_fit, y_cal = np.log1p(fit_df["price"]), np.log1p(cal_df["price"])
    y_test = test_df["price"].to_numpy()
    fit_pipeline(pipe, fit_df, y_fit)
    residuals = y_cal.to_numpy() - pipe.predict(cal_df[FEATURES])
    q10, q90 = float(np.quantile(residuals, 0.10)), float(np.quantile(residuals, 0.90))

    pred_log = pipe.predict(test_df[FEATURES])
    metrics = _metrics(y_test, pred_log)
    low, high = np.expm1(pred_log + q10), np.expm1(pred_log + q90)
    metrics["interval_coverage"] = float(np.mean((y_test >= low) & (y_test <= high)))
    return metrics, q10, q90, pred_log


def train_era_models(best, fit_df, cal_df, test_df, out_dir, commit, data_hash):
    """Train pre-2010 and 2010+ models with the champion's settings.

    Returns a dict describing both, plus whether the two era models together beat
    the single global champion on the test set. Returns None if an era has too
    few rows.
    """
    cfg = best["cfg"]
    name, params = cfg["model"], cfg.get("params", {})
    masks = {
        "pre2010": lambda d: (d["year"] < ERA_YEAR).to_numpy(),
        "2010plus": lambda d: (d["year"] >= ERA_YEAR).to_numpy(),
    }
    pred_all = np.full(len(test_df), np.nan)
    eras = {}
    for key, mask in masks.items():
        f, c, t = fit_df[mask(fit_df)], cal_df[mask(cal_df)], test_df[mask(test_df)]
        if min(len(f), len(c), len(t)) < MIN_ERA_ROWS:
            print(f"Skipping era models: '{key}' has too few rows "
                  f"({len(f)} fit / {len(c)} cal / {len(t)} test).")
            return None
        pipe = build_pipeline(name, params)
        with mlflow.start_run(run_name=f"era_{key}_{cfg['name']}") as run:
            mlflow.log_params({"model": name, "scope": f"era_{key}",
                               **{f"param_{k}": v for k, v in params.items()}})
            mlflow.log_param("data_hash", data_hash)
            mlflow.set_tag("git_commit", commit)
            mlflow.set_tag("model_scope", f"era_{key}")
            metrics, q10, q90, pred = fit_and_evaluate(pipe, f, c, t)
            mlflow.log_metrics(metrics)
        pred_all[mask(test_df)] = pred
        joblib.dump(pipe, Path(out_dir) / ERA_FILES[key])
        eras[key] = {"file": ERA_FILES[key], "rows_train": len(f), "rows_test": len(t),
                     "q10": q10, "q90": q90, "metrics": metrics, "run_id": run.info.run_id}
        print(f"era {key:<9} MAE={metrics['mae_pkr']:,.0f} "
              f"MdAPE={metrics['mdape']:.1%} coverage={metrics['interval_coverage']:.0%}")

    combined = _metrics(test_df["price"].to_numpy(), pred_all)
    glob = best["metrics"]
    era_better = combined["mae_pkr"] < glob["mae_pkr"]
    with mlflow.start_run(run_name="era_vs_global_summary"):
        mlflow.set_tag("model_scope", "summary")
        mlflow.log_metrics({"global_mae_pkr": glob["mae_pkr"],
                            "era_combined_mae_pkr": combined["mae_pkr"],
                            "global_mdape": glob["mdape"],
                            "era_combined_mdape": combined["mdape"]})
    print(f"global MAE={glob['mae_pkr']:,.0f}  era-combined MAE={combined['mae_pkr']:,.0f}"
          f"  -> {'era models are better' if era_better else 'global model is better'}")
    return {"split_year": ERA_YEAR, "models": eras, "combined_metrics": combined,
            "era_better": bool(era_better)}


def quantile_models_check(fit_df, test_df, champion_coverage, data_hash, commit):
    """Train LightGBM quantile models (10% and 90%) and measure their coverage.

    This is a second way to get a price range, compared with the residual method
    used by the champion. Returns None if LightGBM is not installed.
    """
    params = {"n_estimators": 300, "learning_rate": 0.05}
    y_fit = np.log1p(fit_df["price"])
    y_test = test_df["price"].to_numpy()
    preds = []
    try:
        for alpha in (0.1, 0.9):
            pipe = build_pipeline("lightgbm", {**params, "objective": "quantile",
                                               "alpha": alpha})
            fit_pipeline(pipe, fit_df, y_fit)
            preds.append(np.expm1(pipe.predict(test_df[FEATURES])))
    except ImportError:
        print("Skipping quantile check: lightgbm is not installed.")
        return None
    low, high = np.minimum(*preds), np.maximum(*preds)
    result = {
        "coverage": float(np.mean((y_test >= low) & (y_test <= high))),
        "median_relative_width": float(np.median((high - low) / y_test)),
        "champion_residual_method_coverage": champion_coverage,
    }
    with mlflow.start_run(run_name="lgbm_quantile_10_90"):
        mlflow.log_params({"model": "lightgbm", "objective": "quantile",
                           "alphas": "0.1,0.9", "data_hash": data_hash,
                           **{f"param_{k}": v for k, v in params.items()}})
        mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("model_scope", "quantile_check")
        mlflow.log_metrics({"interval_coverage": result["coverage"],
                            "median_relative_width": result["median_relative_width"]})
    print(f"quantile check: coverage={result['coverage']:.0%} "
          f"(champion residual method: {champion_coverage:.0%})")
    return result


def register_champion(pipe, run_id):
    """Log the champion to MLflow and give it the alias 'production'.

    Needs a database tracking URI (for example sqlite:///mlflow.db). Returns the
    registered version number as a string, or None if registration is not possible.
    """
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.sklearn.log_model(pipe, "model", serialization_format="cloudpickle")
        version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_NAME).version
        mlflow.MlflowClient().set_registered_model_alias(
            REGISTERED_NAME, "production", version
        )
        print(f"Registered {REGISTERED_NAME} version {version} with alias 'production'.")
        return str(version)
    except Exception as exc:  # registration is a bonus; never stop training for it
        print(f"Could not register the champion in MLflow ({type(exc).__name__}): {exc}")
        return None


def train_all(train_df, test_df, configs, out_dir, tracking_uri=None,
              experiment="carcompass-price", data_hash="unknown",
              extras=False, register=False):
    """Train every config, log to MLflow, save the one with the lowest test MAE.

    extras=True also trains era models and the quantile check.
    register=True also registers the champion in the MLflow model registry.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    fit_df, cal_df = train_test_split(train_df, test_size=0.15, random_state=42)
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
            mlflow.set_tag("model_scope", "global")

            metrics, q10, q90, _ = fit_and_evaluate(pipe, fit_df, cal_df, test_df)
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    era = quantile = None
    if extras:
        era = train_era_models(best, fit_df, cal_df, test_df, out_dir, commit, data_hash)
        quantile = quantile_models_check(
            fit_df, test_df, best["metrics"]["interval_coverage"], data_hash, commit
        )
    registry_version = register_champion(best["pipe"], best["run_id"]) if register else None

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
        "era": era,
        "quantile_check": quantile,
    }
    versioned = out_dir / f"price_model_{version}.joblib"
    joblib.dump(best["pipe"], versioned)
    shutil.copy(versioned, out_dir / MODEL_FILE)  # the file the API loads
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))

    version_info = {
        "version": version,
        "champion": {"run_name": meta["run_name"], "model": meta["model"],
                     "params": meta["params"], "metrics": meta["metrics"],
                     "mlflow_run_id": meta["run_id"], "file": MODEL_FILE},
        "data_hash": data_hash,
        "git_commit": commit,
        "era_models": None if not era else {
            "split_year": era["split_year"],
            "better_than_global": era["era_better"],
            "files": {k: v["file"] for k, v in era["models"].items()},
        },
        "registered_model": None if not registry_version else {
            "name": REGISTERED_NAME, "version": registry_version, "alias": "production"},
        "all_runs": [{"run_name": r["cfg"]["name"], "model": r["cfg"]["model"],
                      **r["metrics"]} for r in results],
    }
    (out_dir / VERSION_FILE).write_text(json.dumps(version_info, indent=2))
    print(f"Best run: {meta['run_name']} -> saved {versioned.name} and {VERSION_FILE}")
    return meta


def resolve_tracking_uri(cli_uri=None) -> str:
    """Pick the MLflow tracking URI.

    MLflow turns the space in a folder name like 'CAYA Technology' into %20 and then
    tries to create a folder with that literal name, which fails on Windows. A
    relative SQLite path (next to where the command is run) avoids the problem.
    """
    uri = cli_uri or mlflow.get_tracking_uri()
    if "%20" in uri:
        return "sqlite:///mlflow.db"
    return uri


def main():
    parser = argparse.ArgumentParser(description="Train price models.")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--out", default="models")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: the MLflow default)")
    parser.add_argument("--no-extras", action="store_true",
                        help="skip era models and the quantile check")
    parser.add_argument("--no-register", action="store_true",
                        help="do not register the champion in the MLflow registry")
    args = parser.parse_args()

    configs = yaml.safe_load(Path(args.config).read_text())["experiments"]
    train_df, test_df = pd.read_csv(args.train), pd.read_csv(args.test)
    uri = resolve_tracking_uri(args.tracking_uri)
    has_database = uri.startswith(("sqlite", "postgresql", "mysql", "http"))
    train_all(train_df, test_df, configs, args.out, tracking_uri=uri,
              data_hash=file_hash(args.train),
              extras=not args.no_extras, register=has_database and not args.no_register)


if __name__ == "__main__":
    main()