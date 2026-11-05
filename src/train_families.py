"""Train the five price-model families, pick champions on validation, write a manifest.

Needs the shared dataset from src/prepare_multi.py. Run:

    python -m src.train_families

What it does, in order:
1. For every family in configs/model_families.yaml, trains every experiment in
   configs/train.yaml on that family's fit rows (one MLflow run each, in the MLflow
   experiment carcompass/<family>). The family champion is the run with the lowest
   VALIDATION MAE; test rows are never used to choose.
2. Calibrates the champion's 10-90% price range on the cal rows and reports its test
   scores (logged on the champion's MLflow run).
3. Applies configs/routing.yaml on the validation rows: picks the default model and
   switches on only the specialists that clearly beat it on their own years.
4. Compares every model on the SAME test rows and writes reports/model_comparison.md.
5. Saves models/<family>.joblib + .json, models/manifest.json (what the API serves)
   and a copy of the default model as models/price_model.joblib for older tools.
   Registers each champion in the MLflow registry (alias "champion"); the models the
   API serves also get the alias "production".
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

from src.model_utils import CATEGORICAL, FEATURES, META_FILE, MODEL_FILE
from src.prepare_data import file_hash
from src.train import _metrics, build_pipeline, git_commit, resolve_tracking_uri
from src.train import typical_values

MANIFEST_FILE = "manifest.json"
MIN_FAMILY_ROWS = 30  # a family with fewer fit, val, cal or test rows is skipped
ABLATION_FAMILY = "price_global_ab"  # the family used to test the assembly column


def family_mask(df: pd.DataFrame, family: dict) -> pd.Series:
    """True for rows that belong to the family (its sources and year range)."""
    mask = df["source"].isin(family["sources"])
    if "year_min" in family:
        mask &= df["year"] >= family["year_min"]
    if "year_max" in family:
        mask &= df["year"] <= family["year_max"]
    return mask


def year_mask(df: pd.DataFrame, rule: dict) -> np.ndarray:
    mask = pd.Series(True, index=df.index)
    if "year_min" in rule:
        mask &= df["year"] >= rule["year_min"]
    if "year_max" in rule:
        mask &= df["year"] <= rule["year_max"]
    return mask.to_numpy()


def fit_weights(df: pd.DataFrame, b_weight: float):
    """B rows count less only when a family mixes A and B."""
    if df["source"].nunique() < 2:
        return None
    return np.where(df["source"] == "B", b_weight, 1.0)


def fit(pipe, df: pd.DataFrame, weights):
    y = np.log1p(df["price"])
    if weights is None:
        return pipe.fit(df[FEATURES], y)
    return pipe.fit(df[FEATURES], y, model__sample_weight=weights)


def score(y_true, pred_log, q10=None, q90=None) -> dict:
    """MAE, MdAPE, row count and (with q10/q90) the share of prices inside the range."""
    y_true = np.asarray(y_true)
    out = {**_metrics(y_true, pred_log), "rows": int(len(y_true))}
    if q10 is not None:
        low, high = np.expm1(pred_log + q10), np.expm1(pred_log + q90)
        out["interval_coverage"] = float(np.mean((y_true >= low) & (y_true <= high)))
    return out


def registry_name(family: str) -> str:
    return "carcompass-" + family.replace("_", "-")


def register(pipe, run_id: str, family: str):
    """Log the champion's model and register it with the alias 'champion'."""
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.sklearn.log_model(pipe, name="model",
                                     serialization_format="cloudpickle")
        version = mlflow.register_model(f"runs:/{run_id}/model",
                                        registry_name(family)).version
        mlflow.MlflowClient().set_registered_model_alias(
            registry_name(family), "champion", version)
        return str(version)
    except Exception as exc:  # the files on disk are still saved
        print(f"Could not register {family} ({type(exc).__name__}): {exc}")
        return None


class _ByteCounter:
    """File-like object that only counts bytes, to size a model without saving it."""

    def __init__(self):
        self.n = 0

    def write(self, data):
        self.n += memoryview(data).nbytes  # data may be bytes or a PickleBuffer


def model_mb(pipe) -> float:
    counter = _ByteCounter()
    pickle.dump(pipe, counter, protocol=pickle.HIGHEST_PROTOCOL)
    return counter.n / 1e6


def logged_runs(name: str, data_hash: str) -> dict:
    """Finished runs of this family on this exact data (newest one per run name)."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[f"carcompass/{name}"],
            filter_string=(f"params.data_hash = '{data_hash}' "
                           "and attributes.status = 'FINISHED'"),
            order_by=["attributes.start_time DESC"],
        )
    except Exception:
        return {}
    found = {}
    for _, row in runs.iterrows():
        run_name = row.get("tags.mlflow.runName")
        if (run_name in found or row.get("tags.ablation") == "no_assembly"
                or pd.isna(row.get("metrics.val_mae_pkr"))):
            continue
        size = row.get("metrics.model_mb")
        found[run_name] = {"run_id": row["run_id"],
                           "val_mae_pkr": float(row["metrics.val_mae_pkr"]),
                           "val_mdape": float(row.get("metrics.val_mdape", np.nan)),
                           "model_mb": None if pd.isna(size) else float(size)}
    return found


def train_family(name, family, df, configs, b_weight, out_dir, data_hash, commit,
                 do_register, max_model_mb=None, resume=False):
    """Train one family. Returns its summary, or None when it has too few rows.

    The champion is the run with the lowest validation MAE whose model is at most
    max_model_mb (None = no limit). With resume, runs already logged in MLflow on the
    same data are not trained again; only the chosen champion is refitted.
    """
    mask = family_mask(df, family)
    part = {s: df[mask & (df["split"] == s)] for s in ("fit", "val", "cal", "test")}
    sizes = {s: len(p) for s, p in part.items()}
    if min(sizes.values()) < MIN_FAMILY_ROWS:
        print(f"Skipping {name}: too few rows {sizes}.")
        return None

    mlflow.set_experiment(f"carcompass/{name}")
    weights = fit_weights(part["fit"], b_weight)
    y_val = part["val"]["price"].to_numpy()
    done = logged_runs(name, data_hash) if resume else {}
    limit = max_model_mb if max_model_mb else float("inf")

    def train_one(cfg, run_id=None):
        """Fit one config (into run_id if given) and log its val MAE and size."""
        params = cfg.get("params", {})
        pipe = build_pipeline(cfg["model"], params)
        with mlflow.start_run(run_id=run_id, run_name=cfg["name"]) as run:
            if run_id is None:
                mlflow.log_params({"family": name, "model": cfg["model"],
                                   "data_hash": data_hash, "rows_fit": sizes["fit"],
                                   **{f"param_{k}": v for k, v in params.items()}})
                mlflow.set_tags({"git_commit": commit, "family": name})
            start = time.perf_counter()
            fit(pipe, part["fit"], weights)
            val = _metrics(y_val, pipe.predict(part["val"][FEATURES]))
            size = model_mb(pipe)
            mlflow.log_metrics({"val_mae_pkr": val["mae_pkr"],
                                "val_mdape": val["mdape"], "model_mb": size,
                                "fit_seconds": time.perf_counter() - start})
        return pipe, {"run_id": run.info.run_id, "val_mae_pkr": val["mae_pkr"],
                      "val_mdape": val["mdape"], "model_mb": size}

    best, runs = None, []
    print(f"\n== {name}  rows {sizes}  (champion size limit: {max_model_mb} MB)")
    for cfg in configs:
        if cfg["name"] in done:
            rec, pipe, note = dict(done[cfg["name"]]), None, "logged before"
        else:
            try:
                pipe, rec = train_one(cfg)
            except ImportError:
                print(f"Skipping {cfg['name']}: library for '{cfg['model']}' missing.")
                continue
            note = f"{rec['model_mb']:.1f} MB"
        runs.append({"run_name": cfg["name"], "model": cfg["model"], **rec})
        print(f"  {cfg['name']:<16} val MAE={rec['val_mae_pkr']:>12,.0f}  "
              f"MdAPE={rec['val_mdape']:.1%}  ({note})")
        fits = pipe is not None and rec["model_mb"] <= limit
        if fits and (best is None or rec["val_mae_pkr"] < best["val_mae_pkr"]):
            best = {"cfg": cfg, "pipe": pipe, **rec}
        del pipe
        gc.collect()

    # A logged run that scores better than the best fitted one is refitted when its
    # size is unknown or within the limit.
    by_name = {c["name"]: c for c in configs}
    for rec in sorted(runs, key=lambda r: r["val_mae_pkr"]):
        if best is not None and rec["val_mae_pkr"] >= best["val_mae_pkr"]:
            break
        if rec["run_name"] not in done or (rec["model_mb"] or 0) > limit:
            continue
        print(f"  refitting logged run {rec['run_name']} as champion candidate")
        pipe, new = train_one(by_name[rec["run_name"]], run_id=rec["run_id"])
        rec.update(new)
        if new["model_mb"] <= limit:
            best = {"cfg": by_name[rec["run_name"]], "pipe": pipe, **new}
            break
        del pipe
        gc.collect()
    if best is None:
        raise RuntimeError(f"{name}: no model fits the {max_model_mb} MB limit.")
    best["val"] = {"mae_pkr": best["val_mae_pkr"]}

    client = mlflow.MlflowClient()
    for old in mlflow.search_runs(experiment_names=[f"carcompass/{name}"],
                                  filter_string="tags.champion = 'true'")["run_id"]:
        if old != best["run_id"]:
            client.set_tag(old, "champion", "false")

    pipe, cfg = best["pipe"], best["cfg"]
    cal = part["cal"]
    residuals = np.log1p(cal["price"].to_numpy()) - pipe.predict(cal[FEATURES])
    q10, q90 = (float(np.quantile(residuals, q)) for q in (0.10, 0.90))
    test = score(part["test"]["price"], pipe.predict(part["test"][FEATURES]), q10, q90)
    val_scores = score(y_val, pipe.predict(part["val"][FEATURES]), q10, q90)

    sample = part["val"].sample(min(2000, sizes["val"]), random_state=42)
    imp = permutation_importance(pipe, sample[FEATURES], np.log1p(sample["price"]),
                                 n_repeats=3, random_state=42)
    ranked = sorted(zip(FEATURES, imp.importances_mean), key=lambda x: -x[1])

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("champion", "true")
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_metrics({"val_interval_coverage": val_scores["interval_coverage"],
                            "q10": q10, "q90": q90})
    version = register(pipe, best["run_id"], name) if do_register else None

    # Predictions on ALL shared val/test rows, for the comparisons and routing.
    shared = {s: pipe.predict(df.loc[df["split"] == s, FEATURES]) for s in ("val", "test")}

    stamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {
        "version": f"{name}@{stamp}",
        "family": name,
        "family_rule": family,
        "run_id": best["run_id"],
        "model": cfg["model"],
        "run_name": cfg["name"],
        "params": cfg.get("params", {}),
        "metrics": test,
        "val_metrics": val_scores,
        "rows": sizes,
        "q10": q10,
        "q90": q90,
        "data_hash": data_hash,
        "git_commit": commit,
        "registered_model": None if not version else {
            "name": registry_name(name), "version": version},
        "model_mb": round(best["model_mb"], 2),
        "importances": [{"feature": f, "importance": round(float(i), 4)}
                        for f, i in ranked],
        "defaults": typical_values(part["fit"]),
        "all_runs": runs,
    }
    joblib.dump(pipe, Path(out_dir) / f"{name}.joblib")
    (Path(out_dir) / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"  champion {cfg['name']}: test MAE={test['mae_pkr']:,.0f} "
          f"MdAPE={test['mdape']:.1%} coverage={test['interval_coverage']:.0%}")
    summary = {"meta": meta, "shared": shared, "cfg": cfg}
    del pipe, best
    gc.collect()
    return summary


def assembly_ablation(summary, df, family, b_weight, data_hash, commit):
    """Retrain the family champion's settings WITHOUT the assembly column."""
    cfg = summary["cfg"]
    part = {s: df[family_mask(df, family) & (df["split"] == s)] for s in ("fit", "val")}
    without = [c for c in CATEGORICAL if c != "assembly"]
    pipe = build_pipeline(cfg["model"], cfg.get("params", {}), categorical=without)
    mlflow.set_experiment(f"carcompass/{ABLATION_FAMILY}")
    with mlflow.start_run(run_name=f"ablation_no_assembly_{cfg['name']}"):
        mlflow.set_tags({"git_commit": commit, "family": ABLATION_FAMILY,
                         "ablation": "no_assembly"})
        mlflow.log_param("data_hash", data_hash)
        fit(pipe, part["fit"], fit_weights(part["fit"], b_weight))
        val = _metrics(part["val"]["price"].to_numpy(), pipe.predict(part["val"][FEATURES]))
        mlflow.log_metrics({"val_mae_pkr": val["mae_pkr"], "val_mdape": val["mdape"]})
    with_mae = summary["meta"]["val_metrics"]["mae_pkr"]
    print(f"assembly check: val MAE with={with_mae:,.0f} without={val['mae_pkr']:,.0f}")
    return {"run_name": cfg["name"], "val_mae_with_assembly": with_mae,
            "val_mae_without_assembly": val["mae_pkr"],
            "assembly_helps": bool(with_mae < val["mae_pkr"])}


def decide_routing(trained, df, routing):
    """Pick the default model and the specialists, using validation rows only."""
    val = df[df["split"] == "val"]
    y = val["price"].to_numpy()
    candidates = {n: _metrics(y, trained[n]["shared"]["val"])["mae_pkr"]
                  for n in routing["default_candidates"] if n in trained}
    if not candidates:
        raise RuntimeError("No default candidate was trained.")
    default = min(candidates, key=candidates.get)

    specialists = {}
    for name, rule in routing.get("specialists", {}).items():
        if name not in trained:
            continue
        m = year_mask(val, rule)
        spec = _metrics(y[m], trained[name]["shared"]["val"][m])["mae_pkr"]
        base = _metrics(y[m], trained[default]["shared"]["val"][m])["mae_pkr"]
        gain = 1 - spec / base
        specialists[name] = {**rule, "val_mae_pkr": spec,
                             "default_val_mae_pkr_same_rows": base,
                             "val_gain": gain,
                             "enabled": bool(gain >= routing.get("min_gain", 0.0))}
    return {"default": default, "default_candidates_val_mae_pkr": candidates,
            "min_gain": routing.get("min_gain", 0.0), "specialists": specialists}


def routed_predictions(trained, df, decision, split="test"):
    rows = df[df["split"] == split]
    pred = trained[decision["default"]]["shared"][split].copy()
    for name, spec in decision["specialists"].items():
        if spec["enabled"]:
            m = year_mask(rows, spec)
            pred[m] = trained[name]["shared"][split][m]
    return pred


def compare_on_test(trained, df, decision):
    """Every model scored on the same test rows (all, A only, B only, by era)."""
    test = df[df["split"] == "test"]
    y = test["price"].to_numpy()
    subsets = {"all": np.ones(len(test), bool),
               "A": (test["source"] == "A").to_numpy(),
               "B": (test["source"] == "B").to_numpy(),
               "pre2010": (test["year"] < 2010).to_numpy(),
               "2010plus": (test["year"] >= 2010).to_numpy()}
    rows = {}
    for name, t in trained.items():
        rows[name] = {k: score(y[m], t["shared"]["test"][m]) for k, m in subsets.items()}
    routed = routed_predictions(trained, df, decision)
    rows["served_system"] = {k: score(y[m], routed[m]) for k, m in subsets.items()}
    return rows


def fmt(v):
    return f"{v:,.0f}"


def write_comparison(path, trained, decision, comparison, ablation, report_rows):
    lines = ["# Model comparison: five price families", "",
             f"Shared dataset: {report_rows['total']} listings, test rows: "
             f"{report_rows['test']} (A {report_rows['test_A']}, B "
             f"{report_rows['test_B']}). B prices are scaled to A's price level.",
             "Champions are chosen on validation rows; test rows are only reported.",
             "", "## Family champions (each on its own rows)", "",
             "| Family | Champion | Train rows | Val MAE | Test MAE | Test MdAPE "
             "| Range coverage |", "|---|---|---|---|---|---|---|"]
    for name, t in trained.items():
        m = t["meta"]
        lines.append(
            f"| `{name}` | `{m['run_name']}` | {m['rows']['fit']} | "
            f"{fmt(m['val_metrics']['mae_pkr'])} | {fmt(m['metrics']['mae_pkr'])} | "
            f"{m['metrics']['mdape']:.1%} | {m['metrics']['interval_coverage']:.0%} |")
    lines += ["", "## Same test rows for every model (MAE in PKR)", "",
              "| Model | All test | A rows | B rows | Pre-2010 | 2010+ |",
              "|---|---|---|---|---|---|"]
    for name, r in comparison.items():
        label = "**served system**" if name == "served_system" else f"`{name}`"
        lines.append(f"| {label} | " + " | ".join(
            fmt(r[k]["mae_pkr"]) for k in ("all", "A", "B", "pre2010", "2010plus")) + " |")
    lines += ["", "Era specialists only make sense on their own years, so read them in "
              "the Pre-2010 or 2010+ column.", "", "## Routing decision (validation rows)",
              "", "| Default candidate | Val MAE (all val rows) |", "|---|---|"]
    for name, mae in decision["default_candidates_val_mae_pkr"].items():
        mark = " ← default" if name == decision["default"] else ""
        lines.append(f"| `{name}`{mark} | {fmt(mae)} |")
    lines += ["", f"A specialist is switched on if it beats the default by at least "
              f"{decision['min_gain']:.0%} on its own years.", "",
              "| Specialist | Its val MAE | Default on same rows | Gain | Switched on |",
              "|---|---|---|---|---|"]
    for name, s in decision["specialists"].items():
        lines.append(f"| `{name}` | {fmt(s['val_mae_pkr'])} | "
                     f"{fmt(s['default_val_mae_pkr_same_rows'])} | {s['val_gain']:+.1%} | "
                     f"{'yes' if s['enabled'] else 'no'} |")
    if ablation:
        lines += ["", "## Does the assembly column help?", "",
                  f"`{ABLATION_FAMILY}` champion settings (`{ablation['run_name']}`): "
                  f"val MAE {fmt(ablation['val_mae_with_assembly'])} with assembly, "
                  f"{fmt(ablation['val_mae_without_assembly'])} without -> assembly "
                  f"{'helps' if ablation['assembly_helps'] else 'does not help'}."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_families(df, fam_cfg, routing, configs, out_dir, report_path,
                   data_hash="unknown", tracking_uri=None, do_register=True,
                   ablation=True, resume=False) -> dict:
    """Train every family, decide routing, write the manifest and report."""
    commit = git_commit()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b_weight = fam_cfg.get("b_weight", 1.0)

    trained = {}
    for name, family in fam_cfg["families"].items():
        summary = train_family(name, family, df, configs, b_weight, out, data_hash,
                               commit, do_register, fam_cfg.get("max_model_mb"),
                               resume)
        if summary:
            trained[name] = summary

    check = None
    if ablation and ABLATION_FAMILY in trained:
        check = assembly_ablation(trained[ABLATION_FAMILY], df,
                                  fam_cfg["families"][ABLATION_FAMILY], b_weight,
                                  data_hash, commit)

    decision = decide_routing(trained, df, routing)
    comparison = compare_on_test(trained, df, decision)
    served = [decision["default"]] + [n for n, s in decision["specialists"].items()
                                      if s["enabled"]]
    if do_register:
        client = mlflow.MlflowClient()
        for name in served:
            reg = trained[name]["meta"]["registered_model"]
            if reg:
                client.set_registered_model_alias(reg["name"], "production",
                                                  reg["version"])

    manifest = {
        "version": time.strftime("%Y%m%d-%H%M%S"),
        "data_hash": data_hash,
        "git_commit": commit,
        "default": decision["default"],
        "served": served,
        "routing": decision,
        "families": {
            name: {"file": f"{name}.joblib", "meta": f"{name}.json",
                   "role": fam_cfg["families"][name].get("role"),
                   "run_name": t["meta"]["run_name"],
                   "version": t["meta"]["version"],
                   "val_mae_pkr": t["meta"]["val_metrics"]["mae_pkr"],
                   "test_mae_pkr": t["meta"]["metrics"]["mae_pkr"],
                   "registered_model": t["meta"]["registered_model"]}
            for name, t in trained.items()
        },
        "test_comparison_mae_pkr": {n: {k: v["mae_pkr"] for k, v in r.items()}
                                    for n, r in comparison.items()},
        "assembly_check": check,
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    shutil.copy(out / f"{decision['default']}.joblib", out / MODEL_FILE)
    shutil.copy(out / f"{decision['default']}.json", out / META_FILE)

    test = df[df["split"] == "test"]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    write_comparison(report_path, trained, decision, comparison, check, {
        "total": len(df), "test": len(test),
        "test_A": int((test["source"] == "A").sum()),
        "test_B": int((test["source"] == "B").sum())})
    print(f"\nDefault model: {decision['default']}. Served: {', '.join(served)}.")
    print(f"Wrote {out / MANIFEST_FILE} and {report_path}.")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Train the five price-model families.")
    p.add_argument("--families", default="configs/model_families.yaml")
    p.add_argument("--routing", default="configs/routing.yaml")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--data", default=None, help="overrides the path in --families")
    p.add_argument("--out", default="models")
    p.add_argument("--report", default="reports/model_comparison.md")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--only", nargs="*", help="train only these experiment names")
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="reuse runs already logged in MLflow on the same data")
    args = p.parse_args()

    fam_cfg = yaml.safe_load(Path(args.families).read_text())
    routing = yaml.safe_load(Path(args.routing).read_text())
    configs = yaml.safe_load(Path(args.config).read_text())["experiments"]
    if args.only:
        configs = [c for c in configs if c["name"] in args.only]
    data_path = args.data or fam_cfg["data"]
    train_families(pd.read_csv(data_path), fam_cfg, routing, configs, args.out,
                   args.report, data_hash=file_hash(data_path),
                   tracking_uri=resolve_tracking_uri(args.tracking_uri),
                   do_register=not args.no_register, ablation=not args.no_ablation,
                   resume=args.resume)


if __name__ == "__main__":
    main()
