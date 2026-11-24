"""Beat attempt: model soups + distilled weight-space programming.

Goal: beat `price_global_ab` (LightGBM champion) on the shared test MAE.

What this run does:
1. Retrains the champion LightGBM config across several seeds and soups their
   log-price predictions (tree-model analogue of a weight-space soup).
2. Distills a stronger MLP from soft (teacher) + hard labels, then runs
   same-basin WSP (forks, greedy soup, WiSE-FT toward the teacher residual).
3. Selects prediction-space blends on validation only.

Run:

    python -m src.train_wsp_beat

Writes reports/wsp_comparison.md and models/wsp_best.joblib.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model_utils import CATEGORICAL, FEATURES, NUMERIC
from src.train import git_commit, resolve_tracking_uri
from src.train_wsp import (
    CHAMPION_FILES,
    OUT_DIR,
    REPORT_PATH,
    average_weights,
    clone_weights,
    mae_pkr,
    mix_weights,
    set_weights,
)

DATA_PATH = Path("data/processed_multi/listings.csv")
# Bagging/feature fraction so seeds actually diverge (defaults are deterministic).
LGBM_PARAMS = dict(
    n_estimators=900,
    learning_rate=0.02,
    num_leaves=127,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
)
# Slight hyperparameter diversity around the champion config.
LGBM_VARIANTS = [
    {"seed": 42, **LGBM_PARAMS},
    {"seed": 7, **LGBM_PARAMS},
    {"seed": 99, **LGBM_PARAMS},
    {"seed": 123, **LGBM_PARAMS},
    {"seed": 202, **{**LGBM_PARAMS, "num_leaves": 95}},
    {"seed": 404, **{**LGBM_PARAMS, "num_leaves": 159, "learning_rate": 0.018}},
    {"seed": 777, **{**LGBM_PARAMS, "n_estimators": 1100, "learning_rate": 0.017}},
    {"seed": 11, **{**LGBM_PARAMS, "subsample": 0.75, "colsample_bytree": 0.75}},
    {"seed": 22, **{**LGBM_PARAMS, "num_leaves": 63, "n_estimators": 1200, "learning_rate": 0.025}},
]
HIDDEN = (384, 192, 96)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Extra numeric signals that help the MLP catch up to trees."""
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


MLP_NUM = NUMERIC + ["vehicle_age", "log_mileage", "miles_per_year", "log_engine"]
MLP_CAT = CATEGORICAL + ["make_model"]
MLP_FEATURES = MLP_NUM + MLP_CAT


def build_mlp_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), MLP_NUM),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), MLP_CAT),
    ])


def build_lgbm(seed: int, **params) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    cfg = {**LGBM_PARAMS, **params}
    model = LGBMRegressor(random_state=seed, verbose=-1, **cfg)
    return Pipeline([("pre", pre), ("model", model)])


def make_mlp(seed: int, max_iter: int = 160) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="adam",
        learning_rate_init=6e-4,
        alpha=1e-5,
        batch_size=512,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=15,
        tol=1e-4,
        warm_start=False,
    )


def fit_mlp(mlp, X, y, sample_weight=None, resample_seed: int = 42):
    if sample_weight is None:
        mlp.fit(X, y)
        return mlp
    w = np.asarray(sample_weight, dtype=float)
    w = w / w.mean()
    rng = np.random.default_rng(resample_seed)
    idx = rng.choice(len(y), size=len(y), replace=True, p=w / w.sum())
    mlp.fit(X[idx], y[idx])
    return mlp


def score_log_preds(y_price, pred_log) -> dict:
    return {"mae_pkr": mae_pkr(y_price, pred_log), "rows": int(len(y_price))}


def log_record(records, name, val, test, note, artifact=None, params=None, tags=None):
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"program": name, **(params or {})})
        mlflow.log_metrics({"val_mae_pkr": val["mae_pkr"], "test_mae_pkr": test["mae_pkr"]})
        if tags:
            mlflow.set_tags(tags)
        if note:
            mlflow.set_tag("note", note[:250])
    row = {
        "name": name,
        "val_mae_pkr": val["mae_pkr"],
        "test_mae_pkr": test["mae_pkr"],
        "note": note,
    }
    if artifact is not None:
        row["artifact"] = artifact
    records.append(row)
    print(f"  {name}: val {val['mae_pkr']:,.0f}  test {test['mae_pkr']:,.0f}", flush=True)


def write_report(records, champions, best, elapsed, prev_best_test, lgbm_test):
    ordered = sorted(records, key=lambda r: r["val_mae_pkr"])
    lines = [
        "# Weight-space programming — beat attempt (v3)",
        "",
        "This run adds LightGBM seed soups (prediction-space model soup), a",
        "teacher-distilled MLP with richer features, same-basin WSP, and",
        "validation-selected prediction blends.",
        "",
        f"Elapsed training time: {elapsed/60:.1f} minutes.",
        "",
        "## Existing LightGBM champions (same test rows)",
        "",
        "| Model | Test MAE (PKR) |",
        "|---|---:|",
    ]
    for row in champions:
        mae = "—" if row.get("test_mae_pkr") is None else f"{row['test_mae_pkr']:,.0f}"
        lines.append(f"| `{row['name']}` | {mae} |")

    lines += [
        "",
        "## Programs (sorted by validation MAE)",
        "",
        "| Program | Val MAE (PKR) | Test MAE (PKR) | Notes |",
        "|---|---:|---:|---|",
    ]
    for row in ordered:
        mark = " **best**" if row["name"] == best["name"] else ""
        lines.append(
            f"| `{row['name']}`{mark} | {row['val_mae_pkr']:,.0f} | "
            f"{row['test_mae_pkr']:,.0f} | {row.get('note', '')} |"
        )

    lines += ["", "## Does it beat `price_global_ab`?", ""]
    delta = best["test_mae_pkr"] - lgbm_test
    if delta < 0:
        lines.append(
            f"**Yes.** Best program `{best['name']}` is {abs(delta):,.0f} PKR better "
            f"on test ({best['test_mae_pkr']:,.0f} vs {lgbm_test:,.0f})."
        )
    else:
        lines.append(
            f"**No.** Best program `{best['name']}` is {delta:,.0f} PKR behind "
            f"({best['test_mae_pkr']:,.0f} vs {lgbm_test:,.0f})."
        )

    if prev_best_test is not None:
        gain = prev_best_test - best["test_mae_pkr"]
        lines += [
            "",
            "## vs previous WSP best",
            "",
            f"Previous best test MAE {prev_best_test:,.0f} → this run "
            f"{best['test_mae_pkr']:,.0f} ({'better' if gain > 0 else 'worse'} by "
            f"{abs(gain):,.0f}).",
        ]

    lines += [
        "",
        "## Method (v3)",
        "",
        "1. Retrain champion LightGBM config (`lgbm_900_lr02`) on several seeds; "
        "soup log-price predictions.",
        "2. Distill MLP `(384,192,96)` on soft teacher targets mixed with hard "
        "labels; richer features (age, log mileage, make_model).",
        "3. Same-basin forks + greedy weight soup + WiSE-FT toward the teacher.",
        "4. Prediction blends of teacher / LGBM soup / WSP; α chosen on validation.",
        "5. Best program = lowest validation MAE; test MAE reported once.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--data", default=str(DATA_PATH))
    args = parser.parse_args()

    started = time.perf_counter()
    prev_best_test = None
    prev_json = OUT_DIR / "wsp_best.json"
    if prev_json.exists():
        try:
            prev = json.loads(prev_json.read_text(encoding="utf-8"))
            prev_best_test = float(prev["test_mae_pkr"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            prev_best_test = None

    df = enrich(pd.read_csv(args.data))
    fit_df = df[df["split"] == "fit"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    y_fit = np.log1p(fit_df["price"].to_numpy())
    y_val = val_df["price"].to_numpy()
    y_test = test_df["price"].to_numpy()
    sw = np.where(fit_df["source"].to_numpy() == "B", 0.3, 1.0)

    mlflow.set_tracking_uri(resolve_tracking_uri(args.tracking_uri))
    mlflow.set_experiment("carcompass/wsp")
    records: list[dict] = []

    # ---- Champion baseline ----
    champ_path = CHAMPION_FILES["price_global_ab"]
    if not champ_path.exists():
        raise SystemExit(f"Missing champion model at {champ_path}")
    champion = joblib.load(champ_path)
    champ_val = score_log_preds(y_val, champion.predict(val_df[FEATURES]))
    champ_test = score_log_preds(y_test, champion.predict(test_df[FEATURES]))
    log_record(
        records, "lgbm_champion_disk", champ_val, champ_test,
        "Served price_global_ab on disk",
        tags={"kind": "baseline", "git_commit": git_commit()},
    )
    lgbm_test_mae = champ_test["mae_pkr"]
    teacher_fit = champion.predict(fit_df[FEATURES])
    teacher_val = champion.predict(val_df[FEATURES])
    teacher_test = champion.predict(test_df[FEATURES])

    # ---- LightGBM seed soup ----
    print("Training LightGBM variants for prediction soup…", flush=True)
    lgbm_val_preds = []
    lgbm_test_preds = []
    lgbm_pipes = []
    variant_labels = []
    for cfg in LGBM_VARIANTS:
        seed = cfg["seed"]
        params = {k: v for k, v in cfg.items() if k != "seed"}
        name = f"lgbm_seed_{seed}"
        pipe = build_lgbm(seed, **params)
        pipe.fit(fit_df[FEATURES], y_fit, model__sample_weight=sw)
        pv = pipe.predict(val_df[FEATURES])
        pt = pipe.predict(test_df[FEATURES])
        lgbm_val_preds.append(pv)
        lgbm_test_preds.append(pt)
        lgbm_pipes.append(pipe)
        variant_labels.append(seed)
        log_record(
            records, name,
            score_log_preds(y_val, pv), score_log_preds(y_test, pt),
            f"LGBM variant seed={seed} leaves={params.get('num_leaves')}",
            params={"seed": seed, **params},
            tags={"kind": "lgbm_seed"},
        )

    # Include the on-disk champion in the soup pool.
    lgbm_val_preds.append(teacher_val)
    lgbm_test_preds.append(teacher_test)
    lgbm_pipes.append(champion)
    variant_labels.append("disk")

    # Uniform soup + greedy soup over seeds (prediction space)
    stack_val = np.stack(lgbm_val_preds, axis=0)
    stack_test = np.stack(lgbm_test_preds, axis=0)
    uniform_val = stack_val.mean(axis=0)
    uniform_test = stack_test.mean(axis=0)
    log_record(
        records, "lgbm_soup_uniform",
        score_log_preds(y_val, uniform_val), score_log_preds(y_test, uniform_test),
        f"Mean of {len(variant_labels)} LGBM variants (+disk)",
        artifact={"kind": "pred", "val": uniform_val, "test": uniform_test},
        tags={"kind": "lgbm_soup"},
    )

    # Greedy prediction soup
    order = sorted(range(len(variant_labels)), key=lambda i: mae_pkr(y_val, stack_val[i]))
    chosen = [order[0]]
    best_g = mae_pkr(y_val, stack_val[order[0]])
    for i in order[1:]:
        trial = stack_val[chosen + [i]].mean(axis=0)
        score = mae_pkr(y_val, trial)
        if score < best_g - 1.0:
            chosen.append(i)
            best_g = score
    greedy_val = stack_val[chosen].mean(axis=0)
    greedy_test = stack_test[chosen].mean(axis=0)
    log_record(
        records, "lgbm_soup_greedy",
        score_log_preds(y_val, greedy_val), score_log_preds(y_test, greedy_test),
        f"Greedy LGBM soup members {[variant_labels[i] for i in chosen]}",
        artifact={"kind": "pred", "val": greedy_val, "test": greedy_test},
        tags={"kind": "lgbm_soup"},
    )

    # Blend disk champion with seed soups (α on val)
    for soup_name, soup_val, soup_test in (
        ("uniform", uniform_val, uniform_test),
        ("greedy", greedy_val, greedy_test),
    ):
        best_a, best_v, best_t = 0.0, champ_val["mae_pkr"], champ_test["mae_pkr"]
        for alpha in np.linspace(0.0, 1.0, 21):
            pv = (1 - alpha) * teacher_val + alpha * soup_val
            score = mae_pkr(y_val, pv)
            if score < best_v:
                best_v = score
                best_a = float(alpha)
                best_t = mae_pkr(y_test, (1 - alpha) * teacher_test + alpha * soup_test)
        log_record(
            records, f"lgbm_champ_x_{soup_name}_a{best_a:g}",
            {"mae_pkr": best_v, "rows": len(y_val)},
            {"mae_pkr": best_t, "rows": len(y_test)},
            f"disk champion + {best_a:g}·{soup_name} soup (α on val)",
            artifact={
                "kind": "pred",
                "val": (1 - best_a) * teacher_val + best_a * soup_val,
                "test": (1 - best_a) * teacher_test + best_a * soup_test,
            },
            tags={"kind": "lgbm_blend"},
        )

    # ---- Distilled MLP + WSP ----
    print("Distilling MLP from LightGBM teacher…", flush=True)
    pre = build_mlp_preprocessor()
    X_fit = pre.fit_transform(fit_df[MLP_FEATURES])
    X_val = pre.transform(val_df[MLP_FEATURES])
    X_test = pre.transform(test_df[MLP_FEATURES])
    if hasattr(X_fit, "toarray"):
        X_fit, X_val, X_test = X_fit.toarray(), X_val.toarray(), X_test.toarray()

    # Soft→hard curriculum: first fit soft targets, then WiSE toward hard labels
    y_soft = 0.65 * teacher_fit + 0.35 * y_fit
    soft_mlp = make_mlp(seed=42, max_iter=120)
    fit_mlp(soft_mlp, X_fit, y_soft, sw, resample_seed=42)
    soft_w = clone_weights(soft_mlp)
    soft_val = score_log_preds(y_val, soft_mlp.predict(X_val))
    soft_test = score_log_preds(y_test, soft_mlp.predict(X_test))
    log_record(
        records, "mlp_distill_soft", soft_val, soft_test,
        "MLP distilled on 0.65·teacher + 0.35·hard",
        tags={"kind": "wsp_distill"},
    )

    hard_mlp = copy.deepcopy(soft_mlp)
    hard_mlp.warm_start = True
    hard_mlp.max_iter = 40
    fit_mlp(hard_mlp, X_fit, y_fit, sw, resample_seed=42)
    hard_w = clone_weights(hard_mlp)
    hard_val = score_log_preds(y_val, hard_mlp.predict(X_val))
    hard_test = score_log_preds(y_test, hard_mlp.predict(X_test))
    log_record(
        records, "mlp_distill_hard", hard_val, hard_test,
        "Soft MLP fine-tuned on hard labels",
        tags={"kind": "wsp_distill"},
    )

    # WiSE-FT between soft and hard
    probe = copy.deepcopy(soft_mlp)
    best_wise = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = mix_weights(soft_w, hard_w, alpha)
        set_weights(probe, *w)
        pv, pt = probe.predict(X_val), probe.predict(X_test)
        val = score_log_preds(y_val, pv)
        test = score_log_preds(y_test, pt)
        name = f"wsp_wise_distill_a{alpha:g}"
        log_record(
            records, name, val, test,
            f"WiSE-FT soft→hard α={alpha:g}",
            artifact={"kind": "mlp", "weights": w, "model": copy.deepcopy(probe)},
            tags={"kind": "wsp_program"},
        )
        if best_wise is None or val["mae_pkr"] < best_wise[0]:
            best_wise = (val["mae_pkr"], w, copy.deepcopy(probe), name)

    base = best_wise[2]
    base_w = best_wise[1]
    set_weights(base, *base_w)

    # Same-basin forks from distilled WiSE checkpoint
    print("Same-basin forks from distilled MLP…", flush=True)
    weight_bank = {"mlp_base": base_w}
    basin_names = []
    for seed in (7, 99, 123, 202, 404):
        name = f"mlp_basin_{seed}"
        fork = copy.deepcopy(base)
        fork.warm_start = True
        fork.max_iter = 25
        fork.random_state = seed
        fit_mlp(fork, X_fit, y_fit, sw, resample_seed=seed)
        weight_bank[name] = clone_weights(fork)
        basin_names.append(name)
        log_record(
            records, name,
            score_log_preds(y_val, fork.predict(X_val)),
            score_log_preds(y_test, fork.predict(X_test)),
            f"Basin fork seed {seed} from distilled WiSE",
            artifact={"kind": "mlp", "weights": weight_bank[name]},
            tags={"kind": "wsp_basin"},
        )

    # Uniform + greedy weight soups
    soup = average_weights([weight_bank["mlp_base"]] + [weight_bank[n] for n in basin_names])
    set_weights(probe, *soup)
    log_record(
        records, "wsp_soup_basin",
        score_log_preds(y_val, probe.predict(X_val)),
        score_log_preds(y_test, probe.predict(X_test)),
        "Uniform soup of distilled base + forks",
        artifact={"kind": "mlp", "weights": soup, "model": copy.deepcopy(probe)},
        tags={"kind": "wsp_program"},
    )

    wise_val = next(r["val_mae_pkr"] for r in records if r["name"] == best_wise[3])
    members = [("mlp_base", weight_bank["mlp_base"], wise_val)]
    for name in basin_names:
        members.append((
            name,
            weight_bank[name],
            next(r["val_mae_pkr"] for r in records if r["name"] == name),
        ))
    members.sort(key=lambda x: x[2])
    soup_list = [members[0][1]]
    soup_names = [members[0][0]]
    best_val = members[0][2]
    for name, w, _ in members[1:]:
        trial = average_weights(soup_list + [w])
        set_weights(probe, *trial)
        tv = mae_pkr(y_val, probe.predict(X_val))
        if tv < best_val - 1.0:
            soup_list.append(w)
            soup_names.append(name)
            best_val = tv
    greedy_w = average_weights(soup_list)
    set_weights(probe, *greedy_w)
    wsp_val_pred = probe.predict(X_val)
    wsp_test_pred = probe.predict(X_test)
    log_record(
        records, "wsp_soup_greedy",
        score_log_preds(y_val, wsp_val_pred),
        score_log_preds(y_test, wsp_test_pred),
        f"Greedy WSP soup {{{', '.join(soup_names)}}}",
        artifact={"kind": "mlp", "weights": greedy_w, "model": copy.deepcopy(probe),
                  "val": wsp_val_pred, "test": wsp_test_pred},
        tags={"kind": "wsp_program"},
    )

    # Prediction blends: LGBM soups × WSP (α on validation)
    print("Validation-selected prediction blends…", flush=True)
    blend_partners = [
        ("teacher", teacher_val, teacher_test),
        ("lgbm_uniform", uniform_val, uniform_test),
        ("lgbm_greedy", greedy_val, greedy_test),
    ]
    for pname, pval, ptest in blend_partners:
        best_a, best_vmae, best_tmae = 0.0, 1e18, None
        best_pv = best_pt = None
        for alpha in np.linspace(0.0, 1.0, 21):
            pv = (1 - alpha) * pval + alpha * wsp_val_pred
            score = mae_pkr(y_val, pv)
            if score < best_vmae:
                best_vmae = score
                best_a = float(alpha)
                best_pt = (1 - alpha) * ptest + alpha * wsp_test_pred
                best_tmae = mae_pkr(y_test, best_pt)
                best_pv = pv
        log_record(
            records, f"blend_{pname}_x_wsp_a{best_a:g}",
            {"mae_pkr": best_vmae, "rows": len(y_val)},
            {"mae_pkr": best_tmae, "rows": len(y_test)},
            f"(1-α)·{pname} + α·WSP, α={best_a:g} on val",
            artifact={"kind": "pred", "val": best_pv, "test": best_pt},
            tags={"kind": "pred_blend"},
        )

    # Three-way blend grid (coarse) on val
    best_3 = None
    for ai in range(0, 11):
        for bi in range(0, 11 - ai):
            a = ai / 10.0
            b = bi / 10.0
            c = 1.0 - a - b
            pv = a * greedy_val + b * uniform_val + c * wsp_val_pred
            score = mae_pkr(y_val, pv)
            if best_3 is None or score < best_3[0]:
                pt = a * greedy_test + b * uniform_test + c * wsp_test_pred
                best_3 = (score, mae_pkr(y_test, pt), a, b, c, pv, pt)
    log_record(
        records, "blend_3way",
        {"mae_pkr": best_3[0], "rows": len(y_val)},
        {"mae_pkr": best_3[1], "rows": len(y_test)},
        f"{best_3[2]:.2f}·lgbm_greedy + {best_3[3]:.2f}·lgbm_uniform + "
        f"{best_3[4]:.2f}·WSP",
        artifact={"kind": "pred", "val": best_3[5], "test": best_3[6]},
        tags={"kind": "pred_blend"},
    )

    # Pick best on validation
    best = min(records, key=lambda r: r["val_mae_pkr"])
    print(
        f"\nBest on validation: {best['name']}  val {best['val_mae_pkr']:,.0f}  "
        f"test {best['test_mae_pkr']:,.0f}",
        flush=True,
    )
    delta = best["test_mae_pkr"] - lgbm_test_mae
    if delta < 0:
        print(f"BEAT champion by {abs(delta):,.0f} PKR on test.", flush=True)
    else:
        print(f"Still {delta:,.0f} PKR behind champion on test.", flush=True)

    # Persist a serveable experiment bundle (MLP + blend weights + optional LGBM soup)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_weights(probe, *greedy_w)
    # Parse blend weights from the winning note when it is a 3-way / LGBM×WSP blend
    blend_lgbm, blend_wsp = 0.8, 0.2
    if best["name"].startswith("blend_"):
        note = best.get("note", "")
        # e.g. "0.80·lgbm_greedy + 0.00·lgbm_uniform + 0.20·WSP"
        m_l = re.search(r"([0-9.]+)\s*[·x*]\s*lgbm_greedy", note, re.I)
        m_w = re.search(r"([0-9.]+)\s*[·x*]\s*WSP", note, re.I)
        if m_l and m_w:
            blend_lgbm, blend_wsp = float(m_l.group(1)), float(m_w.group(1))

    # Mean of the greedy LGBM soup members → one pipeline for serving
    lgbm_serve = None
    if chosen:
        # Average predictions via first member's structure is awkward; keep the
        # list and average at predict time. For a single artifact, distill to the
        # first greedy member (best val) plus note the full set.
        lgbm_serve = copy.deepcopy(lgbm_pipes[chosen[0]])

    bundle = {
        "program": best["name"],
        "val_mae_pkr": best["val_mae_pkr"],
        "test_mae_pkr": best["test_mae_pkr"],
        "note": best.get("note", ""),
        "preprocessor": pre,
        "mlp": (
            best.get("artifact", {}).get("model")
            if isinstance(best.get("artifact"), dict) else None
        ),
        "champion_test_mae_pkr": lgbm_test_mae,
        "wsp_greedy_mlp": copy.deepcopy(probe),
        "wsp_preprocessor": pre,
        "lgbm_greedy_pipeline": lgbm_serve,
        "lgbm_greedy_member_seeds": [variant_labels[i] for i in chosen],
        "blend_weights": {"lgbm": blend_lgbm, "wsp": blend_wsp},
    }
    # Residual quantiles of the winning program on validation (for API ranges)
    art = best.get("artifact") or {}
    if isinstance(art, dict) and art.get("kind") == "pred" and art.get("val") is not None:
        resid = y_val - np.asarray(art["val"], dtype=float)
    else:
        set_weights(probe, *greedy_w)
        resid = y_val - probe.predict(X_val)
    q10, q90 = np.quantile(resid, [0.1, 0.9])

    joblib.dump(bundle, OUT_DIR / "wsp_best.joblib")
    (OUT_DIR / "wsp_best.json").write_text(json.dumps({
        "program": best["name"],
        "val_mae_pkr": best["val_mae_pkr"],
        "test_mae_pkr": best["test_mae_pkr"],
        "note": best.get("note", ""),
        "champion_test_mae_pkr": lgbm_test_mae,
        "beat_champion": bool(best["test_mae_pkr"] < lgbm_test_mae),
        "blend_weights": {"lgbm": blend_lgbm, "wsp": blend_wsp},
        "q10": float(q10),
        "q90": float(q90),
        "serve": True,
    }, indent=2), encoding="utf-8")

    champions = []
    for name, path in CHAMPION_FILES.items():
        if not path.exists():
            champions.append({"name": name, "test_mae_pkr": None})
            continue
        pipe = joblib.load(path)
        champions.append({
            "name": name,
            "test_mae_pkr": mae_pkr(y_test, pipe.predict(test_df[FEATURES])),
        })

    elapsed = time.perf_counter() - started
    report_rows = [{k: v for k, v in r.items() if k != "artifact"} for r in records]
    write_report(report_rows, champions, best, elapsed, prev_best_test, lgbm_test_mae)
    print(f"Wrote {REPORT_PATH}")
    print(f"Saved {OUT_DIR / 'wsp_best.joblib'}")


if __name__ == "__main__":
    main()
