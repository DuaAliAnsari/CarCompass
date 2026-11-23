"""Weight-space programming experiment on the shared A+B price task.

LightGBM is the served champion, but WSP needs a continuous weight vector, so
this script:

1. Fits a scaled preprocessor (MLP-friendly).
2. Trains a stronger base MLP, then same-basin forks and era specialists.
3. Runs weight-space programs: same-basin soup, WiSE-FT / task arithmetic
   (single and multi-task), TIES, and DARE-lite.
4. Picks the best program on validation MAE and compares test MAE with the
   existing LightGBM family champions.

Run:

    python -m src.train_wsp

Writes reports/wsp_comparison.md and models/wsp_best.joblib, and logs runs
to the MLflow experiment carcompass/wsp.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model_utils import CATEGORICAL, FEATURES, NUMERIC
from src.train import _metrics, git_commit, resolve_tracking_uri

DATA_PATH = Path("data/processed_multi/listings.csv")
REPORT_PATH = Path("reports/wsp_comparison.md")
OUT_DIR = Path("models")
CHAMPION_FILES = {
    "price_global_ab": OUT_DIR / "price_global_ab.joblib",
    "price_new_cars_2010plus": OUT_DIR / "price_new_cars_2010plus.joblib",
    "price_old_cars_pre2010": OUT_DIR / "price_old_cars_pre2010.joblib",
}
HIDDEN = (256, 128, 64)


def build_preprocessor() -> ColumnTransformer:
    # MLP needs scaled numeric inputs; trees in the served pipeline do not.
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


def mae_pkr(y_true: np.ndarray, pred_log: np.ndarray) -> float:
    # Guard against exploding weight-space edits before expm1.
    pred_log = np.asarray(pred_log, dtype=float)
    pred_log = np.nan_to_num(pred_log, nan=0.0, posinf=20.0, neginf=-20.0)
    pred_log = np.clip(pred_log, -20.0, 20.0)
    return float(_metrics(y_true, pred_log)["mae_pkr"])


def clone_weights(mlp: MLPRegressor):
    return (
        [c.copy() for c in mlp.coefs_],
        [b.copy() for b in mlp.intercepts_],
    )


def set_weights(mlp: MLPRegressor, coefs, intercepts) -> MLPRegressor:
    mlp.coefs_ = [c.copy() for c in coefs]
    mlp.intercepts_ = [b.copy() for b in intercepts]
    return mlp


def clone_weights_tuple(weights):
    coefs, bias = weights
    return [c.copy() for c in coefs], [b.copy() for b in bias]


def mix_weights(base, other, alpha: float):
    """W' = W_base + alpha * (W_other - W_base)."""
    b_coefs, b_bias = base
    o_coefs, o_bias = other
    coefs = [bc + alpha * (oc - bc) for bc, oc in zip(b_coefs, o_coefs)]
    bias = [bb + alpha * (ob - bb) for bb, ob in zip(b_bias, o_bias)]
    return coefs, bias


def multi_task_arith(base, tasks: list[tuple], alphas: list[float]):
    """W' = W_base + sum_i alpha_i * (W_i - W_base)."""
    coefs = [c.copy() for c in base[0]]
    bias = [b.copy() for b in base[1]]
    for task, alpha in zip(tasks, alphas):
        for i, (bc, tc) in enumerate(zip(base[0], task[0])):
            coefs[i] = coefs[i] + alpha * (tc - bc)
        for i, (bb, tb) in enumerate(zip(base[1], task[1])):
            bias[i] = bias[i] + alpha * (tb - bb)
    return coefs, bias


def average_weights(weight_list):
    n = len(weight_list)
    coefs = [
        sum(w[0][i] for w in weight_list) / n
        for i in range(len(weight_list[0][0]))
    ]
    bias = [
        sum(w[1][i] for w in weight_list) / n
        for i in range(len(weight_list[0][1]))
    ]
    return coefs, bias


def _delta_parts(base, other):
    return (
        [oc - bc for bc, oc in zip(base[0], other[0])],
        [ob - bb for bb, ob in zip(base[1], other[1])],
    )


def _apply_delta(base, deltas_c, deltas_b, alpha: float):
    coefs = [bc + alpha * dc for bc, dc in zip(base[0], deltas_c)]
    bias = [bb + alpha * db for bb, db in zip(base[1], deltas_b)]
    return coefs, bias


def ties_lite(base, other, keep: float = 0.2, alpha: float = 1.0):
    """Keep the top-|delta| fraction of the task vector, then add alpha * trimmed."""
    deltas_c, deltas_b = _delta_parts(base, other)
    flat = np.concatenate([d.ravel() for d in deltas_c + deltas_b])
    if flat.size == 0:
        return clone_weights_tuple(base)
    k = max(1, int(np.ceil(keep * flat.size)))
    thresh = np.partition(np.abs(flat), -k)[-k]

    def trim(delta):
        out = delta.copy()
        out[np.abs(out) < thresh] = 0.0
        return out

    return _apply_delta(base, [trim(d) for d in deltas_c], [trim(d) for d in deltas_b], alpha)


def ties_merge(base, others: list, keep: float = 0.2, alpha: float = 1.0):
    """TIES over several task vectors: trim, elect sign, average agreeing deltas."""
    if not others:
        return clone_weights_tuple(base)
    trimmed = []
    for other in others:
        deltas_c, deltas_b = _delta_parts(base, other)
        flat = np.concatenate([d.ravel() for d in deltas_c + deltas_b])
        k = max(1, int(np.ceil(keep * flat.size)))
        thresh = np.partition(np.abs(flat), -k)[-k]

        def trim(delta, thr=thresh):
            out = delta.copy()
            out[np.abs(out) < thr] = 0.0
            return out

        trimmed.append(([trim(d) for d in deltas_c], [trim(d) for d in deltas_b]))

    n_layers_c = len(base[0])
    n_layers_b = len(base[1])
    merged_c = []
    for i in range(n_layers_c):
        stack = np.stack([t[0][i] for t in trimmed], axis=0)
        signs = np.sign(stack)
        elect = np.sign(signs.sum(axis=0))
        elect[elect == 0] = 1.0
        agree = signs == elect
        mag = np.where(agree, np.abs(stack), 0.0)
        counts = np.maximum(agree.sum(axis=0), 1)
        avg = mag.sum(axis=0) / counts
        merged_c.append(elect * avg)

    merged_b = []
    for i in range(n_layers_b):
        stack = np.stack([t[1][i] for t in trimmed], axis=0)
        signs = np.sign(stack)
        elect = np.sign(signs.sum(axis=0))
        elect[elect == 0] = 1.0
        agree = signs == elect
        mag = np.where(agree, np.abs(stack), 0.0)
        counts = np.maximum(agree.sum(axis=0), 1)
        avg = mag.sum(axis=0) / counts
        merged_b.append(elect * avg)

    return _apply_delta(base, merged_c, merged_b, alpha)


def dare_lite(base, other, drop: float = 0.5, alpha: float = 1.0, seed: int = 0):
    """Randomly drop delta entries and rescale survivors (DARE-style)."""
    deltas_c, deltas_b = _delta_parts(base, other)
    rng = np.random.default_rng(seed)
    keep = 1.0 - drop
    if keep <= 0:
        return clone_weights_tuple(base)

    def mask(delta):
        m = rng.random(delta.shape) < keep
        out = delta.copy()
        out[~m] = 0.0
        out[m] /= keep
        return out

    return _apply_delta(base, [mask(d) for d in deltas_c], [mask(d) for d in deltas_b], alpha)


def make_mlp(seed: int, max_iter: int = 120) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="adam",
        learning_rate_init=8e-4,
        alpha=3e-5,
        batch_size=512,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        tol=1e-4,
        warm_start=False,
    )


def fit_mlp(mlp: MLPRegressor, X, y, sample_weight=None, resample_seed: int = 42) -> MLPRegressor:
    # MLPRegressor does not take sample_weight in older sklearn; resample instead.
    if sample_weight is None:
        mlp.fit(X, y)
        return mlp
    weight = np.asarray(sample_weight, dtype=float)
    weight = weight / weight.mean()
    rng = np.random.default_rng(resample_seed)
    n = len(y)
    idx = rng.choice(n, size=n, replace=True, p=weight / weight.sum())
    mlp.fit(X[idx], y[idx])
    return mlp


def predict_mlp(mlp: MLPRegressor, X) -> np.ndarray:
    return mlp.predict(X)


def score_split(mlp, X, y_price) -> dict:
    pred = predict_mlp(mlp, X)
    return {"mae_pkr": mae_pkr(y_price, pred), "rows": int(len(y_price))}


def log_record(records, name, val, test, note, weights=None, params=None, tags=None):
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
    if weights is not None:
        row["weights"] = weights
    records.append(row)
    print(
        f"  {name}: val {val['mae_pkr']:,.0f}  test {test['mae_pkr']:,.0f}",
        flush=True,
    )


def champion_scores(df: pd.DataFrame) -> list[dict]:
    """Score the served LightGBM champions on the shared test rows."""
    test = df[df["split"] == "test"]
    rows = []
    for name, path in CHAMPION_FILES.items():
        if not path.exists():
            rows.append({"name": name, "status": "missing", "test_mae_pkr": None})
            continue
        pipe = joblib.load(path)
        pred = pipe.predict(test[FEATURES])
        rows.append({
            "name": name,
            "status": "loaded",
            "test_mae_pkr": mae_pkr(test["price"].to_numpy(), pred),
            "kind": "lightgbm_champion",
        })
    return rows


def write_report(records: list[dict], champions: list[dict], best: dict, elapsed: float,
                 prev_best_test: float | None = None):
    lines = [
        "# Weight-space programming experiment",
        "",
        "LightGBM is the served system. Weight-space programming needs a continuous",
        "weight vector, so this run trains an MLP on the same A+B split, builds",
        "task vectors in the same loss basin, then edits the base weights.",
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

    # Sort programs by val MAE for readability
    ordered = sorted(records, key=lambda r: r["val_mae_pkr"])
    lines += [
        "",
        "## Weight-space programs (MLP)",
        "",
        "| Program | Val MAE (PKR) | Test MAE (PKR) | Notes |",
        "|---|---:|---:|---|",
    ]
    for row in ordered:
        mark = " **best WSP**" if row["name"] == best["name"] else ""
        lines.append(
            f"| `{row['name']}`{mark} | {row['val_mae_pkr']:,.0f} | "
            f"{row['test_mae_pkr']:,.0f} | {row.get('note', '')} |"
        )

    lgbm = next((c for c in champions if c["name"] == "price_global_ab"
                 and c.get("test_mae_pkr") is not None), None)
    lines += ["", "## Does WSP beat the served champion?", ""]
    if lgbm is None:
        lines.append("The global LightGBM champion was not on disk, so no head-to-head.")
    else:
        delta = best["test_mae_pkr"] - lgbm["test_mae_pkr"]
        if delta < 0:
            lines.append(
                f"**Yes on test MAE.** Best WSP `{best['name']}` is "
                f"{abs(delta):,.0f} PKR better than `price_global_ab` "
                f"({best['test_mae_pkr']:,.0f} vs {lgbm['test_mae_pkr']:,.0f})."
            )
        else:
            lines.append(
                f"**Not on test MAE.** Best WSP `{best['name']}` is "
                f"{delta:,.0f} PKR behind `price_global_ab` "
                f"({best['test_mae_pkr']:,.0f} vs {lgbm['test_mae_pkr']:,.0f}). "
                "The LightGBM champion stays the served model; WSP is recorded as "
                "an experiment, not a replacement."
            )

    if prev_best_test is not None:
        gain = prev_best_test - best["test_mae_pkr"]
        lines += [
            "",
            "## vs previous WSP run",
            "",
            f"Previous best WSP test MAE was {prev_best_test:,.0f}. "
            f"This run's best is {best['test_mae_pkr']:,.0f} "
            f"({'better' if gain > 0 else 'worse'} by {abs(gain):,.0f} PKR).",
        ]

    lines += [
        "",
        "## Method (v2)",
        "",
        f"1. Shared preprocessor (median impute + StandardScaler, one-hot). "
        f"Base MLP `{HIDDEN}` with longer early-stopped Adam.",
        "2. Same-basin forks: continue from the base with different bootstrap "
        "resamples (avoids independent-seed soup collapse).",
        "3. Warm-start era specialists (2010+ / pre-2010) from the base.",
        "4. Programs: same-basin soup; WiSE-FT / task arithmetic; multi-task "
        "arithmetic; TIES; DARE-lite; greedy soup; basin×greedy mixes.",
        "5. Choose the program with the lowest **validation** MAE; report test MAE.",
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

    df = pd.read_csv(args.data)
    fit_df = df[df["split"] == "fit"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    pre = build_preprocessor()
    X_fit = pre.fit_transform(fit_df[FEATURES])
    X_val = pre.transform(val_df[FEATURES])
    X_test = pre.transform(test_df[FEATURES])
    if hasattr(X_fit, "toarray"):
        X_fit = X_fit.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    y_fit = np.log1p(fit_df["price"].to_numpy())
    y_val = val_df["price"].to_numpy()
    y_test = test_df["price"].to_numpy()
    weights = np.where(fit_df["source"].to_numpy() == "B", 0.3, 1.0)

    mlflow.set_tracking_uri(resolve_tracking_uri(args.tracking_uri))
    mlflow.set_experiment("carcompass/wsp")

    records: list[dict] = []
    weight_bank = {}

    print("Training base MLP on all fit rows…", flush=True)
    base = make_mlp(seed=42, max_iter=140)
    fit_mlp(base, X_fit, y_fit, weights, resample_seed=42)
    weight_bank["mlp_base"] = clone_weights(base)
    val = score_split(base, X_val, y_val)
    test = score_split(base, X_test, y_test)
    log_record(
        records, "mlp_base", val, test, "Stronger MLP on all fit rows",
        params={"hidden": ",".join(map(str, HIDDEN)), "max_iter": 140},
        tags={"git_commit": git_commit(), "kind": "wsp_base"},
    )

    # Same-basin forks: continue from base with different bootstrap draws.
    print("Training same-basin forks…", flush=True)
    basin_names = []
    for seed in (7, 99, 123, 202, 404):
        name = f"mlp_basin_{seed}"
        fork = copy.deepcopy(base)
        fork.warm_start = True
        fork.max_iter = 30
        fork.random_state = seed
        fit_mlp(fork, X_fit, y_fit, weights, resample_seed=seed)
        weight_bank[name] = clone_weights(fork)
        basin_names.append(name)
        val = score_split(fork, X_val, y_val)
        test = score_split(fork, X_test, y_test)
        log_record(
            records, name, val, test,
            f"Same-basin fork, bootstrap seed {seed}",
            params={"bootstrap_seed": seed, "warm_start_from": "mlp_base"},
            tags={"kind": "wsp_basin"},
        )

    # Era specialists from the base.
    for label, mask in [
        ("mlp_2010plus", fit_df["year"].to_numpy() >= 2010),
        ("mlp_pre2010", fit_df["year"].to_numpy() < 2010),
    ]:
        print(f"Warm-start specialist {label}…", flush=True)
        specialist = copy.deepcopy(base)
        specialist.warm_start = True
        specialist.max_iter = 40
        fit_mlp(specialist, X_fit[mask], y_fit[mask], weights[mask], resample_seed=42)
        weight_bank[label] = clone_weights(specialist)
        val = score_split(specialist, X_val, y_val)
        test = score_split(specialist, X_test, y_test)
        log_record(
            records, label, val, test, "Warm-started era specialist",
            params={"warm_start_from": "mlp_base"},
            tags={"kind": "wsp_specialist"},
        )

    programs = []
    # Same-basin soup (should not collapse like independent seeds).
    soup = average_weights([weight_bank["mlp_base"]] + [weight_bank[n] for n in basin_names])
    programs.append(("wsp_soup_basin", soup, "Average of base + same-basin forks"))

    # WiSE-FT / task arithmetic on 2010+ specialist
    for alpha in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        programs.append((
            f"wsp_ta_2010_a{alpha:g}",
            mix_weights(weight_bank["mlp_base"], weight_bank["mlp_2010plus"], alpha),
            f"W_base + {alpha:g}·(W_2010 − W_base)",
        ))

    # Task arithmetic on pre-2010
    for alpha in (0.1, 0.25, 0.5):
        programs.append((
            f"wsp_ta_pre_a{alpha:g}",
            mix_weights(weight_bank["mlp_base"], weight_bank["mlp_pre2010"], alpha),
            f"W_base + {alpha:g}·(W_pre − W_base)",
        ))

    # Multi-task arithmetic grid (both era vectors)
    for a2010, apre in ((0.15, 0.05), (0.2, 0.1), (0.25, 0.1), (0.3, 0.05), (0.1, 0.1)):
        programs.append((
            f"wsp_multi_a{a2010:g}_b{apre:g}",
            multi_task_arith(
                weight_bank["mlp_base"],
                [weight_bank["mlp_2010plus"], weight_bank["mlp_pre2010"]],
                [a2010, apre],
            ),
            f"W + {a2010:g}·Δ2010 + {apre:g}·Δpre",
        ))

    # TIES single-task
    for keep, alpha in ((0.1, 0.5), (0.2, 0.5), (0.2, 1.0), (0.3, 0.75)):
        programs.append((
            f"wsp_ties_k{keep:g}_a{alpha:g}",
            ties_lite(weight_bank["mlp_base"], weight_bank["mlp_2010plus"], keep=keep, alpha=alpha),
            f"TIES-lite keep={keep:g} α={alpha:g} on 2010",
        ))

    # TIES merge of both era vectors + one basin fork
    for keep, alpha in ((0.2, 0.5), (0.2, 1.0)):
        programs.append((
            f"wsp_ties_merge_k{keep:g}_a{alpha:g}",
            ties_merge(
                weight_bank["mlp_base"],
                [
                    weight_bank["mlp_2010plus"],
                    weight_bank["mlp_pre2010"],
                    weight_bank["mlp_basin_7"],
                ],
                keep=keep,
                alpha=alpha,
            ),
            f"TIES-merge eras+basin keep={keep:g} α={alpha:g}",
        ))

    # DARE-lite
    for drop, alpha in ((0.3, 0.5), (0.5, 0.75), (0.5, 1.0)):
        programs.append((
            f"wsp_dare_d{drop:g}_a{alpha:g}",
            dare_lite(
                weight_bank["mlp_base"],
                weight_bank["mlp_2010plus"],
                drop=drop,
                alpha=alpha,
                seed=0,
            ),
            f"DARE-lite drop={drop:g} α={alpha:g} on 2010",
        ))

    # Soup of WiSE-FT checkpoints along the 2010 path.
    wise_path = [
        mix_weights(weight_bank["mlp_base"], weight_bank["mlp_2010plus"], a)
        for a in (0.1, 0.2, 0.3, 0.5)
    ]
    programs.append((
        "wsp_soup_wise_path",
        average_weights(wise_path),
        "Soup of WiSE-FT checkpoints α∈{0.1,0.2,0.3,0.5}",
    ))

    print(f"Evaluating {len(programs)} weight-space programs…", flush=True)
    probe = copy.deepcopy(base)
    for name, program_weights, note in programs:
        set_weights(probe, *program_weights)
        val = score_split(probe, X_val, y_val)
        test = score_split(probe, X_test, y_test)
        log_record(
            records, name, val, test, note,
            weights=program_weights,
            tags={"kind": "wsp_program"},
        )

    # Greedy soup (Wortsman et al.): start from best val member, add if val improves.
    print("Building greedy soup on validation…", flush=True)
    members = []
    for name in ["mlp_base", *basin_names, "mlp_2010plus", "mlp_pre2010"]:
        rec = next(r for r in records if r["name"] == name)
        members.append((name, weight_bank[name], rec["val_mae_pkr"]))
    members.sort(key=lambda x: x[2])
    soup_list = [members[0][1]]
    soup_names = [members[0][0]]
    best_val = members[0][2]
    for name, w, _ in members[1:]:
        trial = average_weights(soup_list + [w])
        set_weights(probe, *trial)
        trial_val = score_split(probe, X_val, y_val)["mae_pkr"]
        if trial_val < best_val - 1.0:  # require a real improvement
            soup_list.append(w)
            soup_names.append(name)
            best_val = trial_val
    greedy = average_weights(soup_list)
    set_weights(probe, *greedy)
    val = score_split(probe, X_val, y_val)
    test = score_split(probe, X_test, y_test)
    log_record(
        records, "wsp_soup_greedy", val, test,
        f"Greedy soup of {{{', '.join(soup_names)}}}",
        weights=greedy,
        tags={"kind": "wsp_program"},
    )

    # Interpolate greedy soup with the uniform basin soup (often helps a little).
    basin_soup = average_weights(
        [weight_bank["mlp_base"]] + [weight_bank[n] for n in basin_names]
    )
    for alpha in (0.25, 0.5, 0.75):
        mixed = mix_weights(basin_soup, greedy, alpha)
        set_weights(probe, *mixed)
        val = score_split(probe, X_val, y_val)
        test = score_split(probe, X_test, y_test)
        log_record(
            records, f"wsp_soup_mix_a{alpha:g}", val, test,
            f"basin_soup + {alpha:g}·(greedy − basin_soup)",
            weights=mixed,
            tags={"kind": "wsp_program"},
        )

    best = min(records, key=lambda r: r["val_mae_pkr"])
    print(
        f"\nBest WSP on validation: {best['name']}  val MAE {best['val_mae_pkr']:,.0f}"
        f"  test MAE {best['test_mae_pkr']:,.0f}",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_mlp = copy.deepcopy(base)
    if "weights" in best:
        set_weights(best_mlp, *best["weights"])
    elif best["name"] in weight_bank:
        set_weights(best_mlp, *weight_bank[best["name"]])
    bundle = {
        "preprocessor": pre,
        "model": best_mlp,
        "program": best["name"],
        "val_mae_pkr": best["val_mae_pkr"],
        "test_mae_pkr": best["test_mae_pkr"],
    }
    joblib.dump(bundle, OUT_DIR / "wsp_best.joblib")
    (OUT_DIR / "wsp_best.json").write_text(json.dumps({
        "program": best["name"],
        "val_mae_pkr": best["val_mae_pkr"],
        "test_mae_pkr": best["test_mae_pkr"],
        "note": best.get("note", ""),
    }, indent=2), encoding="utf-8")

    champions = champion_scores(df)
    elapsed = time.perf_counter() - started
    report_rows = [{k: v for k, v in r.items() if k != "weights"} for r in records]
    write_report(report_rows, champions, best, elapsed, prev_best_test=prev_best_test)
    print(f"Wrote {REPORT_PATH}")
    print(f"Saved {OUT_DIR / 'wsp_best.joblib'}")


if __name__ == "__main__":
    main()
