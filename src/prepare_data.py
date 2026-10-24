"""Prepare the PakWheels listings: parse, clean, de-duplicate, split.

Run:  python -m src.prepare_data --config configs/data.yaml
"""
import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

UNITS = {"crore": 1e7, "lac": 1e5, "lacs": 1e5, "lakh": 1e5, "lakhs": 1e5, "thousand": 1e3}


# ---------- parsers ----------
def parse_pkr(x):
    """'PKR 23.9 lacs' -> 2390000.0, 'PKR 3.5 crore' -> 35000000.0, junk -> NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).lower().replace(",", "").replace("pkr", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(crore|lacs|lac|lakhs|lakh|thousand)?", s)
    if not m:
        return np.nan
    return float(m.group(1)) * UNITS.get(m.group(2), 1)


def parse_km(x):
    """'120,000 km' -> 120000.0"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(x))
    return float(m.group().replace(",", "")) if m else np.nan


def parse_cc(x):
    """'660 cc' / '800cc' -> 660.0 / 800.0. Electric ('71.0 kWh') -> NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if "cc" not in s:
        return np.nan
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    return float(m.group().replace(",", "")) if m else np.nan


def split_name(name, multiword_makes, multiword_models):
    """'Suzuki Alto VXL AGS 2022' -> (make='Suzuki', model='Alto', variant='VXL AGS')."""
    s = str(name).strip()
    s = re.sub(r"\s+(19|20)\d{2}$", "", s)          # drop the trailing year
    parts = s.split()
    if not parts:
        return pd.Series([np.nan, np.nan, np.nan])
    make, rest = parts[0], parts[1:]
    for mw in multiword_makes:
        n = len(mw.split())
        if " ".join(parts[:n]).lower() == mw.lower():
            make, rest = mw, parts[n:]
            break
    if not rest:
        return pd.Series([make, np.nan, np.nan])
    model, variant = rest[0], rest[1:]
    # longest matching multi-word model wins
    for mm in sorted(multiword_models, key=lambda t: -len(t.split())):
        n = len(mm.split())
        if len(rest) >= n and " ".join(rest[:n]).lower() == mm.lower():
            model, variant = mm, rest[n:]
            break
    return pd.Series([make, model, " ".join(variant) if variant else np.nan])


def canonical_case(series):
    """Map 'KIA'/'Kia'/'kia' to the most frequent spelling."""
    s = series.dropna().astype(str).str.strip()
    best = s.groupby(s.str.lower()).agg(lambda v: v.value_counts().idxmax())
    return series.astype("string").str.strip().str.lower().map(best)


def lump_rare(series, min_count):
    counts = series.value_counts()
    keep = set(counts[counts >= min_count].index)
    return series.where(series.isna() | series.isin(keep), "Other")


# ---------- main pipeline ----------
class Log(dict):
    """Records how many rows each step removed, for the report."""

    def step(self, label, before, after):
        self[label] = (before, after)


def run(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    log = Log()
    df = pd.read_csv(cfg["raw_path"])
    log["raw rows"] = len(df)

    # 1. schema mapping (drops Features, Owner nam, url on purpose)
    df = df.rename(columns=cfg["column_map"])[list(cfg["column_map"].values())].copy()
    df["price"] = df["price_raw"].map(parse_pkr)
    df["mileage"] = df["mileage_raw"].map(parse_km)
    df["engine_cc"] = df["engine_raw"].map(parse_cc)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[["make", "model", "variant"]] = df.apply(
        lambda r: split_name(r["name"], cfg["multiword_makes"], cfg["multiword_models"]),
        axis=1,
    )
    df["make"] = canonical_case(df["make"])
    df["model"] = canonical_case(df["model"])
    for c in ["fuel", "transmission", "assembly", "body_type", "location", "color"]:
        df[c] = df[c].astype("string").str.strip()
    df = df.drop(columns=["price_raw", "mileage_raw", "engine_raw", "name"])

    # 2. required fields
    n = len(df)
    df = df.dropna(subset=["make", "model", "year", "mileage", "price"])
    log.step("dropped: missing required field", n, len(df))

    # 3. range checks and outliers (one line per rule)
    b = cfg["bounds"]
    rules = {
        "price out of range": df.price.between(*b["price"]),
        "year out of range": df.year.between(*b["year"]),
        "mileage out of range": df.mileage.between(*b["mileage"]),
        "engine_cc out of range": df.engine_cc.isna() | df.engine_cc.between(*b["engine_cc"]),
    }
    lm = cfg["low_mileage_rule"]
    rules["old car with implausibly low mileage"] = ~(
        ((lm["reference_year"] - df.year) >= lm["min_age_years"]) & (df.mileage < lm["min_km"])
    )
    for label, ok in rules.items():
        n = len(df)
        df = df[ok.reindex(df.index, fill_value=True)]
        log.step(f"dropped: {label}", n, len(df))

    # 4. duplicates
    n = len(df)
    df = df.drop_duplicates(subset=["ad_id"])
    log.step("dropped: duplicate Ad Reference", n, len(df))
    n = len(df)
    df = df.drop_duplicates(subset=["make", "model", "year", "mileage", "price", "location"])
    log.step("dropped: exact duplicate listing", n, len(df))

    # 5. rare (make, model) pairs, then group rare locations and colours
    n = len(df)
    pair_counts = df.groupby(["make", "model"])["price"].transform("size")
    df = df[pair_counts >= cfg["min_listings_per_model"]]
    log.step("dropped: rare make/model", n, len(df))
    df["location"] = lump_rare(df["location"], cfg["min_count_location"])
    df["color"] = lump_rare(df["color"], cfg["min_count_color"])

    # 6. listing_group: near-duplicates (re-listed cars) must stay on the same side of the split
    gkey = (
        df[["make", "model", "year", "location"]].astype(str).agg("|".join, axis=1)
        + "|" + (df.mileage // 1000).astype(int).astype(str)
    )
    df["listing_group"] = gkey.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:12])

    # 7. validate
    df = df.reset_index(drop=True)
    validate(df, cfg)

    # 8. grouped 80/20 split, seed 42
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["test_size"], random_state=cfg["seed"])
    tr, te = next(gss.split(df, groups=df["listing_group"]))
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    assert set(train.listing_group).isdisjoint(test.listing_group), "group leaked across split"
    log["final rows"], log["train rows"], log["test rows"] = len(df), len(train), len(test)

    # 9. save, hash, report, MLflow
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    h = hashlib.sha256((out / "train.parquet").read_bytes()).hexdigest()[:16]
    write_report(cfg, log, df, h)
    log_mlflow(cfg, log, h)
    print(f"Done. train={len(train):,} test={len(test):,} hash={h}")
    return train, test


def validate(df, cfg):
    try:
        import pandera.pandas as pa
    except ImportError:  # older pandera versions
        import pandera as pa
    b = cfg["bounds"]
    schema = pa.DataFrameSchema(
        {
            "make": pa.Column(nullable=False, checks=pa.Check(lambda s: s.astype(str).str.len() > 0)),
            "model": pa.Column(nullable=False, checks=pa.Check(lambda s: s.astype(str).str.len() > 0)),
            "year": pa.Column(float, pa.Check.in_range(*b["year"]), coerce=True),
            "mileage": pa.Column(float, pa.Check.in_range(*b["mileage"]), coerce=True),
            "price": pa.Column(float, pa.Check.in_range(*b["price"]), coerce=True),
            "engine_cc": pa.Column(float, nullable=True, coerce=True),
            "listing_group": pa.Column(nullable=False),
        },
        strict=False,
    )
    schema.validate(df)


def write_report(cfg, log, df, h):
    lines = ["# Data-quality report", "", f"Source: `{cfg['raw_path']}`", "",
             "## Rows after each step", "", "| Step | Before | After | Removed |", "|---|---|---|---|"]
    lines.append(f"| raw rows | | {log['raw rows']:,} | |")
    for k, v in log.items():
        if isinstance(v, tuple):
            lines.append(f"| {k} | {v[0]:,} | {v[1]:,} | {v[0] - v[1]:,} |")
    lines += ["", f"Final rows: {log['final rows']:,} (train {log['train rows']:,}, test {log['test rows']:,})", "",
              f"Train file hash (sha256, first 16 chars): `{h}`", "",
              "## Era counts", "",
              f"Pre-2010: {(df.year < 2010).sum():,} | 2010 and later: {(df.year >= 2010).sum():,}", "",
              "## Missing values (fraction)", "",
              df[["engine_cc", "body_type", "color", "location", "fuel", "transmission", "assembly"]]
              .isna().mean().round(3).to_frame("missing").to_markdown(), "",
              "## Price (PKR)", "", df.price.describe().round(0).to_frame("price").to_markdown(), "",
              "## Top 15 make/model", "",
              df.groupby(["make", "model"]).size().sort_values(ascending=False).head(15)
              .to_frame("listings").to_markdown()]
    p = Path(cfg["report_path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")


def log_mlflow(cfg, log, h):
    import mlflow
    mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment("carcompass-data")
    with mlflow.start_run(run_name="prepare_data"):
        mlflow.log_params({"seed": cfg["seed"], "test_size": cfg["test_size"], "raw_path": cfg["raw_path"]})
        mlflow.log_metrics({
            "raw_rows": log["raw rows"], "final_rows": log["final rows"],
            "train_rows": log["train rows"], "test_rows": log["test rows"],
        })
        mlflow.set_tag("train_hash", h)
        mlflow.log_artifact(cfg["report_path"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    run(ap.parse_args().config)
