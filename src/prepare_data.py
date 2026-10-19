"""Data preparation: schema mapping, validation, de-duplication and split.

Run:  python -m src.prepare_data --input data/sample/sample_listings.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from src.model_utils import CATEGORICAL, OPTIONAL, REQUIRED_INPUT, normalize_text

REQUIRED_COLUMNS = REQUIRED_INPUT + ["price"]
STANDARD_COLUMNS = REQUIRED_COLUMNS + OPTIONAL
RANGES = {
    "price": (50_000, 200_000_000),
    "mileage": (0, 1_000_000),
    "year": (1980, date.today().year + 1),
    "engine_cc": (50, 8000),
}
# Rows equal on all of DEDUP_COLS are duplicates and are dropped.
DEDUP_COLS = ["make", "model", "year", "mileage", "price", "city"]
# Near-copies of one listing share these columns, so they stay on the same side of the split.
GROUP_COLS = ["make", "model", "year", "mileage", "city"]


def parse_price_pkr(value) -> float:
    """Turn '27.4 lacs', '1.2 crore', '2,740,000' or 2740000 into a PKR number."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"([\d.]+)\s*(crore|cr|lacs?|lakhs?)?", text)
    if not match:
        return np.nan
    try:
        number = float(match.group(1))
    except ValueError:
        return np.nan
    unit = match.group(2)
    if unit in ("crore", "cr"):
        number *= 10_000_000
    elif unit and unit.startswith(("lac", "lakh")):
        number *= 100_000
    return number


def parse_number(value) -> float:
    """Pull the first number out of text like '75,000 km' or '1300 cc'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"[\d.]+", str(value).replace(",", ""))
    try:
        return float(match.group(0)) if match else np.nan
    except ValueError:
        return np.nan


def map_schema(df: pd.DataFrame, column_map: dict | None = None) -> pd.DataFrame:
    """Rename raw columns to the standard schema and parse strings into numbers."""
    df = df.rename(columns=column_map or {}).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in STANDARD_COLUMNS:
        if col not in df:
            df[col] = np.nan
    df["price"] = df["price"].map(parse_price_pkr)
    for col in ("year", "mileage", "engine_cc"):
        df[col] = df[col].map(parse_number)
    for col in CATEGORICAL:
        df[col] = df[col].map(normalize_text)
    return df[STANDARD_COLUMNS]


def validate(df: pd.DataFrame):
    """Drop rows with missing required values or impossible price/mileage/year."""
    report = {}
    n = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    report["dropped_missing"] = n - len(df)

    n = len(df)
    keep = pd.Series(True, index=df.index)
    for col in ("price", "mileage", "year"):
        low, high = RANGES[col]
        keep &= df[col].between(low, high)
    df = df[keep].copy()
    report["dropped_out_of_range"] = n - len(df)

    low, high = RANGES["engine_cc"]
    bad_engine = df["engine_cc"].notna() & ~df["engine_cc"].between(low, high)
    df.loc[bad_engine, "engine_cc"] = np.nan
    report["engine_set_to_missing"] = int(bad_engine.sum())

    df["year"] = df["year"].astype(int)
    df["mileage"] = df["mileage"].astype(int)
    return df.reset_index(drop=True), report


def drop_duplicates(df: pd.DataFrame):
    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_COLS).reset_index(drop=True)
    return df, before - len(df)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """80/20 split where near-copies of one listing always land on the same side."""
    groups = df[GROUP_COLS].astype(str).agg("|".join, axis=1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def prepare(raw: pd.DataFrame, column_map: dict | None = None, test_size: float = 0.2,
            seed: int = 42):
    """Full preparation. Returns (train, test, quality_report)."""
    df = map_schema(raw, column_map)
    rows_in = len(df)
    df, report = validate(df)
    df, dropped_dupes = drop_duplicates(df)
    train, test = split(df, test_size, seed)
    report.update(
        rows_in=rows_in,
        dropped_duplicates=dropped_dupes,
        rows_out=len(df),
        train_rows=len(train),
        test_rows=len(test),
        seed=seed,
    )
    return train, test, report


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def load_column_map(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text()) or {}).get("column_map", {})


def main():
    parser = argparse.ArgumentParser(description="Prepare listings for training.")
    parser.add_argument("--input", default="data/sample/sample_listings.csv")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--schema", default="configs/schema.yaml")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    train, test, report = prepare(raw, load_column_map(args.schema))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    report["train_hash"] = file_hash(out / "train.csv")
    report["test_hash"] = file_hash(out / "test.csv")
    (out / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
