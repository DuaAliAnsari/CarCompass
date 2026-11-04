"""Data preparation: schema mapping, validation, de-duplication and split.

Run:  python -m src.prepare_data --input data/sample/sample_listings.csv

For the real PakWheels file (combined name column, optional extra cleaning rules,
MLflow logging):

    python -m src.prepare_data --input data/raw/pakwheels.csv \
        --min-model-count 5 --min-city-count 20 --check-low-mileage --log-mlflow

Two price datasets at once (each file becomes one `source`):

    python -m src.prepare_data \
        --input data/raw/pakistan_used_cars.csv data/raw/pakwheels_2022.csv \
        --out data/processed_merged --report reports/data_quality_merged.md \
        --rescale-prices
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
# Near-copies of one listing share these columns, so they stay on the same side of
# the split.
GROUP_COLS = ["make", "model", "year", "mileage", "city"]

# PakWheels gives one combined name such as "Toyota Corolla XLI 1.3". We split it
# into make and model. Add to these sets if a make or model has two words.
TWO_WORD_MAKES = {"land rover", "mercedes benz", "rolls royce", "alfa romeo",
                  "aston martin"}
TWO_WORD_MODELS = {"land cruiser", "grand cherokee", "grand vitara", "wagon r",
                   "n wgn", "n box", "n one"}

# Optional low-mileage rule: a car this old with fewer km than this is treated as a
# data error (odometer left at 0 or typed wrongly) and dropped.
LOW_MILEAGE_MIN_AGE = 3
LOW_MILEAGE_KM = 1_000


# Cross-source price check: only make/model/year groups with at least this many
# listings in BOTH sources are compared.
AGREEMENT_MIN_ROWS = 3
AGREEMENT_TOLERANCE = 0.15   # medians within +-15% count as agreeing


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


def split_name(value) -> tuple:
    """Split 'Toyota Corolla XLI 1.3' into ('Toyota', 'Corolla').

    The variant ('XLI 1.3') is dropped because the model does not use it.
    Returns (nan, nan) when the name is missing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, np.nan
    tokens = str(value).strip().split()
    if not tokens:
        return np.nan, np.nan
    lowered = [t.lower() for t in tokens]

    # PakWheels lists these without the "Land Rover" make in front.
    if " ".join(lowered[:2]) == "range rover":
        return "Land Rover", "Range Rover"

    n_make = 2 if " ".join(lowered[:2]) in TWO_WORD_MAKES else 1
    make = " ".join(tokens[:n_make])

    rest = tokens[n_make:]
    if not rest:
        return make, np.nan
    if " ".join(lowered[n_make:n_make + 2]) in TWO_WORD_MODELS:
        return make, " ".join(rest[:2])
    return make, rest[0]


def _fill_from_name(df: pd.DataFrame) -> pd.DataFrame:
    """If a 'name' column exists, use it to fill make and model where missing."""
    if "name" not in df.columns:
        return df
    parts = df["name"].map(split_name)
    for i, col in enumerate(("make", "model")):
        from_name = parts.map(lambda pair, i=i: pair[i])
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), from_name)
        else:
            df[col] = from_name
    return df


def map_schema(df: pd.DataFrame, column_map: dict | None = None,
               source: str = "default") -> pd.DataFrame:
    """Rename raw columns to the standard schema and parse strings into numbers.

    `source` labels where the rows came from (usually the file name), so two
    datasets can be mixed and still told apart.
    """
    df = df.rename(columns=column_map or {}).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = _fill_from_name(df)
    for col in STANDARD_COLUMNS:
        if col not in df:
            df[col] = np.nan
    df["price"] = df["price"].map(parse_price_pkr)
    for col in ("year", "mileage", "engine_cc"):
        df[col] = df[col].map(parse_number)
    for col in CATEGORICAL:
        df[col] = df[col].map(normalize_text)
    df["source"] = source
    return df[STANDARD_COLUMNS + ["source"]]


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


def apply_quality_rules(df: pd.DataFrame, min_model_count: int = 0,
                        min_city_count: int = 0, check_low_mileage: bool = False):
    """Optional extra cleaning for the real dataset. All rules are off by default.

    - check_low_mileage: drop old cars with an implausibly low odometer reading.
    - min_model_count: drop make/model pairs with fewer listings than this, since
      the model cannot learn a price for a car it has barely seen.
    - min_city_count: group cities with fewer listings than this into 'Other'.
    """
    df = df.copy()
    report = {}

    if check_low_mileage:
        age = date.today().year - df["year"]
        bad = (age >= LOW_MILEAGE_MIN_AGE) & (df["mileage"] < LOW_MILEAGE_KM)
        report["dropped_low_mileage"] = int(bad.sum())
        df = df[~bad]

    if min_model_count > 0:
        counts = df.groupby(["make", "model"])["price"].transform("size")
        rare = counts < min_model_count
        report["dropped_rare_model"] = int(rare.sum())
        df = df[~rare]

    if min_city_count > 0:
        city_counts = df["city"].value_counts()
        rare_cities = city_counts[city_counts < min_city_count].index
        mask = df["city"].isin(rare_cities)
        df.loc[mask, "city"] = "Other"
        report["listings_with_city_grouped"] = int(mask.sum())

    return df.reset_index(drop=True), report


def add_listing_group(df: pd.DataFrame) -> pd.DataFrame:
    """Give every row a listing_group id: rows describing the same car share one id.

    The split keeps each group on one side, so near-copies of a listing cannot
    appear in both train and test.
    """
    df = df.copy()
    keys = df[GROUP_COLS].astype(str).agg("|".join, axis=1)
    df["listing_group"] = keys.map(
        lambda k: hashlib.md5(k.encode("utf-8")).hexdigest()[:10]
    )
    return df


def price_agreement(df: pd.DataFrame, min_rows: int = AGREEMENT_MIN_ROWS,
                    tolerance: float = AGREEMENT_TOLERANCE) -> dict:
    """Do two sources price the same make/model/year alike?

    For every make/model/year with at least `min_rows` listings in both the first
    source and another source, compare the two median prices. Prices drift over
    time, so a ratio far from 1 means the scrape dates (or markets) differ and the
    sources should not be mixed blindly.
    """
    sources = list(dict.fromkeys(df["source"]))
    result = {}
    if len(sources) < 2:
        return result
    base = sources[0]
    key = ["make", "model", "year"]
    med = df.groupby(key + ["source"])["price"].agg(["median", "size"]).reset_index()
    base_rows = med[med["source"] == base].set_index(key)
    for other in sources[1:]:
        other_rows = med[med["source"] == other].set_index(key)
        both = base_rows.join(other_rows, lsuffix="_base", rsuffix="_other", how="inner")
        both = both[(both["size_base"] >= min_rows) & (both["size_other"] >= min_rows)]
        if both.empty:
            result[other] = {"base": base, "groups_compared": 0, "sources_agree": False}
            continue
        ratio = both["median_other"] / both["median_base"]
        within = float(((ratio - 1).abs() <= tolerance).mean())
        median_ratio = float(ratio.median())
        result[other] = {
            "base": base,
            "groups_compared": int(len(both)),
            "median_price_ratio": round(median_ratio, 3),
            "share_within_tolerance": round(within, 3),
            "sources_agree": bool(abs(median_ratio - 1) <= tolerance and within >= 0.5),
        }
    return result


def rescale_sources(train: pd.DataFrame, test: pd.DataFrame, min_groups: int = 20) -> dict:
    """Bring older sources to the first source's price level (modifies train and test).

    The factor is the inverse of the median price ratio over make/model/year groups
    present in both sources, measured on the TRAIN rows only so the test set does not
    influence it. It corrects an overall level shift (inflation between scrape dates)
    but not differences that vary by car. Returns {source: factor}.
    """
    factors = {}
    for other, info in price_agreement(train).items():
        ok = info.get("groups_compared", 0) >= min_groups and "median_price_ratio" in info
        factor = 1.0 / info["median_price_ratio"] if ok else 1.0
        factors[other] = round(factor, 4)
        for frame in (train, test):
            frame.loc[frame["source"] == other, "price"] *= factor
    return factors


def check_schema(df: pd.DataFrame) -> str:
    """Validate the cleaned data with Pandera. Returns 'passed' or 'skipped'.

    Raises a Pandera error if the data breaks the schema. If Pandera is not
    installed the check is skipped (run `pip install pandera` to enable it).
    """
    try:
        import pandera.pandas as pa
    except ImportError:
        try:
            import pandera as pa
        except ImportError:
            return "skipped (pandera not installed)"

    low_year, high_year = RANGES["year"]
    schema = pa.DataFrameSchema(
        {
            "make": pa.Column(str, nullable=False),
            "model": pa.Column(str, nullable=False),
            "year": pa.Column(int, pa.Check.in_range(low_year, high_year)),
            "mileage": pa.Column(int, pa.Check.in_range(*RANGES["mileage"])),
            "price": pa.Column(float, pa.Check.in_range(*RANGES["price"])),
            "engine_cc": pa.Column(
                float, pa.Check.in_range(*RANGES["engine_cc"]), nullable=True
            ),
            "transmission": pa.Column(str, nullable=True),
            "fuel": pa.Column(str, nullable=True),
            "city": pa.Column(str, nullable=True),
            "listing_group": pa.Column(str, nullable=False),
            "source": pa.Column(str, nullable=False),
        },
        strict=True,
    )
    schema.validate(df)
    return "passed"


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """80/20 split where near-copies of one listing always land on the same side."""
    groups = df["listing_group"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


def prepare(raw, column_map: dict | None = None, test_size: float = 0.2,
            seed: int = 42, min_model_count: int = 0, min_city_count: int = 0,
            check_low_mileage: bool = False, rescale_prices: bool = False):
    """Full preparation. Returns (train, test, quality_report).

    `raw` is one DataFrame, or a dict {source_name: DataFrame} to mix several
    datasets. Sources are cleaned together, so a listing that appears in both
    (same make, model, year, mileage, price, city) is kept once.
    rescale_prices=True then moves every later source to the first source's price
    level (see rescale_sources); use it when the scrape dates differ.
    """
    if isinstance(raw, dict):
        df = pd.concat([map_schema(frame, column_map, name)
                        for name, frame in raw.items()], ignore_index=True)
    else:
        df = map_schema(raw, column_map)
    rows_in = len(df)
    rows_in_by_source = df["source"].value_counts().to_dict()
    df, report = validate(df)
    df, dropped_dupes = drop_duplicates(df)
    df, rule_report = apply_quality_rules(
        df, min_model_count, min_city_count, check_low_mileage
    )
    df = add_listing_group(df)
    report["schema_check"] = check_schema(df)
    train, test = split(df, test_size, seed)
    report.update(rule_report)
    report.update(
        rows_in=rows_in,
        dropped_duplicates=dropped_dupes,
        rows_out=len(df),
        rows_pre_2010=int((df["year"] < 2010).sum()),
        rows_2010_and_later=int((df["year"] >= 2010).sum()),
        train_rows=len(train),
        test_rows=len(test),
        seed=seed,
    )
    if df["source"].nunique() > 1:
        report["rows_in_by_source"] = {k: int(v) for k, v in rows_in_by_source.items()}
        report["rows_out_by_source"] = {
            k: int(v) for k, v in df["source"].value_counts().items()}
        report["test_rows_by_source"] = {
            k: int(v) for k, v in test["source"].value_counts().items()}
        report["price_agreement"] = price_agreement(df)
        if rescale_prices:
            report["price_rescale_factors"] = rescale_sources(train, test)
            report["price_agreement_after_rescale"] = price_agreement(train)
    return train, test, report


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def load_column_map(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text()) or {}).get("column_map", {})


def write_markdown_report(report: dict, path) -> None:
    """Human-readable version of the quality report."""
    lines = [
        "# Data quality report",
        "",
        "| Step | Rows |",
        "|---|---|",
        f"| Rows read | {report['rows_in']} |",
        f"| Dropped: missing required values | {report['dropped_missing']} |",
        f"| Dropped: out of range | {report['dropped_out_of_range']} |",
        f"| Dropped: duplicates | {report['dropped_duplicates']} |",
    ]
    if "dropped_low_mileage" in report:
        lines.append(f"| Dropped: low mileage | {report['dropped_low_mileage']} |")
    if "dropped_rare_model" in report:
        lines.append(f"| Dropped: rare make/model | {report['dropped_rare_model']} |")
    if "listings_with_city_grouped" in report:
        lines.append(
            f"| City grouped as 'Other' | {report['listings_with_city_grouped']} |"
        )
    lines += [
        f"| Engine set to missing | {report['engine_set_to_missing']} |",
        f"| **Rows kept** | **{report['rows_out']}** |",
        f"| Kept: pre-2010 cars | {report['rows_pre_2010']} |",
        f"| Kept: 2010 and later | {report['rows_2010_and_later']} |",
        f"| Train rows | {report['train_rows']} |",
        f"| Test rows | {report['test_rows']} |",
        "",
        f"Seed: {report['seed']}",
        f"Pandera schema check: {report['schema_check']}",
    ]
    if "rows_out_by_source" in report:
        lines += ["", "## Sources", "", "| Source | Rows read | Rows kept | Test rows |",
                  "|---|---|---|---|"]
        for name, kept in report["rows_out_by_source"].items():
            lines.append(
                f"| {name} | {report['rows_in_by_source'].get(name, 0)} | {kept} | "
                f"{report['test_rows_by_source'].get(name, 0)} |")
    for other, info in report.get("price_agreement", {}).items():
        lines += ["", f"## Price agreement: {other} vs {info['base']}", ""]
        if info["groups_compared"] == 0:
            lines.append("No make/model/year group had enough listings in both sources.")
        else:
            lines += [
                f"- make/model/year groups compared: {info['groups_compared']}",
                f"- median price ratio ({other} / {info['base']}): "
                f"{info['median_price_ratio']}",
                f"- groups within 15%: {info['share_within_tolerance']:.0%}",
                f"- sources agree: **{'yes' if info['sources_agree'] else 'no'}**",
            ]
    if "price_rescale_factors" in report:
        lines += ["", "## Price rescaling", "",
                  "Prices of later sources were multiplied by these factors (measured on "
                  "train rows only) to match the first source's price level:", ""]
        for name, factor in report["price_rescale_factors"].items():
            lines.append(f"- {name}: x{factor}")
        for other, info in report.get("price_agreement_after_rescale", {}).items():
            if info.get("groups_compared"):
                lines.append(
                    f"- after rescaling, {other}: median ratio {info['median_price_ratio']}, "
                    f"{info['share_within_tolerance']:.0%} of groups within 15%")
    if "train_hash" in report:
        lines.append(f"Train file hash: `{report['train_hash']}`")
        lines.append(f"Test file hash: `{report['test_hash']}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_to_mlflow(report: dict, out: Path, tracking_uri: str | None = None,
                  experiment: str = "carcompass-data") -> None:
    """Record row counts and file hashes of this data version in MLflow."""
    import mlflow  # imported here so plain data prep does not need MLflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    keys = ("seed", "train_hash", "test_hash", "schema_check")
    params = {k: report[k] for k in keys if k in report}
    counts = {
        k: v for k, v in report.items()
        if isinstance(v, (int, float)) and k not in ("seed",)
    }
    with mlflow.start_run(run_name="prepare_data"):
        mlflow.log_params(params)
        mlflow.log_metrics(counts)
        for name in ("quality_report.json", "quality_report.md"):
            if (out / name).exists():
                mlflow.log_artifact(str(out / name))


def main():
    parser = argparse.ArgumentParser(description="Prepare listings for training.")
    parser.add_argument("--input", nargs="+", default=["data/sample/sample_listings.csv"],
                        help="one CSV, or several to mix datasets (one source per file)")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--schema", default="configs/schema.yaml")
    parser.add_argument("--min-model-count", type=int, default=0,
                        help="drop make/model pairs with fewer listings (0 = off)")
    parser.add_argument("--min-city-count", type=int, default=0,
                        help="group cities with fewer listings as 'Other' (0 = off)")
    parser.add_argument("--check-low-mileage", action="store_true",
                        help="drop old cars with implausibly low mileage")
    parser.add_argument("--rescale-prices", action="store_true",
                        help="move later sources to the first source's price level")
    parser.add_argument("--report", default="reports/data_quality.md",
                        help="where to write the human-readable quality report")
    parser.add_argument("--log-mlflow", action="store_true",
                        help="log row counts and file hashes to MLflow")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    frames = {Path(f).stem: pd.read_csv(f, encoding="utf-8-sig")  # utf-8-sig drops the BOM
              for f in args.input}
    train, test, report = prepare(
        frames,
        load_column_map(args.schema),
        min_model_count=args.min_model_count,
        min_city_count=args.min_city_count,
        check_low_mileage=args.check_low_mileage,
        rescale_prices=args.rescale_prices,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    report["train_hash"] = file_hash(out / "train.csv")
    report["test_hash"] = file_hash(out / "test.csv")
    (out / "quality_report.json").write_text(json.dumps(report, indent=2))
    write_markdown_report(report, out / "quality_report.md")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(report, args.report)
    if args.log_mlflow:
        log_to_mlflow(report, out, args.tracking_uri)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()