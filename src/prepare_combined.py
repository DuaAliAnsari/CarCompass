"""Add dataset B (second PakWheels listing set, JSON) to the training data.

Dataset A is prepared exactly as before by prepare_data.prepare(), so its test set
does not change. Dataset B is prepared the same way and added to the TRAINING data
only. Run:

    python -m src.prepare_combined --input-a data/raw/pakistan_used_cars.csv \
        --input-b data/raw/usedCars.json --min-model-count 5 --min-city-count 20 \
        --check-low-mileage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.prepare_data import (
    add_listing_group,
    apply_quality_rules,
    check_schema,
    drop_duplicates,
    file_hash,
    load_column_map,
    map_schema,
    prepare,
    validate,
    write_markdown_report,
)

PRICE_FACTOR_KEYS = ["make", "model", "year"]


def load_dataset_b(path) -> pd.DataFrame:
    """Read usedCars.json into a flat table and build a 'name' column.

    Dataset B has separate manufacturer and model columns, but its model names are
    more detailed than dataset A's ("Corolla Axio" against "Corolla"). We join them
    into one name and let split_name() cut it exactly as it does for dataset A, so
    both datasets use the same make and model words.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    df = pd.json_normalize(data["usedCars"])
    maker = df["manufacturer"].astype(str)
    model = df["model"].astype(str)
    # Dataset A lists Range Rovers without the "Land Rover" make, so do the same.
    is_range_rover = model.str.lower().str.startswith("range rover")
    df["name"] = np.where(is_range_rover, model, maker + " " + model)
    return df.drop(columns=["manufacturer", "model"])


def estimate_price_factor(train_a: pd.DataFrame, b: pd.DataFrame, min_rows: int = 5):
    """How much higher are dataset A's prices than dataset B's (2021) for the same car?

    For every make/model/year that has at least min_rows listings in both datasets,
    take the median price in each and divide A's by B's. The median of those ratios
    is the factor. Only A's training rows are used, so A's test prices never leak in.
    Returns (factor, number_of_cells_used).
    """
    def cells(df):
        return df.groupby(PRICE_FACTOR_KEYS)["price"].agg(median="median", n="size")

    # merge on the shared make/model/year index (same result as the old DataFrame.join,
    # without the deprecated lsuffix/rsuffix keywords)
    joined = cells(train_a).merge(
        cells(b), left_index=True, right_index=True, how="inner",
        suffixes=("_a", "_b"),
    )
    joined = joined[(joined["n_a"] >= min_rows) & (joined["n_b"] >= min_rows)]
    if joined.empty:
        return 1.0, 0
    return float((joined["median_a"] / joined["median_b"]).median()), len(joined)


def _pair(df: pd.DataFrame) -> pd.Series:
    """'Toyota|Corolla' for every row, used to count listings per make/model."""
    return df["make"] + "|" + df["model"]


def prepare_b(raw_b: pd.DataFrame, column_map: dict, train_a: pd.DataFrame,
              test_a: pd.DataFrame, min_model_count: int = 0,
              check_low_mileage: bool = False):
    """Clean dataset B with the same rules as A. Returns (rows for training, report).

    B rows are dropped if their listing_group appears in A's test set, so a car in the
    test set can never have a near-copy in the training data.
    """
    df = map_schema(raw_b, column_map)
    report = {"b_rows_in": len(df)}
    df, rep = validate(df)
    report["b_dropped_missing"] = rep["dropped_missing"]
    report["b_dropped_out_of_range"] = rep["dropped_out_of_range"]
    df, dupes = drop_duplicates(df)
    report["b_dropped_duplicates"] = dupes
    df, rep = apply_quality_rules(df, check_low_mileage=check_low_mileage)
    report["b_dropped_low_mileage"] = rep.get("dropped_low_mileage", 0)

    # Use A's city list: a city that A does not have becomes 'Other'.
    known = set(train_a["city"].dropna()) | set(test_a["city"].dropna())
    unknown_city = ~df["city"].isin(known)  # also true when the city is missing
    df.loc[unknown_city, "city"] = "Other"
    report["b_city_set_to_other"] = int(unknown_city.sum())

    # Keep a make/model only if it has at least min_model_count rows in A train + B.
    if min_model_count > 0:
        counts = pd.concat([_pair(train_a), _pair(df)]).value_counts()
        rare = _pair(df).map(counts) < min_model_count
        report["b_dropped_rare_model"] = int(rare.sum())
        df = df[~rare]

    df = add_listing_group(df)
    leak = df["listing_group"].isin(set(test_a["listing_group"]))
    report["b_dropped_group_in_test"] = int(leak.sum())
    df = df[~leak].reset_index(drop=True)
    report["b_rows_kept"] = len(df)
    return df, report


def prepare_combined(raw_a, map_a, raw_b, map_b, seed=42, min_model_count=0,
                     min_city_count=0, check_low_mileage=False, adjust_prices=True):
    """Prepare A as before, prepare B, and add B to the training data."""
    train_a, test_a, report = prepare(
        raw_a, map_a, seed=seed, min_model_count=min_model_count,
        min_city_count=min_city_count, check_low_mileage=check_low_mileage)
    b, b_report = prepare_b(raw_b, map_b, train_a, test_a, min_model_count,
                            check_low_mileage)
    report.update(b_report)

    factor, n_cells = estimate_price_factor(train_a, b)
    report["b_price_factor"] = round(factor, 4) if adjust_prices else 1.0
    report["b_price_factor_cells"] = n_cells
    if adjust_prices:
        b = b.assign(price=(b["price"] * factor).round())

    train = pd.concat([train_a.assign(source="A"), b.assign(source="B")],
                      ignore_index=True)
    test = test_a.assign(source="A")
    report["train_rows_a"], report["train_rows_b"] = len(train_a), len(b)
    report["train_rows"], report["test_rows"] = len(train), len(test)
    report["schema_check_combined"] = check_schema(train.drop(columns="source"))
    return train, test, report


B_SECTION = """
## Dataset B (added to the training data only)

| Step | Rows |
|---|---|
| Rows read | {b_rows_in} |
| Dropped: missing required values | {b_dropped_missing} |
| Dropped: out of range | {b_dropped_out_of_range} |
| Dropped: duplicates | {b_dropped_duplicates} |
| Dropped: low mileage | {b_dropped_low_mileage} |
| Dropped: rare make/model | {b_dropped_rare_model} |
| Dropped: listing group also in the test set | {b_dropped_group_in_test} |
| City set to 'Other' | {b_city_set_to_other} |
| **Rows added to training** | **{b_rows_kept}** |

Price factor applied to B (A's prices are this many times B's for the same car):
{b_price_factor}, from {b_price_factor_cells} make/model/year groups.
Train rows: {train_rows_a} from A + {train_rows_b} from B = {train_rows}.
Test rows: {test_rows} (A only). Pandera schema check of the combined data:
{schema_check_combined}
"""


def main():
    p = argparse.ArgumentParser(description="Prepare dataset A plus dataset B.")
    p.add_argument("--input-a", default="data/raw/pakistan_used_cars.csv")
    p.add_argument("--input-b", default="data/raw/usedCars.json")
    p.add_argument("--schema-a", default="configs/schema.yaml")
    p.add_argument("--schema-b", default="configs/schema_b.yaml")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--report", default="reports/data_quality.md")
    p.add_argument("--min-model-count", type=int, default=0)
    p.add_argument("--min-city-count", type=int, default=0)
    p.add_argument("--check-low-mileage", action="store_true")
    p.add_argument("--no-price-adjust", action="store_true",
                   help="do not scale B's 2021 prices up to A's price level")
    args = p.parse_args()

    raw_a = pd.read_csv(args.input_a, encoding="utf-8-sig")
    raw_b = load_dataset_b(args.input_b)
    train, test, report = prepare_combined(
        raw_a, load_column_map(args.schema_a), raw_b, load_column_map(args.schema_b),
        min_model_count=args.min_model_count, min_city_count=args.min_city_count,
        check_low_mileage=args.check_low_mileage,
        adjust_prices=not args.no_price_adjust)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    report["train_hash"] = file_hash(out / "train.csv")
    report["test_hash"] = file_hash(out / "test.csv")
    (out / "quality_report.json").write_text(json.dumps(report, indent=2))
    for path in (out / "quality_report.md", Path(args.report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(report, path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(B_SECTION.format(**report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()