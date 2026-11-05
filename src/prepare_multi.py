"""Build one shared, leakage-safe dataset from Dataset A and Dataset B.

Every model family (A only, B only, A+B, 2010+, pre-2010) is trained and compared on
this one file, so their scores are measured on exactly the same rows. Run:

    python -m src.prepare_multi --input-a data/raw/pakistan_used_cars.csv \
        --input-b data/raw/usedCars.json

Steps:
1. Map both files to the standard schema and validate them with the usual rules.
2. Remove duplicates inside each source, then across sources: a B listing whose
   listing_group (make, model, year, mileage, city) is also in A is the same car
   listed twice, so only the A copy is kept.
3. Apply the quality rules (rare make/model, rare city, low mileage) on A+B together.
4. One group-aware split into fit / val / cal / test (seed 42). A listing group is
   always on one side only. The split is stored in a `split` column.
   - fit: model training   - val: choosing the champion of every family
   - cal: the price range  - test: final scores only, never used to choose
5. Scale B's 2021 prices to A's price level. The factor is measured on fit rows only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.prepare_combined import estimate_price_factor, load_dataset_b
from src.prepare_data import (
    add_listing_group,
    apply_quality_rules,
    check_schema,
    drop_duplicates,
    file_hash,
    load_column_map,
    map_schema,
    validate,
)

SPLITS = ("fit", "val", "cal", "test")


def clean_source(raw: pd.DataFrame, column_map: dict, source: str,
                 check_low_mileage: bool = False):
    """Map, validate and de-duplicate one source. Returns (rows, report)."""
    df = map_schema(raw, column_map)
    report = {"rows_in": len(df)}
    df, rep = validate(df)
    report["dropped_missing"] = rep["dropped_missing"]
    report["dropped_out_of_range"] = rep["dropped_out_of_range"]
    report["engine_set_to_missing"] = rep["engine_set_to_missing"]
    df, report["dropped_duplicates"] = drop_duplicates(df)
    df, rep = apply_quality_rules(df, check_low_mileage=check_low_mileage)
    report["dropped_low_mileage"] = rep.get("dropped_low_mileage", 0)
    report["rows_clean"] = len(df)
    return df.assign(source=source), report


def drop_cross_source_duplicates(a: pd.DataFrame, b: pd.DataFrame):
    """Drop B rows that describe a car already listed in A (same listing_group)."""
    a, b = add_listing_group(a), add_listing_group(b)
    dup = b["listing_group"].isin(set(a["listing_group"]))
    return a, b[~dup], int(dup.sum())


def assign_splits(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2,
                  val_size: float = 0.1, cal_size: float = 0.1) -> pd.Series:
    """Group-aware fit/val/cal/test labels.

    test_size is a share of all rows; val_size and cal_size are shares of what is
    left after the test rows are taken out.
    """
    labels = pd.Series("fit", index=df.index)
    groups = df["listing_group"]

    def take(index, size, label, offset):
        splitter = GroupShuffleSplit(n_splits=1, test_size=size,
                                     random_state=seed + offset)
        sub = df.loc[index]
        _, picked = next(splitter.split(sub, groups=groups.loc[index]))
        labels.loc[sub.index[picked]] = label

    take(df.index, test_size, "test", 0)
    take(labels.index[labels == "fit"], val_size, "val", 1)
    take(labels.index[labels == "fit"], cal_size / (1 - val_size), "cal", 2)
    return labels


def prepare_multi(raw_a, map_a, raw_b, map_b, seed=42, min_model_count=5,
                  min_city_count=20, check_low_mileage=True, adjust_prices=True):
    """Full preparation. Returns (listings with source and split columns, report)."""
    a, rep_a = clean_source(raw_a, map_a, "A", check_low_mileage)
    b, rep_b = clean_source(raw_b, map_b, "B", check_low_mileage)
    a, b, cross_dupes = drop_cross_source_duplicates(a, b)

    df = pd.concat([a, b], ignore_index=True)
    df, rules = apply_quality_rules(
        df.drop(columns="listing_group"), min_model_count, min_city_count
    )
    df = add_listing_group(df)
    report = {"a": rep_a, "b": rep_b, "dropped_b_also_in_a": cross_dupes, **rules}
    report["schema_check"] = check_schema(df.drop(columns="source"))

    df["split"] = assign_splits(df, seed)
    fit = df[df["split"] == "fit"]
    factor, cells = estimate_price_factor(fit[fit["source"] == "A"],
                                          fit[fit["source"] == "B"])
    if not adjust_prices:
        factor = 1.0
    is_b = df["source"] == "B"
    df.loc[is_b, "price"] = (df.loc[is_b, "price"] * factor).round()
    report["b_price_factor"] = round(factor, 4)
    report["b_price_factor_cells"] = cells

    counts = df.groupby(["split", "source"]).size().unstack(fill_value=0)
    report["rows_out"] = len(df)
    report["rows_by_split"] = {
        s: {src: int(counts.loc[s].get(src, 0)) for src in ("A", "B")}
        for s in SPLITS if s in counts.index
    }
    report["rows_pre_2010"] = int((df["year"] < 2010).sum())
    report["rows_2010_and_later"] = int((df["year"] >= 2010).sum())
    report["seed"] = seed
    return df, report


def write_report(report: dict, path) -> None:
    a, b = report["a"], report["b"]
    rows = [
        ("Rows read", a["rows_in"], b["rows_in"]),
        ("Dropped: missing required values", a["dropped_missing"], b["dropped_missing"]),
        ("Dropped: out of range", a["dropped_out_of_range"], b["dropped_out_of_range"]),
        ("Dropped: duplicates (within source)", a["dropped_duplicates"],
         b["dropped_duplicates"]),
        ("Dropped: low mileage", a["dropped_low_mileage"], b["dropped_low_mileage"]),
        ("Clean rows", a["rows_clean"], b["rows_clean"]),
        ("Dropped: same car already in A", "", report["dropped_b_also_in_a"]),
    ]
    lines = ["# Data quality report: shared A+B dataset", "",
             "| Step | Dataset A | Dataset B |", "|---|---|---|"]
    lines += [f"| {name} | {va} | {vb} |" for name, va, vb in rows]
    split = report["rows_by_split"]
    lines += [
        "",
        f"Rules on A+B together: dropped {report.get('dropped_rare_model', 0)} rows of "
        f"rare make/model pairs, grouped {report.get('listings_with_city_grouped', 0)} "
        "rows of rare cities as 'Other'.",
        "",
        "| Split | Dataset A | Dataset B | Total | Used for |",
        "|---|---|---|---|---|",
    ]
    used_for = {"fit": "training", "val": "choosing champions and routing",
                "cal": "price range", "test": "final scores only"}
    for s in SPLITS:
        if s in split:
            n_a, n_b = split[s]["A"], split[s]["B"]
            lines.append(f"| {s} | {n_a} | {n_b} | {n_a + n_b} | {used_for[s]} |")
    lines += [
        f"| **all** | | | **{report['rows_out']}** | |",
        "",
        f"Pre-2010 cars: {report['rows_pre_2010']}, 2010 and later: "
        f"{report['rows_2010_and_later']}.",
        f"B prices scaled by {report['b_price_factor']} (A's prices are this many "
        f"times B's 2021 prices for the same car), from "
        f"{report['b_price_factor_cells']} make/model/year groups in the fit rows.",
        f"Seed: {report['seed']}. Pandera schema check: {report['schema_check']}.",
    ]
    if "data_hash" in report:
        lines.append(f"File hash: `{report['data_hash']}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Build the shared A+B dataset.")
    p.add_argument("--input-a", default="data/raw/pakistan_used_cars.csv")
    p.add_argument("--input-b", default="data/raw/usedCars.json")
    p.add_argument("--schema-a", default="configs/schema.yaml")
    p.add_argument("--schema-b", default="configs/schema_b.yaml")
    p.add_argument("--out", default="data/processed_multi")
    p.add_argument("--report", default="reports/data_quality_multi.md")
    p.add_argument("--no-price-adjust", action="store_true")
    args = p.parse_args()

    df, report = prepare_multi(
        pd.read_csv(args.input_a, encoding="utf-8-sig"), load_column_map(args.schema_a),
        load_dataset_b(args.input_b), load_column_map(args.schema_b),
        adjust_prices=not args.no_price_adjust,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "listings.csv", index=False)
    report["data_hash"] = file_hash(out / "listings.csv")
    (out / "quality_report.json").write_text(json.dumps(report, indent=2))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_report(report, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
