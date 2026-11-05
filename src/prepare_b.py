"""Prepare Dataset B independently for training and evaluation.

Run:
    python -m src.prepare_b

Dataset B is cleaned and split independently. No Dataset A data,
test set, city list, or price adjustment is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.prepare_data import (
    prepare,
    load_column_map,
    file_hash,
    write_markdown_report,
    log_to_mlflow,
)


def load_dataset_b(path) -> pd.DataFrame:
    """Load usedCars.json and convert it into a flat DataFrame.

    Dataset B stores manufacturer and model separately.
    We combine them into 'name' so the existing schema mapping
    can split them into the standard make and model columns.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    df = pd.json_normalize(data["usedCars"])

    maker = df["manufacturer"].astype(str)
    model = df["model"].astype(str)

    # Match Dataset A's naming convention for Range Rover.
    is_range_rover = model.str.lower().str.startswith("range rover")

    df["name"] = np.where(
        is_range_rover,
        model,
        maker + " " + model,
    )

    return df.drop(columns=["manufacturer", "model"])


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Dataset B independently."
    )

    parser.add_argument(
        "--input",
        default="data/raw/usedCars.json",
        help="Path to Dataset B JSON file.",
    )
    parser.add_argument(
        "--out",
        default="data/processed_b",
        help="Separate output folder for Dataset B.",
    )
    parser.add_argument(
        "--schema",
        default="configs/schema_b.yaml",
        help="Dataset B schema mapping.",
    )
    parser.add_argument(
        "--min-model-count",
        type=int,
        default=0,
        help="Drop make/model pairs with fewer Dataset B listings.",
    )
    parser.add_argument(
        "--min-city-count",
        type=int,
        default=0,
        help="Group cities with fewer Dataset B listings as Other.",
    )
    parser.add_argument(
        "--check-low-mileage",
        action="store_true",
        help="Drop old cars with implausibly low mileage.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of Dataset B reserved for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log Dataset B preparation details to MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI.",
    )

    args = parser.parse_args()

    # 1. Load Dataset B.
    raw_b = load_dataset_b(args.input)

    # 2. Clean and split Dataset B using the existing preparation functions.
    train, test, report = prepare(
        raw=raw_b,
        column_map=load_column_map(args.schema),
        test_size=args.test_size,
        seed=args.seed,
        min_model_count=args.min_model_count,
        min_city_count=args.min_city_count,
        check_low_mileage=args.check_low_mileage,
    )

    # 3. Save into a separate folder.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    # 4. Record hashes so we can identify these exact data files later.
    report["train_hash"] = file_hash(out / "train.csv")
    report["test_hash"] = file_hash(out / "test.csv")

    # 5. Save quality reports.
    (out / "quality_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        report,
        out / "quality_report.md",
    )

    # 6. Optionally log the preparation run separately in MLflow.
    if args.log_mlflow:
        log_to_mlflow(
            report,
            out,
            tracking_uri=args.tracking_uri,
            experiment="carcompass-data-b",
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()