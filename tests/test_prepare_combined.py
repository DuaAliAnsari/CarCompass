import json

import numpy as np
import pandas as pd

from src.prepare_combined import (
    estimate_price_factor,
    load_dataset_b,
    prepare_b,
    prepare_combined,
)
from src.prepare_data import prepare
from src.train import B_WEIGHT, fit_weights

B_MAP = {
    "modelDate": "year",
    "mileageFromOdometer": "mileage",
    "price": "price",
    "fuelType": "fuel",
    "vehicleTransmission": "transmission",
    "vehicleEngine.engineDisplacement": "engine_cc",
    "extraFeatures.RegisteredIn": "city",
}


def make_a(n=300):
    """Small fake dataset A in the standard column names."""
    rng = np.random.default_rng(0)
    cars = [("Toyota", "Corolla"), ("Honda", "Civic"), ("Suzuki", "Cultus")]
    rows = []
    for i in range(n):
        make, model = cars[i % 3]
        rows.append({
            "make": make, "model": model, "year": 2010 + i % 10,
            "mileage": 10_000 + i * 137,
            "price": 2_000_000 + int(rng.integers(0, 1_000_000)),
            "engine_cc": 1300, "transmission": "Manual", "fuel": "Petrol",
            "city": ["Lahore", "Karachi"][i % 2],
        })
    return pd.DataFrame(rows)


def make_b(n=300):
    """Small fake dataset B as load_dataset_b returns it (one 'name' column)."""
    rng = np.random.default_rng(1)
    cars = [("Toyota", "Corolla Axio"), ("Honda", "Civic"), ("Suzuki", "Cultus")]
    rows = []
    for i in range(n):
        maker, model = cars[i % 3]
        rows.append({
            "name": f"{maker} {model}", "modelDate": 2010 + i % 10,
            "mileageFromOdometer": f"{20_000 + i * 211:,} km",
            "price": 1_400_000 + int(rng.integers(0, 700_000)),
            "fuelType": "Petrol", "vehicleTransmission": "Manual",
            "vehicleEngine.engineDisplacement": "1300cc",
            "extraFeatures.RegisteredIn": ["Lahore", "Peshawar"][i % 2],
        })
    return pd.DataFrame(rows)


def test_load_dataset_b_builds_one_name_column(tmp_path):
    listing = {"manufacturer": "Toyota", "model": "Corolla Axio", "price": 1}
    rover = {"manufacturer": "Land Rover", "model": "Range Rover Sport", "price": 1}
    path = tmp_path / "usedCars.json"
    path.write_text(json.dumps({"usedCars": [listing, rover]}), encoding="utf-8")
    df = load_dataset_b(path)
    assert df["name"].tolist() == ["Toyota Corolla Axio", "Range Rover Sport"]
    assert "model" not in df.columns and "manufacturer" not in df.columns


def test_b_make_and_model_are_cut_like_dataset_a_and_cities_match():
    train_a, test_a, _ = prepare(make_a(), {})
    b, report = prepare_b(make_b(), B_MAP, train_a, test_a)
    # "Corolla Axio" is cut to "Corolla", the same as dataset A
    assert set(b["model"]) == {"Corolla", "Civic", "Cultus"}
    assert set(b["city"]) == {"Lahore", "Other"}  # Peshawar is not a city in A
    assert report["b_rows_kept"] == len(b)


def test_price_factor_is_the_median_ratio_of_a_to_b():
    cell = {"make": "Toyota", "model": "Corolla", "year": 2016}
    a = pd.DataFrame([{**cell, "price": 3_000_000}] * 6)
    b = pd.DataFrame([{**cell, "price": 2_000_000}] * 6)
    assert estimate_price_factor(a, b) == (1.5, 1)
    no_overlap = b.assign(year=1999)
    assert estimate_price_factor(a, no_overlap) == (1.0, 0)


def test_test_set_is_a_only_unchanged_and_has_no_group_leakage():
    train, test, report = prepare_combined(make_a(), {}, make_b(), B_MAP, seed=42)
    _, test_a, _ = prepare(make_a(), {}, seed=42)
    assert (test["source"] == "A").all()
    assert test["listing_group"].tolist() == test_a["listing_group"].tolist()
    assert set(train["source"]) == {"A", "B"}
    assert set(train["listing_group"]).isdisjoint(set(test["listing_group"]))
    assert report["b_price_factor"] > 1  # B's older prices are scaled up
    assert report["train_rows"] == report["train_rows_a"] + report["train_rows_b"]


def test_b_rows_get_a_lower_fit_weight_and_a_only_data_has_none():
    df = pd.DataFrame({"source": ["A", "B"]})
    assert fit_weights(df).tolist() == [1.0, B_WEIGHT]
    assert fit_weights(df.drop(columns="source")) is None
