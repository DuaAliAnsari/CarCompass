import pytest

from src.model_utils import load_artifacts, make_frame, predict_price


def test_load_artifacts_returns_pipeline_and_metadata(model_dir):
    art = load_artifacts(model_dir)
    for key in ("version", "q10", "q90", "defaults", "importances", "metrics"):
        assert key in art.meta
    pred = art.pipeline.predict(make_frame({"make": "Toyota", "model": "Corolla",
                                            "year": 2016, "mileage": 75000}))
    assert len(pred) == 1


def test_load_artifacts_raises_when_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(tmp_path)


def test_predict_price_flags_assumed_fields_and_widens_range(model_dir):
    art = load_artifacts(model_dir)
    base = {"make": "Toyota", "model": "Corolla", "year": 2016, "mileage": 75000}
    sparse = predict_price(art, base)
    full = predict_price(art, {**base, "engine_cc": 1300, "transmission": "Manual",
                               "fuel": "Petrol", "city": "Karachi"})
    assert set(sparse["assumed_fields"]) == {"engine_cc", "transmission", "fuel", "city"}
    assert full["assumed_fields"] == []
    assert sparse["range_low_pkr"] <= sparse["estimated_price_pkr"] <= sparse["range_high_pkr"]
    sparse_width = sparse["range_high_pkr"] - sparse["range_low_pkr"]
    full_width = full["range_high_pkr"] - full["range_low_pkr"]
    assert sparse_width >= full_width


def test_newer_car_costs_more_than_older_car(model_dir):
    art = load_artifacts(model_dir)
    car = {"make": "Toyota", "model": "Corolla"}
    new = predict_price(art, {**car, "year": 2023, "mileage": 10000})
    old = predict_price(art, {**car, "year": 2006, "mileage": 200000})
    assert new["estimated_price_pkr"] > old["estimated_price_pkr"]
