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


def test_metrics_by_source_reports_each_source():
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyRegressor

    from src.model_utils import FEATURES
    from src.train import metrics_by_source

    df = pd.DataFrame({c: [1.0] * 6 for c in FEATURES})
    df["price"] = [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
    df["source"] = ["a", "a", "a", "b", "b", "b"]
    pipe = DummyRegressor(strategy="constant", constant=np.log1p(100.0))
    pipe.fit(df[FEATURES], df["price"])
    out = metrics_by_source(pipe, df)
    assert out["a"]["mae_pkr"] < 1e-6 and out["b"]["mae_pkr"] > 50
    assert metrics_by_source(pipe, df.assign(source="a")) == {}
