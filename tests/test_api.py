import pytest
from fastapi.testclient import TestClient

from src.app import create_app

CAR = {"make": "Toyota", "model": "Corolla", "year": 2016, "mileage": 75000}


@pytest.fixture()
def client(model_dir):
    return TestClient(create_app(str(model_dir)))


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_returns_price_range_and_version(client):
    res = client.post("/predict", json=CAR)
    assert res.status_code == 200
    body = res.json()
    assert body["range_low_pkr"] <= body["estimated_price_pkr"] <= body["range_high_pkr"]
    assert body["model_version"]
    assert "assumed_fields" in body


@pytest.mark.parametrize("bad", [
    {**CAR, "year": 1850},
    {**CAR, "mileage": -5},
    {**CAR, "make": ""},
    {"make": "Toyota", "model": "Corolla"},
])
def test_predict_rejects_invalid_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_explain_returns_top_factors(client):
    body = client.post("/explain", json=CAR).json()
    assert len(body["top_factors"]) >= 1
    assert "feature" in body["top_factors"][0]


def test_metrics_counts_requests(client):
    client.get("/health")
    client.post("/predict", json=CAR)
    body = client.get("/metrics").json()
    assert body["request_count"] >= 2
    assert body["avg_latency_ms"] >= 0


def test_predict_returns_503_without_model(tmp_path):
    client = TestClient(create_app(str(tmp_path)))
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/predict", json=CAR).status_code == 503
