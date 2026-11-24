"""WSP experiment serving helpers and API switch."""
from fastapi.testclient import TestClient

from src.app import create_app
from src.wsp_serve import WspExperiment, _parse_blend_weights

CAR = {"make": "Toyota", "model": "Corolla", "year": 2016, "mileage": 75000}


def test_parse_blend_weights_from_note():
    a, b = _parse_blend_weights("0.80·lgbm_greedy + 0.00·lgbm_uniform + 0.20·WSP")
    assert abs(a - 0.8) < 1e-6 and abs(b - 0.2) < 1e-6


def test_wsp_load_missing_returns_none(tmp_path):
    assert WspExperiment.load(tmp_path) is None


def test_predict_wsp_without_artifact_is_503(model_dir):
    client = TestClient(create_app(str(model_dir)))
    res = client.post("/predict", json={**CAR, "predictor": "wsp"})
    assert res.status_code == 503


def test_health_reports_wsp_flag(model_dir):
    body = TestClient(create_app(str(model_dir))).get("/health").json()
    assert "wsp_loaded" in body
    assert body["wsp_loaded"] is False
