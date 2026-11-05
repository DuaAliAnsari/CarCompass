import json

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.model_router import ModelRouter
from src.prepare_multi import prepare_multi
from src.train_families import train_families
from tests.conftest import SMALL_CONFIGS
from tests.test_prepare_combined import B_MAP, make_a, make_b

FAMILIES = {
    "b_weight": 0.3,
    "families": {
        "price_global_ab": {"sources": ["A", "B"], "role": "default_candidate"},
        "price_dataset_a": {"sources": ["A"], "role": "default_candidate"},
        "price_new_cars_2010plus": {"sources": ["A", "B"], "year_min": 2010,
                                    "role": "specialist"},
        "price_old_cars_pre2010": {"sources": ["A", "B"], "year_max": 2009,
                                   "role": "specialist"},
    },
}
# min_gain below -100% switches every specialist on, so routing can be tested.
ROUTING = {"default_candidates": ["price_global_ab", "price_dataset_a"],
           "min_gain": -10.0,
           "specialists": {"price_new_cars_2010plus": {"year_min": 2010},
                           "price_old_cars_pre2010": {"year_max": 2009}}}


@pytest.fixture(scope="module")
def family_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("families")
    a, b = make_a(900), make_b(900)
    a["year"] = 2000 + a.index % 24  # cars on both sides of 2010
    b["modelDate"] = 2000 + b.index % 24
    df, _ = prepare_multi(a, {}, b, B_MAP, min_model_count=0, min_city_count=0,
                          check_low_mileage=False)
    train_families(df, FAMILIES, ROUTING, SMALL_CONFIGS, base / "models",
                   base / "comparison.md", tracking_uri=f"file:{base / 'mlruns'}",
                   do_register=False, ablation=False)
    return base / "models"


def test_manifest_lists_every_family_and_the_served_models(family_dir):
    manifest = json.loads((family_dir / "manifest.json").read_text())
    assert set(manifest["families"]) == set(FAMILIES["families"])
    assert manifest["default"] in ROUTING["default_candidates"]
    assert manifest["served"][0] == manifest["default"]
    assert (family_dir / "price_model.joblib").exists()  # copy for older tools


def test_2010_boundary_goes_to_the_right_specialist(family_dir):
    router = ModelRouter.load(family_dir)
    assert router.choose(2009)[0] == "price_old_cars_pre2010"
    assert router.choose(2010)[0] == "price_new_cars_2010plus"


def test_disabled_specialist_falls_back_to_default(family_dir, tmp_path):
    for f in family_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["routing"]["specialists"]["price_old_cars_pre2010"]["enabled"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    router = ModelRouter.load(tmp_path)
    assert router.choose(2005)[0] == manifest["default"]
    assert router.choose(2015)[0] == "price_new_cars_2010plus"


def test_missing_specialist_file_falls_back_to_default(family_dir, tmp_path):
    for f in family_dir.iterdir():
        if f.name != "price_new_cars_2010plus.joblib":
            (tmp_path / f.name).write_bytes(f.read_bytes())
    router = ModelRouter.load(tmp_path)
    assert router.choose(2020)[0] == router.default_name
    assert "price_new_cars_2010plus" not in router.models


def test_api_names_the_model_it_used(family_dir):
    client = TestClient(create_app(str(family_dir)))
    car = {"make": "Toyota", "model": "Corolla", "mileage": 75000}
    old = client.post("/predict", json={**car, "year": 2005}).json()
    new = client.post("/predict", json={**car, "year": 2018, "assembly": "Local"}).json()
    assert old["selected_model"] == "price_old_cars_pre2010"
    assert new["selected_model"] == "price_new_cars_2010plus"
    body = client.get("/models").json()
    assert set(body["models"]) == {body["default"], "price_old_cars_pre2010",
                                   "price_new_cars_2010plus"}
    assert client.get("/health").json()["models_served"]
