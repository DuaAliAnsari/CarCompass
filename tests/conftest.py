import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.prepare_data import load_column_map, prepare  # noqa: E402
from src.train import train_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMALL_CONFIGS = [
    {"name": "baseline_median", "model": "baseline", "params": {}},
    {"name": "ridge_alpha_1", "model": "ridge", "params": {"alpha": 1.0}},
    {"name": "rf_small", "model": "random_forest", "params": {"n_estimators": 20}},
]


@pytest.fixture(scope="session")
def raw_sample():
    return pd.read_csv(ROOT / "data" / "sample" / "sample_listings.csv")


@pytest.fixture(scope="session")
def column_map():
    return load_column_map(ROOT / "configs" / "schema.yaml")


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, raw_sample, column_map):
    """Train a small model once for the whole test session."""
    base = tmp_path_factory.mktemp("trained")
    train, test, _ = prepare(raw_sample, column_map)
    train_all(train, test, SMALL_CONFIGS, base / "models",
              tracking_uri=f"file:{base / 'mlruns'}", data_hash="test")
    return base / "models"