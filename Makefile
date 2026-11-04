PY := .venv/bin/python
IMAGE := carcompass
TAG := dev

.PHONY: setup prepare train prepare-real train-real prepare-two train-two run test lint format mlflow-ui docker-build docker-run all

setup:
	python3.11 -m venv .venv || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

prepare:
	$(PY) -m src.prepare_data --input data/sample/sample_listings.csv

train: prepare
	$(PY) -m src.train

# Real Dataset A (put the CSV at data/raw/pakistan_used_cars.csv first).
prepare-real:
	$(PY) -m src.prepare_data --input data/raw/pakistan_used_cars.csv \
		--min-model-count 5 --min-city-count 20 --check-low-mileage \
		--log-mlflow --tracking-uri sqlite:///mlflow.db

# Trains on the real data and registers the champion as @production in MLflow.
train-real: prepare-real
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(PY) -m src.train

# Two price datasets mixed (A first, then the 2022 PakWheels scrape).
# Results go to separate folders so the A-only champion in models/ is untouched.
prepare-two:
	$(PY) -m src.prepare_data \
		--input data/raw/pakistan_used_cars.csv data/raw/pakwheels_2022.csv \
		--out data/processed_merged --report reports/data_quality_merged.md \
		--min-model-count 5 --min-city-count 20 --check-low-mileage --rescale-prices \
		--log-mlflow --tracking-uri sqlite:///mlflow.db

train-two: prepare-two
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(PY) -m src.train \
		--train data/processed_merged/train.csv --test data/processed_merged/test.csv \
		--out models_merged --experiment carcompass-price-merged --no-register

run:
	$(PY) -m uvicorn src.app:app --reload --port 8000

test:
	$(PY) -m pytest -v

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff check . --fix

mlflow-ui:
	$(PY) -m mlflow ui --port 5000

docker-build:
	docker build --build-arg APP_VERSION=$(TAG) -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE):$(TAG)

all: setup train test
