PY := .venv/bin/python
IMAGE := carcompass
TAG := dev
RAW_A := data/raw/pakistan_used_cars.csv
RAW_B := data/raw/usedCars.json

.PHONY: setup prepare train prepare-real train-real prepare-multi train-multi run test lint format mlflow-ui docker-build docker-run all

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

# Five model families on one shared A+B dataset (put both raw files in data/raw/).
# Writes models/manifest.json, which the API uses to route requests.
prepare-multi:
	$(PY) -m src.prepare_multi --input-a $(RAW_A) --input-b $(RAW_B)

train-multi: prepare-multi
	$(PY) -m src.train_families

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
