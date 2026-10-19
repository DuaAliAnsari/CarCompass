PY := .venv/bin/python
IMAGE := carcompass
TAG := dev

.PHONY: setup prepare train run test lint format mlflow-ui docker-build docker-run all

setup:
	python3.11 -m venv .venv || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

prepare:
	$(PY) -m src.prepare_data --input data/sample/sample_listings.csv

train: prepare
	$(PY) -m src.train

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
