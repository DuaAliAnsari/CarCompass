# CarCompass: Used-Car Price Estimation for Pakistan

CarCompass takes a car's make, model, year and mileage and returns an estimated price in PKR with a range and the top factors behind it. It is served through a FastAPI service that is tracked in MLflow, containerised with Docker, tested in CI and deployed to the cloud.

> Estimates reflect historical asking prices. They are a guide, not a valuation.

## Architecture

```text
Dataset A (raw) ──► src/prepare_data.py ──► src/train.py ──► src/app.py
                       validate, dedup,       train models       FastAPI
                       split                  + MLflow            /predict
                       data/processed/         models/             /explain
                                                                  /health
Dataset B (raw) ──► src/prepare_b.py ──► data/processed_b/       /metrics
                       flatten JSON,
                       validate, dedup,
                       split

                         Docker image ──► Render / Railway
```

## Quick start

Requires Python 3.11 (Linux, macOS, or WSL on Windows) and `make`.

```bash
git clone <your-repo-url> && cd carcompass
make setup      # creates .venv and installs dependencies
make train      # prepares data, trains models, logs runs to MLflow, saves models/
make run        # API at http://localhost:8000 (docs at /docs)
```

Try it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Corolla","year":2016,"mileage":75000}'
```

View experiments: `make mlflow-ui`, then open http://localhost:5000.

## Dataset and task

- **Task:** Regression — predict used-car asking prices in PKR, trained on log-price.
- **Metric:** Mean Absolute Error (MAE) in PKR. Median Absolute Percentage Error (MdAPE) and prediction interval coverage are also reported.
- **Split:** 80/20, seed 42. Similar listings are grouped to prevent leakage between training and test sets.
- **Dataset A:** Pakistan Used Cars Dataset (Kaggle, muhamedumarjamil).
- **Dataset B:** Additional used-car listings supplied in `usedCars.json`.
- **Combined training:** Dataset A and Dataset B can be combined. Dataset B records receive a lower sample weight (0.3) during combined model training.
- **Evaluation:** The planned A-only, B-only and A+B comparison uses a common held-out Dataset A test set.
- **Sample data:** `data/sample/sample_listings.csv` contains synthetic data for testing the pipeline. It is not real listings.
- Raw datasets are not committed to the repository.

## Data preparation

### Dataset A preparation (`src/prepare_data.py`)

Dataset A is processed using the column mappings in `configs/schema.yaml`.

The preparation pipeline:

1. Loads the raw CSV file.
2. Maps the original column names to the standard CarCompass schema.
3. Cleans and validates prices, mileage, year and other fields.
4. Removes duplicates and invalid records.
5. Groups similar listings to help prevent data leakage.
6. Creates training and test datasets.

### Dataset B preparation (`src/prepare_b.py`)

Dataset B is prepared separately using `usedCars.json` and `configs/schema_b.yaml`.

The preparation pipeline:

1. Flattens the JSON records into a tabular dataset.
2. Combines manufacturer and model information into the standard make/model fields.
3. Maps Dataset B columns to the common CarCompass schema.
4. Validates prices, mileage, year and other fields.
5. Removes duplicate and rare-model listings and applies optional low-mileage filtering.
6. Creates grouped 80/20 training and test splits.

Run Dataset B preparation:

```bash
python -m src.prepare_b \
  --input data/raw/usedCars.json \
  --schema configs/schema_b.yaml \
  --min-model-count 5 \
  --min-city-count 20 \
  --check-low-mileage
```

Output is saved in `data/processed_b/`, including training data, test data and quality reports.

## Experiments and best run

### Dataset A-only results

| Run | MAE (PKR) | MdAPE | Interval coverage |
|---|---:|---:|---:|
| LightGBM (`lgbm_900_lr02`) | 352,511 | 6.2% | 80% |

### Dataset B-only results

| Run | MAE (PKR) | MdAPE | Interval coverage |
|---|---:|---:|---:|
| Random Forest (`rf_300_depth20`) | 197,710 | 5.8% | 80% |

Dataset A and Dataset B results above were evaluated on separate test sets. Their MAE values should not be interpreted as a direct model comparison.

**Dataset B best run:** Random Forest with 300 trees and maximum depth 20 achieved the lowest MAE among the tested Dataset B configurations.

The combined Dataset A+B model will be evaluated separately using the same held-out Dataset A test set. This will help determine whether adding Dataset B improves predictions on Dataset A listings.

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Returns estimated price, price range, assumed fields and model version |
| `POST /explain` | Returns the prediction and the model's top factors |
| `GET /health` | Liveness check |
| `GET /metrics` | Request count, average latency and errors |
| `GET /docs` | Auto-generated API documentation |

Optional inputs (`engine_cc`, `transmission`, `fuel`, `city`) narrow the prediction range. If they are missing, typical values for that make/model are used, flagged in `assumed_fields`, and the range widens.

## Tests and lint

```bash
make test    # pytest: data preparation, model loading, API endpoints
make lint    # ruff
```

## Docker

```bash
make train
make docker-build TAG=v1.0
make docker-run TAG=v1.0
```

The API will be available at http://localhost:8000.

`models/*.joblib` is git-ignored. Before building the image you deploy, make sure the champion model is in `models/` (see `RUNBOOK.md`).

## Project status

- [x] Milestone 1 skeleton: preparation, training, MLflow, API, tests and CI.
- [x] Dataset A preparation and initial model training.
- [x] Dataset B preparation and initial model training.
- [ ] Complete fair Dataset A vs Dataset B vs Dataset A+B comparison.
- [ ] Finalise model selection and best-run justification.
- [ ] Milestone 2: deploy, `RUNBOOK.md`, and cURL examples against the live URL.
- [ ] Milestone 2: vision models, `/classify-image`, and MLflow Model Registry export.

## Repository layout

```text
src/              prepare_data.py, prepare_b.py, train.py, model_utils.py, app.py
tests/            pytest files
configs/          schema.yaml, schema_b.yaml, train.yaml
data/sample/      small synthetic sample only
data/raw/         original datasets (not committed)
data/processed/   Dataset A and combined processed data (git-ignored)
data/processed_b/ Dataset B processed data (git-ignored)
models/           trained artifacts (git-ignored)
models_b/         Dataset B model artifacts (git-ignored)
.github/          CI workflow, PR template
```

## Credits

- **Dataset A:** *Pakistan Used Cars Dataset*, Kaggle (muhamedumarjamil). Add the applicable licence.
- **Dataset B:** `usedCars.json`. Add the original source and applicable licence.
- **Libraries:** FastAPI, scikit-learn, MLflow, LightGBM and pandas. Cite any template or code you reuse.
- **AI assistance:** See `AI_USAGE.md`.