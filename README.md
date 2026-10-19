# CarCompass: Used-Car Price Estimation for Pakistan

CarCompass takes a car's make, model, year and mileage and returns an estimated
price in PKR with a range and the top factors behind it, served through a FastAPI
service that is tracked in MLflow, containerised with Docker, tested in CI and deployed to the cloud.

> Estimates reflect historical asking prices. They are a guide, not a valuation.

## Architecture

```
 data/raw (Dataset A)        src/prepare_data.py          src/train.py                 src/app.py
 ┌───────────────┐  ──►  validate, dedup, split  ──►  train + MLflow runs  ──►  FastAPI /predict /explain
 └───────────────┘        data/processed/*.csv         models/price_model.joblib     /health /metrics /docs
                                                                    │
                                                          Docker image ──► Render / Railway
```

## Quick start

Requires Python 3.11 (Linux, macOS, or WSL on Windows) and `make`.

```bash
git clone <your-repo-url> && cd carcompass
make setup      # creates .venv and installs dependencies
make train      # prepares data, trains models, logs runs to MLflow, saves models/
make run        # API at http://localhost:8000  (docs at /docs)
```

Try it:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Corolla","year":2016,"mileage":75000}'
```

View experiments: `make mlflow-ui`, then open http://localhost:5000.

## Dataset and task

- **Task:** regression, predict used-car price (PKR), trained on log-price.
- **Metric:** MAE in PKR on a held-out 20% test set (also reported: median absolute
  percentage error and interval coverage of the 10th-90th percentile range).
- **Split:** 80/20, seed 42, near-copies of one listing stay on the same side.
- **Included sample:** `data/sample/sample_listings.csv` is **synthetic** data made so
  the pipeline and tests run out of the box. It is not real listings.
- **Real data:** Dataset A, *Pakistan Used Cars Dataset* (Kaggle, muhamedumarjamil),
  is downloaded, not committed. Edit `configs/schema.yaml` so the column names match.
- Licences and credits for every dataset used go in the **Credits** section below.

## Data preparation (`src/prepare_data.py`)

1. Map raw columns to one schema and parse values like `27.4 lacs` into PKR.
2. Validation: drop rows with missing required fields or impossible price, mileage, year.
3. Remove duplicate listings (make, model, year, mileage, price, city).
4. Split 80/20 (seed 42) and write `data/processed/` plus `quality_report.json`.

## Experiments and best run

`configs/train.yaml` lists the runs (median baseline, Ridge, Random Forest, LightGBM).
Each run logs its parameters, MAE, median % error, interval coverage, the data hash and the git commit.
The lowest test MAE becomes the champion and is saved as `models/price_model_<version>.joblib`.

| Run | MAE (PKR) | Notes |
|---|---|---|
| _fill in after training on the real data_ | | |

**Best run and why:** _write your justification here after training._

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | price, range, assumed fields, model version |
| `POST /explain` | same plus the model's top factors |
| `GET /health` | liveness check |
| `GET /metrics` | request count, average latency, errors |
| `GET /docs` | auto-generated API docs |

Optional inputs (`engine_cc`, `transmission`, `fuel`, `city`) narrow the range. If they are
missing, typical values for that make/model are used, flagged in `assumed_fields`, and the range widens.

## Tests and lint

```bash
make test    # pytest: data preparation, model loading, API endpoints
make lint    # ruff
```

## Docker

```bash
make train                 # the image copies models/, so train first
make docker-build TAG=v1.0
make docker-run TAG=v1.0   # http://localhost:8000
```

`models/*.joblib` is git-ignored. Before building the image you deploy, make sure the
champion model is in `models/` (see `RUNBOOK.md`).

## Project status

- [x] Milestone 1 skeleton: prepare, train + MLflow, API, tests, CI
- [ ] Milestone 1: train on real Dataset A, fill in results table and best-run note
- [ ] Milestone 2: deploy, `RUNBOOK.md`, cURL examples against the live URL
- [ ] Milestone 2: vision models, `/classify-image`, MLflow Model Registry export

## Repository layout

```
src/            prepare_data.py, train.py, model_utils.py, app.py
tests/          pytest files
configs/        schema.yaml (column mapping), train.yaml (experiments)
data/sample/    small synthetic sample only
models/         trained artifacts (git-ignored)
.github/        CI workflow, PR template
```

## Credits

- Dataset A: _Pakistan Used Cars Dataset_, Kaggle (muhamedumarjamil). Add licence here.
- Libraries: FastAPI, scikit-learn, MLflow, LightGBM, pandas (cite any template or code you reuse).
- AI assistance: see `AI_USAGE.md`.
