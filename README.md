# CarCompass: Used-Car Price Estimation for Pakistan

CarCompass takes a car's make, model, year and mileage and returns an estimated
price in PKR with a range and the top factors behind it, served through a FastAPI
service that is tracked in MLflow, containerised with Docker, tested in CI and deployed to the cloud.

> Estimates reflect historical asking prices. They are a guide, not a valuation.

## Architecture

```
 data/raw (PakWheels)        src/prepare_data.py          src/train.py                 src/app.py
 ┌───────────────┐  ──►  validate, dedup, split  ──►  train + MLflow runs  ──►  FastAPI /predict /explain
 └───────────────┘        data/processed/*.csv         models/price_model.joblib     /health /metrics /docs
                                                                    │
                                                          Docker image ──► Render / Railway
```

## Quick start

Requires Python 3.11 or newer (developed on 3.13) and `make`. On Windows, use WSL or run the
`python -m ...` commands below directly.

```bash
git clone <your-repo-url> && cd carcompass
make setup      # creates .venv and installs dependencies
make train      # DEMO: prepares the synthetic sample and trains on it (overwrites data/processed and models/)
make run        # API at http://localhost:8000  (docs at /docs)
```

Try it:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Corolla","year":2016,"mileage":75000}'
```

View experiments: `make mlflow-ui`, then open http://localhost:5000.

### Reproduce the results on the real PakWheels data

`make train` is a quick demo on the small synthetic sample. To reproduce the numbers in this README:

1. Download the Kaggle dataset (see **Dataset and task**) and save it as
   `data/raw/pakistan_used_cars.csv`.
2. Run:

```bash
make train-real    # prepare (seed 42, same split for everyone), then train; about 5 to 10 minutes
```

Without `make`, the same two steps are:

```bash
python -m src.prepare_data --input data/raw/pakistan_used_cars.csv \
  --min-model-count 5 --min-city-count 20 --check-low-mileage
python -m src.train
```

This writes `data/processed/train.csv` and `test.csv`, a data-quality report to
`reports/data_quality.md`, and the champion model plus `VERSION.json` to `models/`.
The champion is registered in the MLflow Model Registry with the alias `production`
(this needs MLflow's default database tracking store, or `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`).

## Dataset and task

- **Task:** regression, predict used-car price (PKR), trained on log-price.
- **Metric:** MAE in PKR on a held-out 20% test set (also reported: median absolute
  percentage error and interval coverage of the 10th-90th percentile range).
- **Split:** 80/20, seed 42. Rows get a `listing_group` id (make, model, year, mileage, city), and
  `GroupShuffleSplit` keeps every group on one side, so copies of one listing never appear in both train and test.
- **Included sample:** `data/sample/sample_listings.csv` is **synthetic** data made so
  the pipeline and tests run out of the box. It is not real listings.
- **Real data:** Dataset A, *Pakistan Used Cars Dataset* (Kaggle, muhamedumarjamil, PakWheels listings),
  is downloaded, not committed. `configs/schema.yaml` already maps its columns (`nam`, `Price`, `Year`,
  `Millage`, `Fuel`, `Transmission`, `Province`, `Engine Capacity`). The Kaggle file is called
  `PakWheels_Dataset.csv`; save it as `data/raw/pakistan_used_cars.csv`.
- **Known limitation:** the dataset has no city column, only `Province`, which mixes provinces,
  cities and a registration status ("Un-Registered"). It is used as `city`, so location is coarse.
- Licences and credits for every dataset used go in the **Credits** section below.

## Data preparation (`src/prepare_data.py`)

1. Map raw columns to one schema, parse values like `27.4 lacs` into PKR and `120,000 km` into numbers,
   and split the combined listing name (`Suzuki Alto VXL AGS 2022`) into make and model.
2. Validation: drop rows with missing required fields or impossible price, mileage or year.
3. Remove duplicate listings (make, model, year, mileage, price, city).
4. Optional quality rules: drop cars 3 or more years old with under 1,000 km (probable odometer error),
   drop make/model pairs with fewer than 5 listings, and group cities with fewer than 20 listings as "Other".
5. Add `listing_group`, check the result with a Pandera schema, split 80/20 (seed 42), and write
   `data/processed/` plus `quality_report.json` and `reports/data_quality.md`.

Result on the full PakWheels file:

| Step | Rows |
|---|---|
| Rows read | 61,919 |
| Dropped: missing required values | 853 |
| Dropped: out of range | 140 |
| Dropped: duplicates | 826 |
| Dropped: low mileage | 1,308 |
| Dropped: rare make/model | 411 |
| **Rows kept** | **58,381** |
| Train / test | 46,680 / 11,701 |

## Experiments and best run

`configs/train.yaml` lists 12 runs, three per model type (median/mean/quantile baseline, Ridge, Random Forest, LightGBM).
Each run logs its parameters, MAE, median % error, interval coverage, the data hash and the git commit.
The lowest test MAE becomes the champion, is tagged `champion` in MLflow and is saved as
`models/price_model_<version>.joblib` (the API loads `models/price_model.joblib`), with
`models/VERSION.json` describing the run. The champion is also registered in the MLflow Model Registry
with the alias `production`.

Test set: 11,701 cars. MdAPE is the median absolute percentage error. Coverage is the share of real prices inside the predicted range (target about 80%).

| Run | MAE (PKR) | MdAPE | Coverage |
|---|---|---|---|
| baseline_median | 2,369,529 | 47.6% | 80% |
| baseline_mean | 2,370,318 | 48.4% | 80% |
| baseline_q40 | 2,411,599 | 48.7% | 80% |
| ridge_alpha_01 | 1,285,310 | 22.4% | 79% |
| ridge_alpha_1 | 1,285,310 | 22.4% | 79% |
| ridge_alpha_10 | 1,285,309 | 22.4% | 79% |
| rf_100_depth12 | 443,875 | 7.8% | 79% |
| rf_200_depth16 | 377,850 | 6.6% | 80% |
| rf_300_depth20 | 363,198 | 6.3% | 80% |
| lgbm_300 | 402,657 | 7.0% | 80% |
| lgbm_600_lr03 | 358,351 | 6.4% | 80% |
| **lgbm_900_lr02 (champion)** | **352,511** | **6.2%** | **80%** |

**Best run and why:** `lgbm_900_lr02`, a LightGBM model with 900 trees, learning rate 0.02 and 127 leaves.
It has the lowest test MAE (about 352,500 PKR) and the lowest median error (6.2%). That is about
85% lower MAE than the median baseline and about 73% lower than Ridge, because boosted trees can learn
interactions that a single linear formula cannot, such as one model losing value differently from another
as it ages. Its 10th-90th percentile range covers 80% of real prices, matching the target. The champion
is close to the best random forest (363,198 PKR), so the gap is modest. LightGBM is also the model
the `/explain` endpoint needs, because it can report per-feature contributions.

**Era models (checked, not used):** separate models for cars before 2010 and 2010 and later were trained
with the champion's settings. Together they reached an MAE of 356,432 PKR, slightly worse than the single
global model (352,511 PKR), so the API uses one global model. The 2010+ model alone is more accurate in
percentage terms (4.9% median error) than the pre-2010 model (10.7%).

**Price range:** the range comes from the 10th and 90th percentile of the champion's errors on a held-out
calibration split. A second method, LightGBM quantile models, gave 78% coverage against 80%, so the
simpler residual method was kept.

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

Without `make`: `pytest` and `ruff check .`. Older MLflow versions needed no setup; newer ones block the
file store, so `tests/conftest.py` sets `MLFLOW_ALLOW_FILE_STORE=true` for the test run.

## Docker

```bash
make train                 # the image copies models/, so train first
make docker-build TAG=v1.0
make docker-run TAG=v1.0   # http://localhost:8000
```

`models/*.joblib` is git-ignored. Before building the image you deploy, make sure the
champion model is in `models/` (see `RUNBOOK.md`).

## Project status

- [x] Milestone 1: prepare data, train + MLflow runs, API, tests, CI
- [x] Milestone 1: trained on real Dataset A, results table and best-run note filled in
- [ ] Milestone 2: deploy, `RUNBOOK.md`, cURL examples against the live URL
- [ ] Milestone 2: vision models, `/classify-image`, MLflow Model Registry export

## Repository layout

```
src/            prepare_data.py, train.py, model_utils.py, app.py
tests/          pytest files
configs/        schema.yaml (column mapping), train.yaml (experiments)
reports/        data_quality.md (generated by prepare_data)
data/sample/    small synthetic sample only
models/         trained artifacts (git-ignored)
.github/        CI workflow, PR template
```

## Credits

- Dataset A: _Pakistan Used Cars Dataset_, Kaggle (muhamedumarjamil), PakWheels listings. Add licence here.
- Libraries: FastAPI, scikit-learn, MLflow, LightGBM, pandas, Pandera (cite any template or code you reuse).
- AI assistance: see `AI_USAGE.md`.