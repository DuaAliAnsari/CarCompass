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

### Training on the real dataset

`make train` uses the small synthetic sample. For the real results:

```bash
python scripts/download_data.py          # or place the CSV at data/raw/pakistan_used_cars.csv
make train-real                          # prepares real data, trains 12 runs, registers @production
make mlflow-ui                           # inspect runs at http://localhost:5000
```

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

1. Map raw columns to one schema (`configs/schema.yaml`) and parse values like `27.4 lacs` into PKR.
2. Validation: drop rows with missing required fields or values outside these ranges:
   price 50,000 to 200,000,000 PKR, mileage 0 to 1,000,000 km, year 1980 to next year.
   An engine size outside 50 to 8000 cc is set to missing instead.
3. Remove duplicate listings (make, model, year, mileage, price, city).
4. Extra rules used for the real data (`make prepare-real`): drop cars 3+ years old with
   under 1,000 km, drop make/model pairs with fewer than 5 listings, group cities with
   fewer than 20 listings as `Other`.
5. Pandera schema check, then an 80/20 split (seed 42) that keeps near-copies of one
   listing on the same side. Writes `data/processed/` and `reports/data_quality.md`.

All limits live in `src/prepare_data.py`.

## Using two price datasets

Besides Dataset A, the 2022 PakWheels scrape (about 90,000 listings, Sep to Oct 2022)
can be mixed in. Save it as `data/raw/pakwheels_2022.csv`. Each file becomes one
`source`, so results can be compared per dataset.

```bash
make train-two      # prepare both files, then train into models_merged/ (not registered)
```

On Windows without `make`:

```
python -m src.prepare_data --input data/raw/pakistan_used_cars.csv data/raw/pakwheels_2022.csv --out data/processed_merged --report reports/data_quality_merged.md --min-model-count 5 --min-city-count 20 --check-low-mileage --rescale-prices --log-mlflow --tracking-uri sqlite:///mlflow.db
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python -m src.train --train data/processed_merged/train.csv --test data/processed_merged/test.csv --out models_merged --experiment carcompass-price-merged --no-register
```

How the second file is handled:
- Its columns are mapped to the same schema (`configs/schema.yaml`). Its
  `registered_in` column matches Dataset A's `Province` column (where the car is
  registered, e.g. Lahore, Karachi, Un-Registered).
- Both sources are cleaned together with the same rules. A listing present in both
  with the same make, model, year, mileage, price and city is kept once, and
  copies of one listing stay on the same side of the train/test split.
- `reports/data_quality_merged.md` lists rows per source and a **price agreement**
  check: for make/model/year groups with at least 3 listings in both sources, the
  ratio of their median prices. Prices move over time, so a ratio far from 1 means
  the sources should not be mixed blindly.
- Training prints and logs the champion's test MAE for each source.

**Different scrape dates.** The 2022 file was scraped in Sep to Oct 2022 and Dataset A
in 2025 (its newest model year is 2025). On the same make/model/year, the 2022 prices are
about 19% lower (median ratio 0.81, only 33% of groups within 15%), so the raw sources
do not agree. `--rescale-prices` (used by `make prepare-two`) multiplies the 2022 prices
by one factor (about 1.23, measured on train rows only) to bring them to Dataset A's
level; afterwards 79% of groups are within 15%. One factor fixes an overall level shift
but not differences that vary by car or segment.

**Decision rule (from the proposal):** keep the merged data only if the sources
agree (after rescaling) and the champion's MAE on Dataset A's test rows is no worse than
the A-only run.
The two test sets are not identical (the split changes when rows are added), so treat
the comparison as approximate. If merging does not help, the A-only model stays.

## Experiments and best run

`configs/train.yaml` lists the runs (median baseline, Ridge, Random Forest, LightGBM).
Each run logs its parameters, MAE, median % error, interval coverage, the data hash and the git commit.
The lowest test MAE becomes the champion and is saved as `models/price_model_<version>.joblib`.

Results on the held-out test set (11,701 listings, seed 42). Coverage is the share of true
prices inside the 10th-90th percentile range; a well-calibrated range gives about 80%.

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

**Best run and why:** `lgbm_900_lr02` has the lowest MAE (PKR 352,511) and the lowest median
percentage error (6.2%), so it is the champion and is registered in MLflow as
`carcompass-price` with the alias `production`. Its MAE is about 85% lower than the median
baseline (2.37 million) and about 73% lower than the best Ridge model (1.29 million). Ridge
barely changes with its regularisation strength, which suggests the linear form is the
limit, not the tuning. The tree models capture the non-linear way price depends on make,
model, year and mileage. The three best models are close (Random Forest 300 trees: 363,198;
LightGBM 600 trees: 358,351), so the gap to the second place is small (about 1.6%).

**Global vs era-specific models.** Separate models for pre-2010 and 2010-onward cars were
tested. Pre-2010 MAE was 267,579 (MdAPE 10.7%) and 2010+ MAE was 396,651 (MdAPE 4.9%).
Combined, the era models reach MAE 356,432, which is slightly worse than the global model
(352,511), so **we keep the single global model**. The era split is not worth the extra
complexity: it needs two models in memory and a routing step in the API, and it brings no
accuracy gain. Older cars are harder in percentage terms (10.7% vs 4.9%), but their
absolute errors are smaller because they are cheaper.

**Price range.** The 80% range comes from residual quantiles on a held-out calibration set.
It covers 80% of test prices. LightGBM quantile models were also checked and covered 78%,
so the residual method was kept.

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
- [x] Milestone 1: structured JSON request logs
- [x] Milestone 1: trained on real Dataset A, results table and best-run note filled in
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

- Dataset A: _Pakistan Used Cars Dataset_, Kaggle (muhamedumarjamil), https://www.kaggle.com/datasets/muhamedumarjamil/pakistan-used-cars-dataset. Licence: check the "License" field on the Kaggle page and write it here.
- Libraries: FastAPI, scikit-learn, MLflow, LightGBM, pandas (cite any template or code you reuse).
- AI assistance: see `AI_USAGE.md`.
