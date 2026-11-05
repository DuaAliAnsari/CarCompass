# CarCompass: Used-Car Price Estimation for Pakistan

CarCompass takes a car's make, model, year and mileage and returns an estimated
price in PKR with a range and the top factors behind it, served through a FastAPI
service that is tracked in MLflow, containerised with Docker, tested in CI and deployed to the cloud.

> Estimates reflect historical asking prices. They are a guide, not a valuation.

## Architecture

```
 Dataset A (CSV) ─┐     src/prepare_multi.py           src/train_families.py              src/app.py
                  ├──►  map, validate, dedup A∪B  ──►  5 families × 12 runs (MLflow)  ──►  FastAPI + router
 Dataset B (JSON)─┘     one fit/val/cal/test split     champion per family (val MAE)       /predict /explain
                        data/processed_multi/          models/manifest.json + *.joblib     /models /health /metrics
                                                                    │
                                                          Docker image ──► Render / Railway
```

The router sends each request to the model that won on validation rows for that car's
year: the 2010+ specialist for cars from 2010 on, the global A+B model for the rest.

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

### Training the five model families (Dataset A + Dataset B)

Save Dataset A as `data/raw/pakistan_used_cars.csv` and Dataset B as
`data/raw/usedCars.json`, then:

```bash
make train-multi       # builds the shared dataset, trains 5 families x 12 runs (about 1-2 h)
make run               # the API now serves the models listed in models/manifest.json
```

The big random forests need a lot of memory. If training is stopped, run
`python -m src.train_families --resume`: runs already logged in MLflow are skipped.

### Training on Dataset A only (Milestone 1)

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

## Multi-model setup: five price families

Design: `Multi-model/CarCompass_Proposed_Multi_Model_Architecture.md`. A *family* says
which rows a model is trained on (`configs/model_families.yaml`); every family runs all
12 experiments in `configs/train.yaml`.

| Family | Trained on | Role |
|---|---|---|
| `price_global_ab` | Dataset A + Dataset B (B rows weighted 0.3) | default candidate |
| `price_dataset_a` | Dataset A only | default candidate / benchmark |
| `price_dataset_b` | Dataset B only | default candidate / benchmark |
| `price_new_cars_2010plus` | A + B, cars from 2010 on | specialist |
| `price_old_cars_pre2010` | A + B, cars before 2010 | specialist |

### Shared dataset (`src/prepare_multi.py`)

1. Both files are mapped to one schema (`configs/schema.yaml`, `configs/schema_b.yaml`)
   and cleaned with the rules below. Dataset B's `RegisteredIn` has the same meaning as
   Dataset A's `Province`.
2. Duplicates are removed inside each source and then **across sources**: 8,558 B
   listings were the same car as an A listing (same make, model, year, mileage, city),
   so only the A copy is kept.
3. One group-aware split (seed 42) for all families: **fit** 64% (training),
   **val** 8% (choosing champions and routing), **cal** 8% (price range),
   **test** 20% (final scores only). A listing group is always on one side.
4. Dataset B was scraped in August 2021, so its prices are scaled to Dataset A's level
   (factor 1.42, measured on fit rows only).

Result: 97,786 listings (A 58,553, B 39,233). Details: `reports/data_quality_multi.md`.

### How champions and routing are chosen

- **Champion of a family:** lowest **validation** MAE among models of at most 100 MB
  (`max_model_mb`). Test rows are never used to choose. The large random forests
  (`rf_300_depth20`, 350-760 MB each) are trained and logged but cannot be served on a
  small hosting plan; LightGBM is within 3% of them at about 12 MB.
- **Default model:** the candidate with the lowest validation MAE on all validation
  rows (A and B).
- **Specialist:** switched on only if it beats the default by at least 1% on the
  validation rows of its own years (`configs/routing.yaml`).
- Every champion is registered in the MLflow Model Registry
  (`carcompass-price-global-ab`, ...) with alias `champion`; the served models also get
  alias `production`.

### Results

All runs of `price_global_ab` (validation MAE in PKR, 7,865 validation rows):

| Run | Val MAE | Val MdAPE | Model size |
|---|---|---|---|
| baseline_median | 2,304,820 | 48.7% | — |
| baseline_mean | 2,300,997 | 48.2% | — |
| baseline_q40 | 2,316,177 | 49.8% | — |
| ridge_alpha_01 | 1,145,734 | 22.0% | — |
| ridge_alpha_1 | 1,145,734 | 22.0% | — |
| ridge_alpha_10 | 1,145,733 | 22.0% | — |
| rf_100_depth12 | 399,073 | 7.4% | — |
| rf_200_depth16 | 340,613 | 6.4% | — |
| rf_300_depth20 | 324,644 | 6.2% | 763 MB (over the limit) |
| lgbm_300 | 381,185 | 7.1% | — |
| lgbm_600_lr03 | 352,052 | 6.5% | — |
| **lgbm_900_lr02 (champion)** | **333,743** | **6.3%** | **12.6 MB** |

Family champions, each scored on its own test rows:

| Family | Champion | Val MAE | Test MAE | Test MdAPE | Range coverage |
|---|---|---|---|---|---|
| `price_global_ab` | `lgbm_900_lr02` | 333,743 | 370,313 | 6.5% | 80% |
| `price_dataset_a` | `lgbm_900_lr02` | 347,817 | 387,479 | 6.1% | 80% |
| `price_dataset_b` | `lgbm_900_lr02` | 292,036 | 319,170 | 5.8% | 80% |
| `price_new_cars_2010plus` | `lgbm_900_lr02` | 376,902 | 413,633 | 5.1% | 80% |
| `price_old_cars_pre2010` | `lgbm_600_lr03` | 240,116 | 263,931 | 9.9% | 80% |

**Fair comparison: every model on the same 19,525 test rows** (MAE in PKR):

| Model | All test | A rows | B rows | Pre-2010 | 2010+ |
|---|---|---|---|---|---|
| `price_global_ab` | 370,313 | 389,722 | 342,167 | 257,490 | 428,281 |
| `price_dataset_a` | 413,223 | 387,479 | 450,554 | 271,935 | 485,816 |
| `price_dataset_b` | 636,365 | 855,102 | 319,170 | 274,047 | 822,524 |
| `price_new_cars_2010plus` | | | | | 413,633 |
| `price_old_cars_pre2010` | | | | 263,931 | |
| **served system** (global + 2010+ specialist) | **360,636** | **376,881** | **337,079** | 257,490 | 413,633 |

What this shows:

- **Adding Dataset B helps.** On all test rows the A+B model (370,313) beats the A-only
  model (413,223) by about 10%, and it is almost as good as the A-only model on A's own
  rows (389,722 vs 387,479). The B-only model does not carry over to A's listings
  (855,102 on A rows).
- **Routing (validation rows):** `price_global_ab` is the default (val MAE 333,743 vs
  379,527 for A only and 560,441 for B only). The **2010+ specialist is switched on**
  (+1.4% better than the default on 2010+ validation cars). The **pre-2010 specialist
  is not** (1.2% worse than the default on pre-2010 cars), so older cars go to the
  global model. On Dataset A alone (Milestone 1) neither era model helped; with the
  extra B rows the 2010+ one now does.
- The served system's test MAE is **360,636 PKR**, 2.6% better than the global model alone.
- **Assembly** (Local / Imported) is a new input from both datasets. It helps a little:
  val MAE 333,743 with it, 334,935 without.

Full report: `reports/model_comparison.md`. Served models: `models/manifest.json`.

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

## Milestone 1 results: Dataset A only (`src/train.py`)

`configs/train.yaml` lists the runs (median baseline, Ridge, Random Forest, LightGBM).
Each run logs its parameters, MAE, median % error, interval coverage, the data hash and the git commit.
The champion is the run with the lowest **validation** MAE (the training rows are split
into fit 70%, val 15% and cal 15%), saved as `models/price_model_<version>.joblib`.
The table below was produced before this change, when the lowest test MAE was used.

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
| `POST /predict` | price, range, assumed fields, model version, `selected_model` |
| `POST /explain` | same plus the model's top factors |
| `GET /models` | the models being served, their years, versions and test MAE |
| `GET /health` | liveness check, models served |
| `GET /metrics` | request count, average latency, errors |
| `GET /docs` | auto-generated API docs |

Optional inputs (`engine_cc`, `transmission`, `fuel`, `city`, `assembly`) narrow the range. If they are
missing, typical values for that make/model are used, flagged in `assumed_fields`, and the range widens.

## Tests and lint

```bash
make test    # pytest: data preparation, A+B dedup and split, router, model loading, API
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
- [x] Multi-model: shared A+B dataset, five families, validation-based champions,
      routing manifest, `/models`, MLflow registry per family
- [ ] Milestone 2: deploy, `RUNBOOK.md`, cURL examples against the live URL
- [ ] Milestone 2: vision models, `/classify-image`, MLflow Model Registry export

## Repository layout

```
src/            prepare_data.py, prepare_multi.py, train.py, train_families.py,
                model_router.py, model_utils.py, app.py
tests/          pytest files
configs/        schema.yaml / schema_b.yaml (column mapping), train.yaml (experiments),
                model_families.yaml (families), routing.yaml (routing rules)
reports/        data quality reports, model_comparison.md
data/sample/    small synthetic sample only
models/         trained artifacts (git-ignored)
.github/        CI workflow, PR template
```

## Credits

- Dataset A: _Pakistan Used Cars Dataset_, Kaggle (muhamedumarjamil), https://www.kaggle.com/datasets/muhamedumarjamil/pakistan-used-cars-dataset. Licence: check the "License" field on the Kaggle page and write it here.
- Dataset B: _Pakistan's Largest PakWheels Automobiles Listings_, Kaggle (asimzahid), https://www.kaggle.com/datasets/asimzahid/pakistans-largest-pakwheels-automobiles-listings. Licence: check the Kaggle page and write it here.
- Libraries: FastAPI, scikit-learn, MLflow, LightGBM, pandas (cite any template or code you reuse).
- AI assistance: see `AI_USAGE.md`.
