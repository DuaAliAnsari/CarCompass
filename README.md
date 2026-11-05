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

Requires Python 3.11. The complete M1 training workflow uses both real datasets and
requires a Kaggle API token (or the two raw files downloaded manually).

```bash
git clone https://github.com/DuaAliAnsari/CarCompass.git && cd CarCompass
make setup
python scripts/download_data.py   # downloads Dataset A and Dataset B into data/raw/
make train-multi                  # prepares the shared data and trains all five families
make run                          # API at http://localhost:8000 (docs at /docs)
```

Try it:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Corolla","year":2016,"mileage":75000}'
```

View the experiments in MLflow with `make mlflow-ui`, then open
http://localhost:5000. The multi-family training run is computationally intensive
(approximately 1–2 hours on the project setup); the large random forests also require
substantial memory. If training is interrupted, resume with
`python -m src.train_families --resume`.

### Quick pipeline check with the included sample

`make train` prepares the small **synthetic** CSV in `data/sample/` and runs the original
single-dataset training script. This is useful for checking the basic preparation and
training workflow, but it is **not** the full M1 experiment and its results must not be
reported as results on the real A+B data.

### Dataset A-only benchmark

The original single-source experiment can still be reproduced separately:

```bash
python scripts/download_data.py   # or place Dataset A at data/raw/pakistan_used_cars.csv
make train-real                   # prepares Dataset A and trains the single-source runs
make mlflow-ui
```

This is retained as a Dataset A benchmark for comparison with the complete A+B system;
it is not a separate milestone or the full project pipeline.

## Dataset and task

- **Task:** regression—estimate a used car's asking price in PKR, with models trained
  on log-price and predictions converted back to PKR.
- **Data used by the complete M1 system:** Dataset A and Dataset B, cleaned, mapped to
  a common schema, and deduplicated across sources before model training.
- **Evaluation:** MAE in PKR is the primary metric. MdAPE and coverage of the calibrated
  10th–90th percentile prediction range are also reported.
- **M1 split:** one group-aware split shared by all model families: fit 64%, validation
  8%, calibration 8%, and held-out test 20% (seed 42). Validation selects champions
  and routing; the test partition is reserved for final evaluation.
- **Included sample:** `data/sample/sample_listings.csv` is synthetic and is provided
  for a lightweight pipeline check and tests. It is not used for the reported real-data
  A+B results.
- **Raw data:** both real datasets are downloaded and are not committed to this repo.
  Dataset A's mapping is in `configs/schema.yaml`; Dataset B's mapping is in
  `configs/schema_b.yaml`.

## Complete M1 modelling pipeline: five price families

A *family* says
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
   listings match an A listing on make, model, year, mileage and city, so only the A
   copy is kept (see the known limitation below).
3. One group-aware split (seed 42) for all families: **fit** 64% (training),
   **val** 8% (choosing champions and routing), **cal** 8% (price range),
   **test** 20% (final scores only). A listing group is always on one side.
4. Dataset B was scraped in August 2021, so its prices are scaled to Dataset A's level
   (factor 1.42, measured on fit rows only).

Result: 97,786 listings (A 58,553, B 39,233). Details: `reports/data_quality_multi.md`.

**Known limitation: the cross-source match is loose.** The key (make, model, year,
mileage, city) also matches different cars. Among the 8,558 matches only 78.5% have
the same transmission and 71.1% the same engine size, and 62% have a round mileage
such as 100,000 km. With a stricter key (same transmission, fuel, assembly and engine
within 50 cc) 5,489 rows still match. So some removed B rows were probably distinct
cars. The error is on the safe side: no test car can leak into training, so the test
scores below stay valid; the cost is less Dataset B training data. Planned for
Milestone 2: keep the B rows, give matching A and B rows the same `listing_group` (so
they stay on one side of the split), drop only strict-key duplicates, and compare MAE.

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
  global model. On the Dataset A-only benchmark neither era model helped; with the
  additional Dataset B rows the 2010+ specialist now does.
- The served system's test MAE is **360,636 PKR**, 2.6% better than the global model alone.
- **Assembly** (Local / Imported) is a new input from both datasets. It helps a little:
  val MAE 333,743 with it, 334,935 without.

Full report: `reports/model_comparison.md`. Served models: `models/manifest.json`.

## Data preparation scripts

The complete M1 A+B workflow is implemented in `src/prepare_multi.py`. It loads
Dataset B through `src/prepare_combined.py` and maps both sources using
`configs/schema.yaml` and `configs/schema_b.yaml`, validates and cleans the records,
removes duplicates within each source and across sources, and writes one shared dataset
for all five model families. Its group-aware fit/validation/calibration/test split is
shared across families so model comparisons are fair. Dataset B's older scraped prices
are adjusted to Dataset A's price level using a factor estimated from fit rows only.

`src/prepare_data.py` holds the shared cleaning rules that `prepare_multi.py` reuses:
price/mileage/year validation, duplicate removal, optional low-mileage and
low-frequency filtering, and the Pandera schema check. Run on its own, it is also the
Dataset A-only preparation path used by the single-source benchmark and the synthetic
sample workflow.

## Dataset A-only benchmark (earlier single-source experiment)

The following section documents the original single-source experiment implemented in
`src/train.py`. It is retained to show how the Dataset A baseline compares with the
complete M1 A+B system above. **It is not a separate M1 milestone, and this table is not
the headline result for the complete project.**

`configs/train.yaml` lists the baseline, Ridge, Random Forest, and LightGBM experiments.
Each run logs parameters, metrics, data hash, and Git commit to MLflow. The script
selects its champion using validation MAE. The table below contains historical held-out
test scores from an earlier run that selected the champion using test MAE; therefore,
these historical figures should be treated as a Dataset A benchmark, not mixed with the
A+B validation or test metrics above.

Results on the held-out Dataset A test set (11,701 listings, seed 42). Coverage is the
share of true prices inside the 10th–90th percentile range.

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

**Best run on Dataset A and why:** `lgbm_900_lr02` has the lowest MAE (PKR 352,511)
and the lowest median percentage error (6.2%). Its MAE is about 85% below the median
baseline (2.37 million) and about 73% below the best Ridge model (1.29 million). Ridge
barely changes with alpha, so the linear form is the limit, not the tuning; the tree
models capture the non-linear effect of make, model, year and mileage. This is the
Dataset A-only result; the A+B system's scores are in the five-family tables above.

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

Train the complete A+B model set first so the image contains the routed model artifacts
and `models/manifest.json`:

```bash
make train-multi
make docker-build TAG=v1.0
make docker-run TAG=v1.0   # http://localhost:8000
```

`models/*.joblib` and the generated manifest are local artifacts and are not committed.
Before building the image for deployment, make sure the required model artifacts are
present in `models/` (see `RUNBOOK.md`).

## Project status

- [x] **M1 data pipeline:** Dataset A and Dataset B ingestion, schema mapping,
      validation, within-source and cross-source deduplication, and group-aware splits
- [x] **M1 experiments:** five model families, 12 configurations per family,
      MLflow tracking, validation-based champion selection, and held-out test evaluation
- [x] **M1 serving:** FastAPI prediction/explanation endpoints and year-based model routing
- [x] **M1 quality and reproducibility:** automated tests, linting, CI workflow,
      data/Git metadata in experiment tracking, and structured request logging
- [ ] **M2:** complete deployment evidence, live URL/cURL examples, and operational
      monitoring/runbook validation (see `RUNBOOK.md`)

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

- Dataset A: _Pakistan Used Cars Dataset_, Kaggle (muhamedumarjamil), https://www.kaggle.com/datasets/muhamedumarjamil/pakistan-used-cars-dataset. Licence: CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International).
- Dataset B: _Pakistan's Largest PakWheels Automobiles Listings_, Kaggle (asimzahid), https://www.kaggle.com/datasets/asimzahid/pakistans-largest-pakwheels-automobiles-listings. Licence: CC BY-NC-SA 4.0. Both licences are non-commercial, so this project is for coursework only. The raw data is not committed to this repository.
- Libraries: FastAPI, scikit-learn, MLflow, LightGBM, pandas (cite any template or code you reuse).
- AI assistance: see `AI_USAGE.md`.
