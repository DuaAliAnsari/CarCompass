# CarCompass system design

CarCompass estimates a used car's asking price in PKR. The running service is a
static site and a FastAPI process. Training happens offline and is not part of
a request.

## Request path

```
Browser page
    │  POST /predict or /explain
    ▼
FastAPI  src/app.py
    │  Pydantic checks make, model, year, mileage
    │  and the optional fields
    ▼
ModelRouter  src/model_router.py
    │  year is 2010 or later, and the specialist is on
    │      → price_new_cars_2010plus
    │  otherwise
    │      → price_global_ab
    ▼
predict_price  src/model_utils.py
    │  missing optional fields are filled from that
    │  make and model, and the range widens 10% per field
    ▼
JSON: price, 80% range, assumed fields, model version
      /explain also returns the top five factors
```

The factors are the model's overall feature importance, not an explanation of
one car.

## Offline path

```
Dataset A (CSV) ─┐
                 ├─ src/prepare_multi.py ─ one fit / val / cal / test split
Dataset B (JSON) ─┘
                          │
                          ▼
                 src/train_families.py
                 5 families × 12 runs
                 logged to local MLflow (sqlite mlflow.db)
                          │
                          ▼
                 champion per family, chosen on validation MAE
                 only models of at most 100 MB can be served
                          │
                          ▼
                 models/*.joblib + models/manifest.json
```

The shared set is 97,786 listings. The split is fit 64%, validation 8%,
calibration 8%, test 20%, seed 42. A listing group stays on one side of the
split. Dataset B prices are scaled to Dataset A's level using a factor fit on
training rows only.

## What is served, and why

| Choice | Reason |
|---|---|
| `lgbm_900_lr02` as the global champion | Lowest validation MAE among models under 100 MB (333,743 PKR). The larger random forest is slightly better and is about 763 MB. |
| `price_global_ab` as the default | Lowest validation MAE of the three full-data candidates. |
| 2010+ specialist switched on | 1.4% better than the default on 2010+ validation cars. |
| Pre-2010 specialist switched off | 1.2% worse than the default on its own validation cars. |
| Residual quantiles for the range | Covers 80% of test prices. |

On the same 19,525 test rows the routed system has MAE 360,636 PKR, against
370,313 for the global model alone.

## Where things live

| Thing | Place | In git? |
|---|---|---|
| Pages, CSS, and page scripts | `src/static/` | yes |
| API and training code | `src/` | yes |
| Column maps and experiment list | `configs/` | yes |
| This design note | `design/` | yes |
| Raw listings | `data/raw/` | no |
| Trained models | `models/` | no, except `models/.gitkeep` |
| MLflow history | `mlflow.db` | no |
| Request counters | process memory | no, they reset on restart |

`GET /metrics` is the monitoring the course asks for: request count, error
count, and average latency. Each request is also one JSON log line.

## Pages and API

The browser never talks to a database. These are the pages:

| Page | Route | What it uses |
|---|---|---|
| Estimate | `/` | `POST /explain`, `GET /options`, `GET /health` |
| Experiments | `/experiments` | the recorded comparison, no live call |
| Models | `/system` | `GET /models`, `GET /health` |
| Pipeline | `/pipeline` | this design, drawn as the stages above |
| Demo | `/demo` | `/health`, `/predict`, `/explain`, `/metrics`, `/docs` |

`GET /models` stays the JSON API. The models page is `/system` so the two do
not collide.

## Deploy shape

The Dockerfile runs `uvicorn src.app:app` on `0.0.0.0` and `$PORT`, and its
healthcheck calls `/health`. A Render web service can use that image. The
image only answers with a price when the joblib files are present at build
time. On `main` those files are gitignored, so a build of `main` starts and
`/predict` returns 503. The branch `infra/docker-models` commits the two
served models for that image.

There is no Postgres in this design. MLflow's sqlite file is for training.
