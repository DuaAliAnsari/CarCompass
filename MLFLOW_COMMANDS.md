# CarCompass: MLflow, training, validation and testing commands

Presentation cheat sheet. Every command below was run on this machine on 2026-11-05
unless marked _(not run)_. The numbers shown are what our local `mlflow.db` contains.

> **Windows note:** `make` is not installed here, and the Makefile uses `.venv/bin/python`
> (Linux/macOS). On Windows use the PowerShell commands below; the `make` target each
> one matches is shown next to it.

---

## 1. How we use MLflow (the whole picture)

| What | Where in code | MLflow feature |
|---|---|---|
| **Tracking store** | `sqlite:///mlflow.db` (git-ignored) + `mlruns/` for files | Backend store + artifact store |
| **Data versioning** | `src/prepare_data.py` `log_to_mlflow()` → experiment `carcompass-data` | One run per data version: row counts as metrics, train/test file hashes as params, `quality_report.json/.md` as artifacts |
| **Dataset A experiments** | `src/train.py` → experiment `carcompass-price` | 12 runs from `configs/train.yaml` (3 baselines, 3 Ridge, 3 Random Forest, 3 LightGBM) |
| **Extra runs (A only)** | `src/train.py` | `era_*` runs (pre-2010 vs 2010+), `era_vs_global_summary`, `lgbm_quantile_10_90` (quantile check); tag `model_scope` tells them apart |
| **Five model families (A+B)** | `src/train_families.py` → experiments `carcompass/<family>` | 12 runs per family on one shared split: `price_global_ab`, `price_dataset_a`, `price_dataset_b`, `price_new_cars_2010plus`, `price_old_cars_pre2010` |
| **Ablation** | `train_families.py` `assembly_ablation()` | Run `ablation_no_assembly_lgbm_900_lr02` (tag `ablation`) in `carcompass/price_global_ab`: same champion config without the Assembly input |
| **Weight-space programming** | `src/train_wsp.py` → experiment `carcompass/wsp` | MLP base + era specialists + soup / task arithmetic / TIES-lite; report in `reports/wsp_comparison.md` |
| **Champion selection** | lowest **validation** MAE among models ≤ 100 MB | Tag `champion=true` on the winning run; test metrics logged onto that run only |
| **Model Registry** | `register()` in both training scripts | Model logged with `mlflow.sklearn.log_model`, registered, aliases set |
| **Reproducibility** | every run | Param `data_hash`, tag `git_commit`, params of every model setting |
| **Resume** | `train_families.py --resume` | `mlflow.search_runs` finds runs already logged for the same data hash and skips them |

**What every family run logs:**
- **Params:** `family`, `model`, `param_*` (n_estimators, learning_rate, num_leaves, max_depth, alpha, strategy, quantile), `rows_fit`, `data_hash`
- **Metrics:** `val_mae_pkr`, `val_mdape`, `val_interval_coverage`, `q10`, `q90`, `model_mb`, `fit_seconds`; champion only: `test_mae_pkr`, `test_mdape`, `test_interval_coverage`, `test_rows`
- **Tags:** `family`, `git_commit`, `champion`

**Registered models in our local registry:**

| Registered model | Aliases |
|---|---|
| `carcompass-price-global-ab` | `champion` = v2, `production` = v2 (default model the API serves) |
| `carcompass-price-new-cars-2010plus` | `champion` = v1, `production` = v1 (2010+ specialist, switched on) |
| `carcompass-price-dataset-a` | `champion` = v2 |
| `carcompass-price-dataset-b` | `champion` = v2 |
| `carcompass-price-old-cars-pre2010` | `champion` = v1 (no `production`: routing switched it off) |
| `carcompass-price` | `production` = v1 (Dataset A-only script, see **section 9**) |

`production` = the models the API actually serves (`models/manifest.json`).

---

## 2. Open the MLflow UI

```powershell
cd D:\IBA\semester_6\MLOps\Project\CarCompass
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
_(make mlflow-ui)_. Open http://localhost:5000. Stop it with Ctrl+C.

---

## 3. What to show in the UI (demo order)

1. **Experiments list (left side):** `carcompass/price_global_ab`, `…/price_dataset_a`,
   `…/price_dataset_b`, `…/price_new_cars_2010plus`, `…/price_old_cars_pre2010`, `carcompass-price`.
2. **Open `carcompass/price_global_ab`:** add the columns `val_mae_pkr`, `model_mb` and
   the `champion` tag; sort by `val_mae_pkr`.
   - Point to make: `rf_300_depth20` has the lowest val MAE (324,644) but is **763 MB**,
     over the 100 MB limit, so `lgbm_900_lr02` (333,743, 12.6 MB) is the champion.
3. **Compare runs:** tick the 3 LightGBM runs (or one per model type) → **Compare** →
   parallel-coordinates plot of `param_n_estimators`, `param_learning_rate` vs `val_mae_pkr`.
4. **Open the champion run `lgbm_900_lr02`:** params, metrics (val + test), tags
   `git_commit` and `data_hash` (reproducibility), the logged model under Artifacts.
5. **Ablation:** in the same experiment, `ablation_no_assembly_lgbm_900_lr02` = 334,935
   vs 333,743 with Assembly, so Assembly helps slightly.
6. **Models tab:** the registered models above, with aliases `champion` and `production`.

**Champion per family (from our MLflow DB):**

| Family | Champion | Val MAE (PKR) | Test MAE (PKR) |
|---|---|---|---|
| price_global_ab | lgbm_900_lr02 | 333,743 | 370,313 |
| price_dataset_a | lgbm_900_lr02 | 347,817 | 387,479 |
| price_dataset_b | lgbm_900_lr02 | 292,036 | 319,170 |
| price_new_cars_2010plus | lgbm_900_lr02 | 376,902 | 413,633 |
| price_old_cars_pre2010 | lgbm_600_lr03 | 240,116 | 263,931 |

Each family is scored on its own test rows, so compare families using
`reports/model_comparison.md` (all on the same 19,525 test rows), not this table.

---

## 4. MLflow from the command line

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"          # hides an MLflow info message

.venv\Scripts\mlflow experiments search       # all experiments + IDs
.venv\Scripts\mlflow runs list --experiment-id 2      # runs of carcompass/price_global_ab
.venv\Scripts\mlflow runs describe --run-id d7941784433c4eb38146883375f91895   # champion run, full JSON
```

Experiment IDs here: 1 `carcompass-price`, 2 `price_global_ab`, 3 `price_dataset_a`,
4 `price_dataset_b`, 5 `price_new_cars_2010plus`, 6 `price_old_cars_pre2010`.

---

## 5. MLflow from Python (queries for the presentation)

Save as e.g. `query.py` and run with `.venv\Scripts\python query.py`:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# All runs of one family, best validation MAE first
runs = mlflow.search_runs(experiment_names=["carcompass/price_global_ab"],
                          order_by=["metrics.val_mae_pkr ASC"])
print(runs[["tags.mlflow.runName", "params.model", "metrics.val_mae_pkr",
            "metrics.model_mb", "tags.champion"]].head(5).to_string(index=False))

# The champion of every family, with its commit
champs = mlflow.search_runs(search_all_experiments=True,
                            filter_string="tags.champion = 'true'")
print(champs[["tags.family", "tags.mlflow.runName", "metrics.val_mae_pkr",
              "metrics.test_mae_pkr", "tags.git_commit"]].to_string(index=False))

# Load the production model straight from the registry by alias
model = mlflow.pyfunc.load_model("models:/carcompass-price-global-ab@production")
```

Serving a registered model directly with MLflow _(not run)_:
`.venv\Scripts\mlflow models serve -m "models:/carcompass-price-global-ab@production" --env-manager local -p 5001`
(Our API in `src/app.py` loads models from `models/manifest.json` instead.)

---

## 6. Training commands

| Step | Command (PowerShell) | make target | Time |
|---|---|---|---|
| Download A + B (needs Kaggle token) | `.venv\Scripts\python scripts/download_data.py` | none | minutes |
| Prepare shared A+B data | `.venv\Scripts\python -m src.prepare_multi --input-a data/raw/pakistan_used_cars.csv --input-b data/raw/usedCars.json` | `prepare-multi` | ~1–2 min |
| Train all 5 families (60 runs + ablation, registers, writes `models/manifest.json`) | `.venv\Scripts\python -m src.train_families` | `train-multi` | **1–2 h, high RAM** |
| Continue after a crash | `.venv\Scripts\python -m src.train_families --resume` | none | |
| Train without touching the registry | `... -m src.train_families --no-register --no-ablation` | none | |
| Weight-space / beat attempt (LGBM soups + distilled MLP WSP → `reports/wsp_comparison.md`) | `.venv\Scripts\python -m src.train_wsp_beat` | `train-wsp` | ~10–15 min |
| Prepare Dataset A only (+ log data version to MLflow) | `.venv\Scripts\python -m src.prepare_data --input data/raw/pakistan_used_cars.csv --min-model-count 5 --min-city-count 20 --check-low-mileage --log-mlflow --tracking-uri sqlite:///mlflow.db` | `prepare-real` | ~1 min |
| Train Dataset A only (12 runs + era + quantile, registers `carcompass-price@production`) | `$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"; .venv\Scripts\python -m src.train` | `train-real` | long (big RFs) |

Our raw files are in `D:\IBA\semester_6\MLOps\Project\Raw_data\` (`PakWheels Dataset.csv`
= Dataset A, `usedCars.json` = Dataset B). Pass those paths to `--input-a`, `--input-b`, `--input`.

**Do not run full training during the presentation.** It takes hours and overwrites
`models/` and the registry.

### Live demo that is safe (~40 seconds)

Trains all 12 configs on the small synthetic sample into a separate `demo/` folder and a
separate experiment, without touching the real models, data or registry:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
.venv\Scripts\python -m src.prepare_data --input data/sample/sample_listings.csv --out demo/processed --report demo/data_quality.md --log-mlflow --tracking-uri sqlite:///mlflow.db
.venv\Scripts\python -m src.train --train demo/processed/train.csv --test demo/processed/test.csv --out demo/models --experiment carcompass-demo --no-register
```

Then refresh the UI: new experiments `carcompass-data` (the data-version run with the quality
report artifact) and `carcompass-demo` (12 runs, champion tagged) appear.
**Keep `--no-register`.** Without it the script would register the sample model as
`carcompass-price@production`. Afterwards delete the `demo/` folder and the two
experiments (UI → experiment → Delete).

---

## 7. Validation (data and model)

| Validation | Where | How to show it |
|---|---|---|
| Range checks, missing values, duplicates, low-mileage, rare make/model/city rules | `src/prepare_data.py`, `src/prepare_multi.py` | `reports/data_quality_multi.md`, `reports/data_quality.md` |
| Pandera schema check | `check_schema()` in `src/prepare_data.py` | "schema check: passed" in the quality reports |
| Group-aware split (fit 64 / val 8 / cal 8 / test 20) | `assign_splits()` in `src/prepare_multi.py` | `data/processed_multi/listings.csv` column `split` |
| Champion chosen on **validation** MAE, test used once | `train_family()` in `src/train_families.py` | `val_mae_pkr` on all runs, `test_*` only on champion |
| Routing decided on validation (min gain 1%) | `decide_routing()`, `configs/routing.yaml` | `reports/model_comparison.md`, `models/manifest.json` |
| Price range calibrated on the cal split | `q10`, `q90` metrics | `val_interval_coverage` (~80% expected) |
| Request validation (bad year/mileage → HTTP 422) | Pydantic in `src/app.py` | http://localhost:8000/docs |

```powershell
Get-Content reports\data_quality_multi.md
Get-Content reports\model_comparison.md
Get-Content models\manifest.json
```

---

## 8. Testing and lint

```powershell
.venv\Scripts\python -m pytest -v                 # make test → 35 passed (~11 s)
.venv\Scripts\python -m pytest tests/test_api.py -v
.venv\Scripts\python -m ruff check .              # make lint
```

| File | Tests | What it covers |
|---|---|---|
| `tests/test_api.py` | 12 | /predict, /explain, /health, /metrics, /models, /options, home page, 422 on bad input, 503 without a model |
| `tests/test_prepare_data.py` | 5 | PKR parsing, bad rows dropped, duplicates, reproducible split without leakage |
| `tests/test_prepare_combined.py` | 5 | Dataset B loading and price factor |
| `tests/test_model_router.py` | 5 | manifest loading, year routing, disabled specialists |
| `tests/test_prepare_multi.py` | 4 | shared dataset, cross-source duplicates, split labels |
| `tests/test_model_loading.py` | 4 | artifacts load, missing model raises, assumed fields widen range |

CI (GitHub Actions, `.github/workflows/ci.yml`) runs ruff, pytest and a Docker build on every push.

### API check

```powershell
.venv\Scripts\python -m uvicorn src.app:app --port 8000      # make run (separate terminal)
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/models
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{\"make\":\"Toyota\",\"model\":\"Corolla\",\"year\":2016,\"mileage\":75000}'
```
Test page: http://localhost:8000/ · API docs: http://localhost:8000/docs

---

## 9. Things to know before presenting (local MLflow vs the slides)

1. **`carcompass-price` in our local MLflow was trained on the synthetic sample**, not real
   Dataset A: champion `ridge_alpha_01`, MAE ~62,003. Slide 7 says `lgbm_900_lr02`, 352,511,
   `carcompass-price@production`; that was from a teammate's run on real Dataset A. If you
   open this experiment live, the numbers will not match. Show the `carcompass/*` experiments
   instead, or re-run Dataset A training first (section 6, long).
2. **No `carcompass-data` experiment yet** locally. The data-versioning run appears only
   after `prepare_data --log-mlflow` (the safe demo in section 6 creates it).
3. **Models are logged without a signature** (input schema). Small gap; fine for M1.
4. Slide 9 says 32 tests; there are now **35**.
5. Slide 10: "8,558 copies of Dataset A cars were removed". These are rows that *match on
   make, model, year, mileage, city*, not proven same cars (see `Multi-model/Known_Limitations.md`).
