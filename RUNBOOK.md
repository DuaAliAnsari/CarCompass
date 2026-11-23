# Runbook

## Why deploy showed “API up, no model loaded”

The FastAPI process starts even when `models/` is empty (`src/app.py` catches
`FileNotFoundError`). `/health` then returns `model_loaded: false`. The site
reads that and shows the red pills; Make/Model stay on “Loading…” because
`/options` needs a loaded encoder.

On Render/Git deploy this usually means `models/*.joblib` never reached the
host: they were gitignored. The served artifacts are now allow-listed in
`.gitignore` so Docker/`COPY models/` gets real files.

## Redeploy

1. Confirm these files are in the branch Render builds from:
   - `models/manifest.json`
   - `models/price_global_ab.joblib` + `.json`
   - `models/price_new_cars_2010plus.joblib` + `.json`
   - `models/price_model.joblib` + `metadata.json` (fallback)
2. Merge/push and wait for CI.
3. Build: `docker build --build-arg APP_VERSION=vX.Y -t carcompass:vX.Y .`
   (build fails if served models are missing).
4. Trigger the Render deploy (or push the image).
5. Check live `GET /health` → `"model_loaded": true`, then reload `/`.

## Roll back

1. Redeploy the previous good image/tag.
2. Confirm `/health` `app_version` and `model_loaded`.

## Live checks

```bash
curl.exe https://YOUR-SERVICE.onrender.com/health
curl.exe https://YOUR-SERVICE.onrender.com/models
curl.exe -X POST https://YOUR-SERVICE.onrender.com/predict -H "Content-Type: application/json" -d "{\"make\":\"Toyota\",\"model\":\"Corolla\",\"year\":2016,\"mileage\":75000}"
```
