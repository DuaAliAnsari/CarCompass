# Runbook

How to release, deploy, check, redeploy and roll back the CarCompass API.

## Why deploy showed “API up, no model loaded”

The FastAPI process starts even when `models/` is empty (`src/app.py` catches
`FileNotFoundError`). `/health` then returns `model_loaded: false`. The site
shows the red pills and Make/Model stay on “Loading…” because `/options` needs
a loaded encoder.

Fix: ship served artifacts in `deploy/models/` (committed), build the image with
`COPY deploy/models/ models/`, and redeploy. `make export-models` refreshes that
pack after retraining.

## What gets deployed

| Piece | Where | Made by |
|---|---|---|
| Served models (default + switched-on specialists, ~25 MB) | `deploy/models/` (committed) | `make export-models` |
| Docker image | `ghcr.io/duaaliansari/carcompass:<tag>` | CI `publish` job, on every `v*` git tag |
| Running service | hosting platform (Render, see below) | deploys one image tag |

Each image contains the code **and** the models, so an image tag fixes both. The image
reports its tag in `/health` and `/metrics` (`app_version`), so you can always see what
is running.

## 1. Release a new version

1. Retrain if the data or the code changed (needs `data/raw/`, see README):
   `make train-multi`. Check `reports/model_comparison.md`.
2. Copy the served models into `deploy/models/` and check they load:
   `make export-models` (it prices a test car; it fails if a model is broken).
3. If you retrained with new library versions, update the pins in
   `requirements-serve.txt` to the versions you trained with
   (`pip list | grep -iE "scikit-learn|lightgbm|numpy|pandas|scipy|joblib"`).
4. Optional local check (needs Docker Desktop running): `make docker-smoke TAG=dev`.
5. Commit on a branch, open a pull request, get one approval, merge to `main`.
   CI must be green: lint, tests, Docker build **and** the smoke test, which starts
   the image and checks that a model is loaded and `/predict` answers.
6. Tag the merge commit and push the tag:
   ```bash
   git switch main && git pull
   git tag -a v2.0-milestone2 -m "Milestone 2"
   git push origin v2.0-milestone2
   ```
   CI builds the image again and publishes `ghcr.io/duaaliansari/carcompass:v2.0-milestone2`
   and `:latest`. Check it under the repository's **Packages** on GitHub.

## 2. First deployment (Render) — preferred: build from Git

Deploy **`main`** (post–PR #11, commit `77ad79b` or newer). That tip has the product
site, `/guide`, `/stats` dashboard, WSP lab, and `deploy/models/` in the image.
Do **not** point Render at the old `v1.0-milestone1` GHCR image if you want the
current UI — that tag is Milestone 1 only.
The live service is already at https://carcompass-51ei.onrender.com (see **Live endpoint**).

### Option A — Dockerfile from GitHub (recommended)

1. Render → **New → Web Service** → connect **`DuaAliAnsari/CarCompass`**
   (or the fork `Sagarlekhraj-19/CarCompass` if that is what Render is linked to).
2. Settings:
   - **Branch:** `main`
   - **Language / Runtime:** **Docker**
   - **Dockerfile path:** `./Dockerfile` (repo root)
   - **Docker Command:** leave **empty** (use the image `CMD`; it binds
     `0.0.0.0` and `$PORT`)
   - **Health check path:** `/health`
   - **Instance:** Free (needs ~250–512 MB RAM)
3. Do **not** hardcode port `8000` in a custom start command. Render sets `PORT`
   (often `10000`); the Dockerfile uses `${PORT:-8000}`.
4. Deploy. First free-tier boot can take a few minutes (build + model load).
5. Run the checks in section 4 and paste the URL under **Live endpoint** + README.

Blueprint alternative: use the repo’s `render.yaml` (**New → Blueprint**).

### Option B — pre-built GHCR image (after you publish a `v*` tag)

1. Tag and push so CI `publish` runs (section 1), e.g. `v2.0-milestone2`.
2. Make the package public (repo → Packages → carcompass) or give Render a
   GitHub token with `read:packages`.
3. Render → **New → Web Service → Existing image**
   `ghcr.io/duaaliansari/carcompass:v2.0-milestone2` (pin a tag, not `latest`).
4. Health check `/health`. Leave start command empty unless you pass `$PORT`.

Railway: same as Option B (image URL) or build from the Dockerfile with `$PORT`.

## 3. Redeploy

- **New version:** release it (section 1), then in Render → Settings → Image URL
  change the tag to the new one (for example `...:v2.0-milestone2`) and click
  **Save, rebuild and deploy**.
- **Same version again** (e.g. after a crash): Render → **Manual Deploy → Deploy latest
  reference**.

Then run the checks in section 4.

## 4. Check a deployment

```bash
URL=https://carcompass-51ei.onrender.com

curl -s $URL/health
# {"status":"ok","model_loaded":true,"model_version":"price_global_ab@...",
#  "models_served":["price_global_ab","price_new_cars_2010plus"],"app_version":"v2.0-milestone2"}

curl -s -X POST $URL/predict -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Corolla","year":2016,"mileage":75000}'

curl -s -X POST $URL/explain -H "Content-Type: application/json" \
  -d '{"make":"Suzuki","model":"Mehran","year":2005,"mileage":150000}'

curl -s $URL/models     # which models are served and for which years
curl -s $URL/stats      # request count, average latency, errors (alias: /metrics)
```

Healthy means: `model_loaded` is `true`, `app_version` is the tag you deployed,
`/predict` returns a price inside its range, and `selected_model` is
`price_new_cars_2010plus` for cars from 2010 and `price_global_ab` for older cars.
The test page at `$URL/` has a **Run all checks** button that does all of this.

## 5. Roll back

If a deployment breaks (health check failing, errors in `/stats`, wrong prices):

1. Find the last good tag: `git tag --sort=-creatordate` or the repository's Packages
   page (every release tag is kept there).
2. Render → Settings → Image URL: set the tag back, e.g.
   the previous release tag that was really published (for example
   `ghcr.io/duaaliansari/carcompass:v2.0-milestone2` when a later tag breaks), and deploy.
   (Render also has **Rollback** on every earlier deploy in the Events tab.)
3. Check `GET /health`: `app_version` must show the old tag. Run the rest of section 4.
4. Fix the problem on a branch and release a new tag; never move or delete a tag
   that was deployed.

A rollback needs an older image that exists. After the first release, publish a second
tag once (for example `v2.0.1-milestone2` from a small change) so there is a tag to roll
back to, and check both appear under the repository's Packages on GitHub.

To roll back only the **models** while keeping new code: revert the commit that changed
`deploy/models/` (`git revert <commit>`), merge, and release a new tag.

## 6. Monitoring

- `GET /stats` (alias `/metrics`): request count, error count, average latency, app and model version.
- Every request is logged as one JSON line (method, path, status, latency in ms, app and
  model version). On Render: Logs tab.
- `GET /health` is polled by Docker's `HEALTHCHECK` and by the platform's health check.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` says `model_loaded: false`, `/predict` gives 503 | `deploy/models/` was empty when the image was built | `make export-models`, commit, release a new tag |
| Image builds but the API crashes on start with an unpickling error | library versions in the image differ from training | set the pins in `requirements-serve.txt` to the training versions |
| Platform restarts the service, out of memory | too many or too big models served | keep champions under `max_model_mb` (`configs/model_families.yaml`) |
| Render cannot pull the image | GHCR package is private | make it public or add registry credentials (section 2B), or use Option A (Git + Dockerfile) |
| Deploy fails: no open ports / port scan timeout | app not bound to `0.0.0.0:$PORT` | leave Docker Command empty; do not force `--port 8000` on Render |
| Health check fails while logs look fine | path wrong or still loading models | set health check to `/health`; wait for free-tier cold start |
| Old UI / no `/guide` or `/stats` page | image or branch is pre–PR #11 | deploy `main` (Option A) or a tag built after `77ad79b` |
| A request gets `422` | invalid input (e.g. year 1850, missing mileage) | expected; see `/docs` for the input rules |

## Live endpoint

**Status:** live at https://carcompass-51ei.onrender.com

Render free tier: the service sleeps when idle. Wake it with `curl -s $URL/health` (the
first call can take about a minute) before using or recording it. Counters in `/stats`
reset to zero after every wake or redeploy. `/health` must show `model_loaded: true` and
`app_version` equal to the deployed image tag.

Redeploy or rebuild with **§2 Option A** (Git + Dockerfile on `main`) so the host stays
on the full current site. Use Option B only after publishing a new `v*` image.

The same URL is in `README.md` under **Deployment**. If the service is stopped after
grading to save cost, replace the URL with:
_Live service shut down after grading to save cost._

### Milestone 2 requirement map (this file)

| Course ask | Covered here |
|---|---|
| Redeploy | §3 |
| Roll back to a previous Docker image tag | §5 |
| Monitoring (count + latency) | §6 (`/stats` + logs) |
| Live URL / shutdown note | this section |
| Health / predict / explain / metrics checks | §4 |