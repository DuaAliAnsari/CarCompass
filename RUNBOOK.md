# Runbook

How to release, deploy, check, redeploy and roll back the CarCompass API.

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

## 2. First deployment (Render)

1. The first time an image is published, open it on GitHub (repository → Packages →
   carcompass → Package settings) and set visibility to **Public**, or give Render
   registry credentials (a GitHub token with `read:packages`).
2. Render → **New → Web Service → Existing image**.
   Image URL: `ghcr.io/duaaliansari/carcompass:v1.0-milestone1` (a fixed tag, not
   `latest`, so you always know what runs).
3. Settings: port **8000**, health check path **`/health`**, instance type Free
   (the API needs about 250 MB of memory).
4. Deploy, then run the checks in section 4 against the new URL and write it under
   **Live endpoint** below and in the README.

Railway works the same way: New project → Deploy a Docker image → same image URL,
then set `PORT=8000` and the health check path `/health`.

## 3. Redeploy

- **New version:** release it (section 1), then in Render → Settings → Image URL
  change the tag to the new one (for example `...:v2.0-milestone2`) and click
  **Save, rebuild and deploy**.
- **Same version again** (e.g. after a crash): Render → **Manual Deploy → Deploy latest
  reference**.

Then run the checks in section 4.

## 4. Check a deployment

```bash
URL=https://<your-service>.onrender.com

curl -s $URL/health
# {"status":"ok","model_loaded":true,"model_version":"price_global_ab@...",
#  "models_served":["price_global_ab","price_new_cars_2010plus"],"app_version":"v2.0-milestone2"}

curl -s -X POST $URL/predict -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Corolla","year":2016,"mileage":75000}'

curl -s -X POST $URL/explain -H "Content-Type: application/json" \
  -d '{"make":"Suzuki","model":"Mehran","year":2005,"mileage":150000}'

curl -s $URL/models     # which models are served and for which years
curl -s $URL/metrics    # request count, average latency, errors
```

Healthy means: `model_loaded` is `true`, `app_version` is the tag you deployed,
`/predict` returns a price inside its range, and `selected_model` is
`price_new_cars_2010plus` for cars from 2010 and `price_global_ab` for older cars.
The test page at `$URL/` has a **Run all checks** button that does all of this.

## 5. Roll back

If a deployment breaks (health check failing, errors in `/metrics`, wrong prices):

1. Find the last good tag: `git tag --sort=-creatordate` or the repository's Packages
   page (every release tag is kept there).
2. Render → Settings → Image URL: set the tag back, e.g.
   `ghcr.io/duaaliansari/carcompass:v1.0-milestone1`, and deploy.
   (Render also has **Rollback** on every earlier deploy in the Events tab.)
3. Check `GET /health`: `app_version` must show the old tag. Run the rest of section 4.
4. Fix the problem on a branch and release a new tag; never move or delete a tag
   that was deployed.

To roll back only the **models** while keeping new code: revert the commit that changed
`deploy/models/` (`git revert <commit>`), merge, and release a new tag.

## 6. Monitoring

- `GET /metrics`: request count, error count, average latency, app and model version.
- Every request is logged as one JSON line (method, path, status, latency in ms, app and
  model version). On Render: Logs tab.
- `GET /health` is polled by Docker's `HEALTHCHECK` and by the platform's health check.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` says `model_loaded: false`, `/predict` gives 503 | `deploy/models/` was empty when the image was built | `make export-models`, commit, release a new tag |
| Image builds but the API crashes on start with an unpickling error | library versions in the image differ from training | set the pins in `requirements-serve.txt` to the training versions |
| Platform restarts the service, out of memory | too many or too big models served | keep champions under `max_model_mb` (`configs/model_families.yaml`) |
| Render cannot pull the image | GHCR package is private | make it public or add registry credentials (section 2) |
| A request gets `422` | invalid input (e.g. year 1850, missing mileage) | expected; see `/docs` for the input rules |

## Live endpoint

_Not deployed yet. Write the URL here after the first deployment, or a note that it was
shut down after grading to save cost._
