# Contribution

## Team

| Member | Name | ERP ID |
|---|---|---|
| 1 | Sagar Lekhraj | 29325 |
| 2 | Dua Ali Ansari | 28649 |
| 3 | Fatima Arif | 31585 |

## Milestone 1: member-to-task table

| Task | Member | Reviewer | Branch / PR |
|---|---|---|---|
| Repo setup, Makefile, ruff config, CI workflow, Dockerfile, PR template | Dua Ali Ansari | — | initial commit on `main` |
| FastAPI app (`/predict`, `/explain`, `/health`, `/metrics`, `/docs`), Pydantic schemas | Dua Ali Ansari | — | initial commit on `main` |
| Tests: API, model loading, data preparation | Dua Ali Ansari | — | initial commit on `main` |
| Real-data run, JSON request logs, `make prepare-real` / `train-real`, README results and champion justification | Dua Ali Ansari | — | `feat/m1-finish` (PR #3) |
| Data preparation: PKR parsing, range checks, dedup with `listing_group`, grouped 80/20 split, Pandera check, data-quality report | Fatima Arif | — | `feat/prepare-data` (PR #1) |
| Training: baseline, Ridge, RF, LightGBM, era models, quantile check, `VERSION.json`, `@production` alias | Fatima Arif | — | `feat/train-models` (PR #2) |
| Review and approval of the README rewrite; audit of the cross-source duplicate removal and its known-limitation note; README fixes before tagging | Sagar Lekhraj | | PR #7 (approval), `fix/m1-submission` |
| Dataset B (`usedCars.json`): schema mapping, B-only and A+B preparation, B sample weight | Fatima Arif | — | `feat/dataset-b` (PR #4) |
| README and Makefile update for the real PakWheels data | Fatima Arif | — | `docs/readme-contribution` (merged via PR #5) |
| Merged-dataset experiment with a 2022 scrape (not merged: the scrape is not in our data) | Dua Ali Ansari | — | `feat/two-datasets` |
| Multi-model setup: shared A+B dataset with cross-source dedup, five model families, validation-based champions with a size limit, routing manifest and router, `/models`, `/options`, assembly feature, test page at `/`, Dataset B download, tests, README multi-model section | Sagar Lekhraj | Dua Ali Ansari | `feat/multi-model` (PR #5) |
| README rewrite: complete A+B workflow first, Dataset A-only results labelled as a benchmark | Dua Ali Ansari | Sagar Lekhraj | PR #7 |
| Tag `v1.0-milestone1`: first created by Fatima Arif on 30 Oct at `7a29be4`, moved to the final M1 commit | Sagar Lekhraj | | tag on `main` |

"Reviewer" lists approvals recorded on GitHub. The initial commit and PRs #1 to #4 were merged without a
recorded GitHub review, so they show "—".

## Milestone 2: member-to-task table

| Task | Member | Reviewer | Branch / PR |
|---|---|---|---|
| Served models for Docker (`deploy/models/`), `export-models`, Docker smoke test, CI build smoke, GHCR publish job, full RUNBOOK | Dua Ali Ansari | Sagar Lekhraj | `infra/docker-models` (merged via PR #10) |
| Multi-page product site (Estimate / Experiments / Models / Pipeline / Demo), glass hero video, listing photos | Sagar Lekhraj | — | `feat/frontend-site` (PR #10) |
| Ship served `.joblib` into the image / allow-list; Docker missing-model check; clearer “no model loaded” UI | Sagar Lekhraj | — | `feat/frontend-site` (PR #10) |
| Weight-space programming (soups / distill / beat attempt), WSP report, experiments page, WSP lab + serve path | Sagar Lekhraj | — | `feat/frontend-site` (PR #10) |
| Merge remaining M2 work to `main`; fix ruff so CI stays green (lint / test / build) | Sagar Lekhraj | — | PR #10 (`332789b`), follow-up lint fixes |
| Product IA: slim top nav (Estimate / Experiments / Models), hero “Get a price” + Experiments, footer → Project guide | Sagar Lekhraj | — | `fix/m2-docs-status` / `fix/progress-bar-ui` |
| Project guide hub (`/guide`), HTML `/stats` dashboard, milestones / pipeline / monitor / checks / WSP subpages | Sagar Lekhraj | — | `fix/m2-docs-status` (PR #11) |
| Fill M2 contribution / checklist rows; document remaining deploy items | Sagar Lekhraj | — | `fix/m2-docs-status` (PR #11) |
| Render deploy fix: honor host `PORT`, longer HEALTHCHECK start, `render.yaml`, smoke/`make` pass `-e PORT`, README/RUNBOOK Render path | Sagar Lekhraj | Dua Ali Ansari | `fix/render-deploy-main` (PR #12) |
| Resolve RUNBOOK/README merge conflicts on docs branch; review/approve live-URL + Project Focus docs | Sagar Lekhraj | — | `docs/m2-submission` fix; PR #14 approve |
| Live Render service, live URL + deployed cURL, `/stats` docs, Project Focus text, RUNBOOK redeploy/rollback update | Dua Ali Ansari | Sagar Lekhraj | `docs/m2-docs` (PR #14) |
| Product UI polish: CarC + compass + mpass brand, live “Model ready” pulse, chevron CTAs, Cruiser live-preview crop, predictor actions | Sagar Lekhraj | — | `fix/progress-bar-ui` (PR #15) |
| Guide / milestones dashboard: progress % bars, SaaS feature cards, metric tiles, requirement boxes; drop unused Templates mock | Sagar Lekhraj | — | `fix/progress-bar-ui` (PR #16) |
| M2 validation: pytest + ruff on final `main`; smoke checks (health, pages, options, predict, bad-input 422, explain, stats) on local, Docker, and live Render; Docker healthcheck + rollback rehearsal | Fatima Arif | Sagar Lekhraj | `docs/fatima-testing` |
| Final presentation deck + recorded demo video (pipeline walkthrough, MLflow, CI, live `/predict` + `/explain`) | Fatima Arif | — | presentation / recording (M2 close-out) |
| Automated smoke script for CI (`scripts/smoke_test.py`) | Dua Ali Ansari | — | `add-smoke-tests` (PR #18) |

## Working agreement

- Branch names: `feat/...`, `fix/...`, `infra/...`.
- Every pull request needs at least one teammate's approval before merging to `main`.
- No direct pushes to `main` (turn on branch protection in GitHub settings).
- Commit regularly across the semester, not in the last 48 hours.
- Everyone must be able to explain every part of the code.

## Per-member reflections (2-3 lines each, add at every milestone)

**Dua Ali Ansari:** I set up the repository, CI, Dockerfile and the FastAPI service with its tests, and ran the real Dataset A training with JSON request logs. I learned how much a reproducible pipeline depends on separating data preparation from training and on logging every run to MLflow.

**Sagar Lekhraj:** I built the multi-model path: shared A+B prepare with cross-source dedup, five families × configs in MLflow, validation champions with a ≤100 MB serve cap, year routing, `/models` / `/options`, and the Milestone-1 test page. I reviewed the README rewrite (PR #7), fixed submission docs (`fix/m1-submission`), and moved the `v1.0-milestone1` tag to the final commit. I learned that champions must be picked on validation (not test), and that a duplicate key is only safe after checking the other columns.

**Fatima Arif:** I built the data preparation and training code, then added Dataset B (PakWheels) with its own schema mapping and sample weight. I learned that dirty values such as `27.4 lacs` need explicit parsing and range checks before any model is worth comparing.

### Milestone 2

**Dua Ali Ansari:** I packaged the served champions for Docker (`deploy/models/`, `export-models`), added the image smoke test that CI runs on every push, and wrote the redeploy/rollback runbook. I learned that a “healthy” API that starts without models is worse than a failed image build.

**Sagar Lekhraj:** I owned the product UI and Project Focus demo path: multi-page site, WSP soups/distill/beat lab, shipping models so Docker/Render actually load, `/guide` + `/stats` + milestones dashboards, nav/hero/footer IA, and the Render `PORT`/healthcheck blueprint (PRs #10–#12, #15–#16). I resolved docs conflicts, approved Dua’s live-URL write-up (PR #14), and kept CI green with ruff fixes. I learned that “Loading…” was `model_loaded: false` from an empty image, soups need real diversity to beat LightGBM, and Render’s free tier sleeps until `/health` wakes it.

**Fatima Arif:** I validated M2 end-to-end (pytest/ruff, then smoke on local, Docker, and Render) and prepared the final presentation deck plus the recorded demo video. I learned that the same model version and prediction across all three environments is the real deploy proof, and that a clear `/health` version label makes rollback easier to show.

---

## Final presentation & demo — what we deliver (complete)

Status: **Done** (presentation, live demo, and individual viva treated as complete).

| # | What to show | Detail (what we built) | Status |
|---|---|---|---|
| 1 | Problem & dataset | Used-car price regression (PKR); Dataset A + PakWheels B; prepare → group-aware split; ~97.8k cleaned rows | Done |
| 2 | Reproducible pipeline | `make prepare-*` / `train-*` (or documented commands); prepare separate from train; artifacts in `models/` / `deploy/models/` | Done |
| 3 | MLflow screenshot | ≥3 runs / five families × configs; champion on **validation** MAE; LightGBM ~12.6 MB served; README justification | Done |
| 4 | FastAPI live | Deployed `/docs`; Pydantic validation; `/predict` + `/explain` on https://carcompass-51ei.onrender.com | Done |
| 5 | Live `/predict` demo | e.g. Toyota Corolla 2016 → price + 80% range + `selected_model` (year routing: pre-2010 vs 2010+) | Done |
| 6 | Live `/explain` demo | Top factors for a listing (e.g. Mehran 2005 → `price_global_ab`) | Done |
| 7 | CI on tagged release | GitHub Actions lint + pytest + Docker build/smoke green on `main`; tag `v2.0-milestone2` | Done |
| 8 | Docker | Image ships `deploy/models/`; `/health` healthcheck; `make docker-smoke` | Done |
| 9 | Monitoring | `/health`, `/stats` (request count + avg latency), JSON request logs, `/demo` checks panel | Done |
| 10 | RUNBOOK | Redeploy + rollback to a previous image tag (`RUNBOOK.md`) | Done |
| 11 | Product site (extra) | Estimate / Experiments / Models; Project guide `/guide`; milestones dashboard; WSP lab | Done |
| 12 | Project Focus (WSP) | Weight-space soups/distill/blend vs LightGBM champion; experiments + `/wsp` | Done |
| 13 | Design trade-off (1–2 min) | e.g. LightGBM size cap vs huge RF; Render free vs VM; WSP blend vs pure soup | Done |
| 14 | Presentation deck + recorded video | Fatima: slides + recorded walkthrough of pipeline / MLflow / CI / live API | Done |
| 15 | Individual viva | All members can explain any part; treated as complete for close-out | Done |

### Ops notes (not blockers)

| Topic | Meaning | Action if you want polish |
|---|---|---|
| Render `app_version: "dev"` | The code defaults to `APP_VERSION=dev` when the env var is unset. **UI and models are still the latest `main`.** | In Render → Environment set `APP_VERSION=v2.0-milestone2`, then Manual Deploy |
| Tag “5 commits behind” `main` | `v2.0-milestone2` points at the UI merge (`d259e97`). Later commits on `main` are docs (Sagar contribution) + smoke tests (PR #18) only — not missing product features | Optional: move tag to current `main` tip (`24a7ae2`) after this docs PR |
| `testing.md` / `.testing/` | Extra playbook, **not** a course requirement — keep local / untracked | Leave as-is |
