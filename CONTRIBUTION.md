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
| _Fatima Arif: add your M2 task(s) here (review, docs, tests or deploy help), with branch / PR_ | Fatima Arif | | |

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

**Fatima Arif:** _Add 2–3 lines on M2 work (review, docs, or deploy help)._