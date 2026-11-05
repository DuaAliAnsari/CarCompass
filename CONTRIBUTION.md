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

| Task | Member | Branch / PR |
|---|---|---|
| _fill in_ | | |

## Working agreement

- Branch names: `feat/...`, `fix/...`, `infra/...`.
- Every pull request needs at least one teammate's approval before merging to `main`.
- No direct pushes to `main` (turn on branch protection in GitHub settings).
- Commit regularly across the semester, not in the last 48 hours.
- Everyone must be able to explain every part of the code.

## Per-member reflections (2-3 lines each, add at every milestone)

**Dua Ali Ansari:** I set up the repository, CI, Dockerfile and the FastAPI service with its tests, and ran the real Dataset A training with JSON request logs. I learned how much a reproducible pipeline depends on separating data preparation from training and on logging every run to MLflow.

**Sagar Lekhraj:** I built the multi-model setup: the shared A+B dataset with cross-source dedup, five model families, validation-based champion selection with a size limit, and the router behind `/predict`. I also approved the README rewrite (PR #7) and audited the cross-source duplicate removal. I learned why champions must be chosen on validation rows and not on the test set, and that a duplicate key must be checked against the other columns before rows are deleted.

**Fatima Arif:** I built the data preparation and training code, then added Dataset B (PakWheels) with its own schema mapping and sample weight. I learned that dirty values such as `27.4 lacs` need explicit parsing and range checks before any model is worth comparing.
