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
| Repo setup, Makefile, ruff config, CI workflow, Dockerfile, PR template | Dua Ali Ansari | Fatima Arif | initial commit on `main` |
| FastAPI app (`/predict`, `/explain`, `/health`, `/metrics`, `/docs`), Pydantic schemas | Dua Ali Ansari | Fatima Arif | initial commit on `main` |
| Tests: API, model loading, data preparation | Dua Ali Ansari | Fatima Arif | initial commit on `main` |
| Real-data run, JSON request logs, `make prepare-real` / `train-real`, README results and champion justification | Dua Ali Ansari | Sagar Lekhraj | `feat/m1-finish` |
| Data preparation: PKR parsing, range checks, dedup with `listing_group`, grouped 80/20 split, Pandera check, data-quality report | Fatima Arif | Sagar Lekhraj | `feat/prepare-data` (PR #1) |
| Training: baseline, Ridge, RF, LightGBM, era models, quantile check, `VERSION.json`, `@production` alias | Fatima Arif | Sagar Lekhraj | `feat/train-models` (PR #2) |
| README sections: datasets, licences, limitations | Sagar Lekhraj | Dua Ali Ansari | `feat/readme-datasets` |
| Review of the PRs above and walkthrough of the code before tagging | Sagar Lekhraj | | PR review comments |
| Dataset B (`usedCars.json`): schema mapping, B-only and A+B preparation, B sample weight | Fatima Arif | Sagar Lekhraj | `feat/dataset-b` |
| README and Makefile update for the real PakWheels data | Fatima Arif | Sagar Lekhraj | `docs/readme-contribution` |
| Merged-dataset experiment with a 2022 scrape (not merged: the scrape is not in our data) | Dua Ali Ansari | Sagar Lekhraj | `feat/two-datasets` |
| Multi-model setup: shared A+B dataset with cross-source dedup, five model families, validation-based champions with a size limit, routing manifest and router, `/models`, `/options`, assembly feature, test page at `/`, Dataset B download, tests | Sagar Lekhraj | Dua Ali Ansari | `feat/multi-model` |
| Final tag `v1.0-milestone1` | Sagar Lekhraj | Dua Ali Ansari | tag on `main` |

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

**Sagar Lekhraj:** I built the multi-model setup: the shared A+B dataset with cross-source dedup, five model families, validation-based champion selection with a size limit, and the router behind `/predict`. I also wrote the dataset sections of the README and reviewed the other branches. I learned why champions must be chosen on validation rows and not on the test set.

**Fatima Arif:** I built the data preparation and training code, then added Dataset B (PakWheels) with its own schema mapping and sample weight. I learned that dirty values such as `27.4 lacs` need explicit parsing and range checks before any model is worth comparing.
