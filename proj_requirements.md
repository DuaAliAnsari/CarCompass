MLOps - Course Project
BS Computer Science – Fall 2026
1. Project Objectives & Core Requirements
The semester project asks each group to take a single ML model from a notebook to a working, served, and minimally monitored system. The emphasis is on completing the full loop correctly, data to model to API to deployment, rather than stacking many advanced tools.
By the end of the semester, every group must demonstrate:
A reproducible pipeline: clone the repo, run one or few commands, get a trained model plus a running API.
A simple, well-understood ML task (classification or regression) with a clean train/test split and one clearly reported evaluation metric.
Experiment tracking using MLflow (runs, parameters, and the chosen metric visible in the MLflow UI).
A served model via a FastAPI endpoint, with basic input validation and auto-generated API docs.
Containerization: the service runs via Docker with a single documented command.
Basic CI: linting and automated tests running on GitHub Actions on every push.
Deployment to one cloud or managed hosting service, a single VM (EC2/Compute Engine/Azure VM) or an equivalent managed platform (e.g., Render, Railway, AWS App Runner, Azure Web App).
Minimal monitoring: a /health endpoint and a simple way to see request count and latency (raw logs, a basic counter endpoint, or one dashboard panel), a full Prometheus/Grafana/Evidently stack is optional.
Core documentation: README.md, CONTRIBUTION.md, and LICENSE.
2. Project Milestone Assignments & Final Presentation Requirements
The project is delivered in two build milestones plus a final presentation. Each milestone is pushed to a public GitHub repository with the specified Git tag, and CI must pass on that tagged commit.
Milestone 0: Proposal Submission
2-3 page document outlining the following:
Team info, Problem Statement, Proposed Solution, Dataset details, Full M1 + M2 milestone plan, Tech stacks to be used, Risks + mitigations
Marks: 10/100 
Milestone 1: Model & API
Objective: turn a notebook prototype into a clean, reproducible pipeline with a properly evaluated model, tracked experiments, and a served prediction endpoint.
README.md: one-line project pitch, a simple architecture diagram (data -> training -> API), quick-start instructions (git clone && make setup or equivalent), and a short note on the dataset and task.
A data-preparation step that is separate from the training step (e.g., a prepare_data.py or a clearly separated function), including basic validation such as checking for missing values or invalid ranges.
A training script that loads the prepared data, trains a model, evaluates it on a held-out test set, and saves the trained model artifact to disk in a versioned or clearly named way.
MLflow tracking: at least three experiment runs logged with different hyperparameters or model choices, each with parameters and the evaluation metric recorded, and the best run clearly identified and justified in README.md.
A FastAPI service (src/app.py) exposing a /predict endpoint that loads the saved model and returns a prediction, with request validation via Pydantic and auto-generated /docs.
Basic unit tests (pytest) covering at least the data-preparation function, the model-loading logic, and the prediction endpoint, with a short note in README.md on how to run them.
CONTRIBUTION.md with student names/ERP IDs and a member-to-task table for this milestone.
Tag: v1.0-milestone1. Marks: 20/100 
Milestone 2: Containerization, CI/CD & Deployment
Objective: package the Milestone-1 system and put it somewhere reachable, with a working automated pipeline behind it.
Dockerfile: a clean build (single or simple multi-stage) that installs dependencies, copies the trained model and source code, runs the API on the correct exposed port, and includes a basic healthcheck hitting /health
GitHub Actions CI (.github/workflows/ci.yml): a lint job (ruff or flake8), a test job running pytest with a visible pass/fail status, and a build job that builds the Docker image on every push to main.
Deployment: the containerized API deployed and reachable on one cloud or managed hosting service — a single VM (EC2/Compute Engine/Azure VM) or an equivalent managed platform (Render, Railway, AWS App Runner, Azure Web App).
Monitoring in place and demonstrable during grading: request count and average latency visible via logs, a simple counter endpoint, or one dashboard panel.
A short RUNBOOK.md or README section describing how to redeploy the service and how to roll back to a previous Docker image tag if the deployment breaks.
README updated with deployment steps, the live endpoint (or a note that it was shut down after grading to save cost), and example cURL requests against the deployed service.
CONTRIBUTION.md with student names/ERP IDs and a member-to-task table for this milestone.
Tag: v2.0-milestone2. Marks: 20/100
Final Presentation & Demo
A short in-class presentation and live demo of the three milestones accomplished.
10-12 minute presentation: problem statement, pipeline walkthrough, one MLflow screenshot, the CI pipeline passing on the latest tag, and a live demo hitting both /predict and /explain on the deployed service.
A brief discussion (1-2 minutes) of one design trade-off the group made, e.g., choice of model, cloud platform, or prompt design, and why.
Live Q&A with the instructor and TA; any member may be asked to explain any part of the system, including parts they did not personally write.
Attendance and participation are mandatory for all group members
Marks: 35/100 for the presentation/demo, 15/100 for individual contribution and viva
3. Group Size Requirements & Work Distribution Guidelines
Groups of 3 students. Individual submissions are not allowed.
For each of the three milestones (M0, M1, and M2), a different group member should give presentation. On the final presentation day, all three members should give the presentation together.
Each group must submit a CONTRIBUTION.md mapping every member to specific tasks.
Git history (commits, pull requests, branches, and issue assignments) is used to cross-check the contribution table; commit activity should be reasonably distributed across members and dated across the semester, not concentrated in the final 48 hours.
Use a consistent branch-naming convention (feat/..., fix/..., infra/...) and require at least one peer review/approval before merging to main; direct pushes to main are discouraged.
Each milestone submission should include a brief per-member reflection (2-3 lines) on what they built and what they learned, appended to CONTRIBUTION.md.
4. Tentative Timeline
Proposal Submission
End of September
Milestone 1
End of October
Milestone 2
End of November
Final Presentation & Demo
Last week of classes
Individual Contribution & Viva
Last week of classes


5. Project Grading Scheme / Marks Distribution (Out of 100 Marks)

Component
Marks
Notes
 Milestone 0 - Proposal Submission
10
Brief document outlining proposed plan of project
 Milestone 1 - Model & API
20
Pipeline, MLflow tracking, FastAPI, tests
 Milestone 2 - Docker, CI/CD & Deployment
20
Dockerfile, CI, live deployment, monitoring
 Final Presentation & Demo
35
Clarity, live demo, Q&A
 Individual Contribution & Viva
15
Based on commits + short individual viva

Indicative rubric within each milestone
Reproducibility & correctness (does it actually run from a clean clone?), largest weight.
Whether the required tool is genuinely used (a real MLflow run history, not one manual screenshot).
CI health (pipeline passes; tests are meaningful, not trivial placeholders).
Documentation clarity (README is enough for a new person to run the project unaided).
6. Online & On-Campus Office Hours
Instructor Office Hours (tentative)
Monday: 3 pm to 4 pm (Faculty Lounge, First Floor, Aman CED)
Wednesday: 5:30 pm to 6:30 pm (Faculty Lounge, First Floor, Aman CED)
Friday 11:30 am to 12:30 pm (Faculty Lounge, Ground Floor, Adamjee Block)
Please email (bhbutt@iba.edu.pk) at least a day in advance to reserve your spot.
TA Office Hours (email for appointment) 
On-Campus: Monday: 1 pm to 4 pm | Wednesday: 5:30 pm to 6:30 pm | Saturday: 8:30 am to 11:30 am
Online: Monday to Thursday: 7:30 pm to 9:00 pm
Email address: a.randhawa.27095@khi.iba.edu.pk
7. Project Ideas & Required Technology Stack
Groups may propose a custom idea (subject to instructor approval) or pick from the list below. Pick a task simple enough to get fully working end-to-end, a well-executed simple project outscores an ambitious broken one.
Student performance / admission-outcome prediction from an academic dataset (classification).
House price or used-car price prediction (regression) from a public tabular dataset.
Customer churn prediction for a telecom or subscription dataset (classification).
Spam / sentiment classification on a public text dataset (simple NLP, no LLM required).
Loan default or credit-risk prediction from a public financial dataset (classification).
Required Technology Stack
Language & API layer: Python 3.11, FastAPI
Model: scikit-learn (or a similarly simple library) is sufficient, deep learning is not required.
Experiment tracking: MLflow (self-hosted locally is fine).
Testing & linting: pytest, ruff or flake8.
Containerization & CI: Docker, GitHub Actions.
Deployment: one cloud VM or one managed hosting platform ( AWS App Runner, Azure Web App, etc.).
8. Additional Essential Guidelines
Repository Directory Structure (minimum expected)
src/ — training script and app.py (FastAPI service)
tests/ — pytest test files
data/ — small sample data only; do not commit large raw datasets
.github/workflows/ — CI pipeline definition
README.md, CONTRIBUTION.md, LICENSE — at repository root
Academic Integrity Policy
Cross-group code copying, or copying from public repositories without attribution, is treated as plagiarism under the university's academic integrity policy.
Reused open-source components (libraries, templates) must be cited in README.md.
All group members must be able to explain any part of the codebase during the viva; inability to do so affects individual marks.
AI Usage Policy
Use of AI coding assistants (e.g., Claude, ChatGPT, Copilot) for boilerplate, debugging, and documentation drafting is permitted and expected.
Each group must add a short AI_USAGE.md noting which tools were used, for what tasks, and confirming the group understood and reviewed any AI-suggested code.
AI tools may not be used to generate a submission without genuine understanding; viva questions will probe design decisions directly.
