# Runbook (complete this in Milestone 2)

## Redeploy
1. Merge to `main` and wait for CI to pass.
2. Make sure the champion model is in `models/`.
3. Build and tag: `docker build --build-arg APP_VERSION=vX.Y -t carcompass:vX.Y .`
4. Push the image or trigger the deploy on the hosting platform. _Add exact steps for the platform you chose._
5. Check `GET /health` and `GET /metrics` on the live URL.

## Roll back
1. Find the last good tag (for example `vX.(Y-1)`).
2. Redeploy that image tag on the hosting platform.
3. Check `GET /health`: `app_version` should show the old tag.

## Canary release (if added)
_Describe how the new image gets a small share of traffic, what to watch in `/metrics`, and how to promote or cancel._

## Live endpoint
_URL here, or a note that it was shut down after grading to save cost._

## Example requests
_Add cURL examples against the deployed service._
