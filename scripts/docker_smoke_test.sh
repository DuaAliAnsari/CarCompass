#!/bin/sh
# Start the image, check that the API really serves a model, then stop it.
# Usage: sh scripts/docker_smoke_test.sh carcompass:dev [port]
# Used by `make docker-smoke` and by the CI build job.
set -eu

IMAGE="${1:?usage: docker_smoke_test.sh IMAGE [PORT]}"
PORT="${2:-8000}"
NAME="carcompass-smoke-$$"
URL="http://localhost:$PORT"
CAR='{"make":"Toyota","model":"Corolla","year":2016,"mileage":75000}'

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" -p "$PORT:8000" "$IMAGE" >/dev/null

fail() {
  echo "FAIL: $1"
  echo "--- container logs ---"
  docker logs "$NAME" 2>&1 | tail -30
  exit 1
}

i=0
until curl -fs "$URL/health" >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -gt 60 ] && fail "API did not start within 60 s"
  sleep 1
done

health=$(curl -fs "$URL/health")
echo "health:  $health"
echo "$health" | grep -q '"model_loaded":true' || fail "no model loaded in the image"

predict=$(curl -fs -X POST "$URL/predict" -H "Content-Type: application/json" -d "$CAR") \
  || fail "/predict did not return 200"
echo "predict: $predict"
echo "$predict" | grep -q '"estimated_price_pkr":[1-9]' || fail "/predict returned no price"

curl -fs -X POST "$URL/explain" -H "Content-Type: application/json" -d "$CAR" \
  | grep -q '"top_factors"' || fail "/explain returned no top factors"

status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL/predict" \
  -H "Content-Type: application/json" -d '{"make":"Toyota"}')
[ "$status" = "422" ] || fail "bad input gave $status, expected 422"

curl -fs "$URL/" | grep -q "Milestone 1 checks" || fail "test page at / not served"
curl -fs "$URL/metrics" | grep -q '"request_count"' || fail "/metrics not working"

echo "PASS: $IMAGE serves a model; /health, /predict, /explain, /, /metrics OK"
