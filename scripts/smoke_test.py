"""Smoke test for a running CarCompass service (local, Docker or deployed).

Usage:
    python scripts/smoke_test.py                       # http://127.0.0.1:8000
    python scripts/smoke_test.py http://localhost:8000
    python scripts/smoke_test.py https://your-app.onrender.com

Uses only the standard library. Exits with code 1 if any check fails.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
CAR = {"make": "Toyota", "model": "Corolla", "year": 2016, "mileage": 75000}


def call(path: str, body: dict | None = None, timeout: int = 60):
    """Return (status_code, parsed_json_or_text)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw, status = resp.read().decode(), resp.status
    except urllib.error.HTTPError as err:
        raw, status = err.read().decode(), err.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def stats_path() -> str:
    return "/stats" if call("/stats")[0] == 200 else "/metrics"


results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        ok, note = fn()
    except Exception as exc:  # network errors, bad JSON, missing keys
        ok, note = False, f"{type(exc).__name__}: {exc}"
    results.append((name, bool(ok), note))


def t_health():
    status, body = call("/health")
    return status == 200 and body["status"] == "ok" and body["model_loaded"] is True, \
        f"status={status} model_loaded={body.get('model_loaded')} v={body.get('model_version')}"


def t_home():
    status, body = call("/")
    return status == 200 and "CarCompass" in str(body), f"status={status}"


def t_options():
    status, body = call("/options")
    makes = body.get("models_by_make", {})
    return status == 200 and bool(makes), f"makes={len(makes)}"


def t_predict():
    status, body = call("/predict", CAR)
    ok = (status == 200 and body["range_low_pkr"] <= body["estimated_price_pkr"]
          <= body["range_high_pkr"])
    return ok, f"status={status} price={body.get('estimated_price_pkr')}"


def t_validation():
    status, _ = call("/predict", {**CAR, "year": 1900})
    return status == 422, f"status={status} (expected 422)"


def t_explain():
    status, body = call("/explain", CAR)
    kind = "per-car" if body.get("car_factors") else "global"
    return status == 200 and bool(body.get("top_factors")), f"status={status} factors={kind}"


def t_counter():
    path = stats_path()
    before = call(path)[1]["request_count"]
    call("/predict", CAR)
    after = call(path)[1]
    return after["request_count"] > before, \
        f"{path}: {before} -> {after['request_count']}, avg={after['avg_latency_ms']} ms"


for label, fn in [("GET /health (model loaded)", t_health), ("GET / (web page)", t_home),
                  ("GET /options", t_options), ("POST /predict", t_predict),
                  ("POST /predict rejects bad year", t_validation),
                  ("POST /explain", t_explain), ("Request counter rises", t_counter)]:
    check(label, fn)

print(f"\nTarget: {BASE}\n")
for name, ok, note in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<34} {note}")
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)