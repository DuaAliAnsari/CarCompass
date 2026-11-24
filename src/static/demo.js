const $ = (id) => document.getElementById(id);
const CAR = { make: "Toyota", model: "Corolla", year: 2016, mileage: 75000 };

async function call(method, path, body) {
  const start = performance.now();
  const res = await fetch(path, {
    method,
    headers: {
      Accept: "application/json",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  return { status: res.status, data, ms: Math.round(performance.now() - start) };
}

async function refreshMetrics() {
  const metrics = await call("GET", "/metrics").catch(() => null);
  if (!metrics || metrics.status !== 200) return;
  $("m-req").textContent = metrics.data.request_count;
  $("m-lat").textContent = metrics.data.avg_latency_ms + " ms";
  $("m-err").textContent = metrics.data.error_count;
}

const CHECKS = [
  ["GET /health says ok and a model is loaded", async () => {
    const response = await call("GET", "/health");
    const data = response.data || {};
    return [response.status === 200 && data.status === "ok" && data.model_loaded, response];
  }],
  ["POST /predict returns a price inside its range", async () => {
    const response = await call("POST", "/predict", CAR);
    const data = response.data || {};
    const ok = response.status === 200 && data.range_low_pkr <= data.estimated_price_pkr
      && data.estimated_price_pkr <= data.range_high_pkr && !!data.model_version;
    return [ok, response];
  }],
  ["POST /predict rejects a bad year", async () => {
    const response = await call("POST", "/predict", { ...CAR, year: 1850 });
    return [response.status === 422, response];
  }],
  ["POST /predict rejects missing fields", async () => {
    const response = await call("POST", "/predict", { make: "Toyota", model: "Corolla" });
    return [response.status === 422, response];
  }],
  ["POST /explain returns top factors", async () => {
    const response = await call("POST", "/explain", CAR);
    return [response.status === 200 && ((response.data || {}).top_factors || []).length > 0, response];
  }],
  ["A newer car with fewer kilometres costs more", async () => {
    const newer = await call("POST", "/predict", { ...CAR, year: 2022, mileage: 20000 });
    const older = await call("POST", "/predict", { ...CAR, year: 2008, mileage: 180000 });
    const ok = newer.status === 200 && older.status === 200
      && newer.data.estimated_price_pkr > older.data.estimated_price_pkr;
    return [ok, { status: newer.status, ms: newer.ms + older.ms }];
  }],
  ["GET /metrics counts the requests", async () => {
    const response = await call("GET", "/metrics");
    const data = response.data || {};
    return [response.status === 200 && data.request_count > 0 && data.avg_latency_ms >= 0, response];
  }],
  ["GET /docs is up", async () => {
    const response = await fetch("/docs");
    return [response.status === 200, { status: response.status, ms: 0 }];
  }],
];

async function runChecks() {
  $("checks-btn").disabled = true;
  $("checks").innerHTML = "";
  let passed = 0;
  for (const [name, fn] of CHECKS) {
    const item = document.createElement("li");
    item.innerHTML = `<span>…</span><span>${name}</span><span class="note"></span>`;
    $("checks").appendChild(item);
    let ok = false;
    let info = {};
    try { [ok, info] = await fn(); } catch (e) { info = { status: "error" }; }
    if (ok) passed += 1;
    item.children[0].className = ok ? "pass" : "fail";
    item.children[0].textContent = ok ? "OK" : "No";
    item.children[2].textContent = `${info.status ?? ""} · ${info.ms ?? 0} ms`;
    if (info.data) $("raw").textContent = JSON.stringify(info.data, null, 2);
  }
  const all = passed === CHECKS.length;
  $("checks-summary").innerHTML = `<span class="${all ? "pass" : "fail"}">${passed} / ${CHECKS.length} checks passed</span>`;
  $("checks-btn").disabled = false;
  refreshMetrics();
}

$("checks-btn").addEventListener("click", runChecks);
refreshMetrics();
