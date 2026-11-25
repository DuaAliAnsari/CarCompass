const $ = (id) => document.getElementById(id);
const CAR = { make: "Toyota", model: "Corolla", year: 2016, mileage: 75000 };

async function call(method, path, body) {
  const started = performance.now();
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  return { status: res.status, data, ms: Math.round(performance.now() - started), path };
}

function showProbe(result) {
  $("probe-status").textContent = String(result.status);
  $("probe-ms").textContent = result.ms + " ms";
  $("probe-path").textContent = result.path;
  $("probe-raw").textContent = JSON.stringify(result.data, null, 2);
}

async function refreshLive() {
  const [health, metrics] = await Promise.all([
    call("GET", "/health").catch(() => null),
    call("GET", "/metrics").catch(() => null),
  ]);
  const pill = $("dash-health");
  if (!health || health.status !== 200) {
    pill.className = "status bad";
    pill.textContent = "API not reachable";
    $("live-api").textContent = "Down";
    return;
  }
  const h = health.data;
  if (h.model_loaded) {
    pill.className = "status ok is-live";
    pill.textContent = "Model ready · " + (h.models_served || []).length + " served";
    $("live-api").textContent = "Up";
  } else {
    pill.className = "status bad";
    pill.textContent = "API up, no model";
    $("live-api").textContent = "No model";
  }
  $("live-api-sub").textContent = h.app_version
    ? "app " + h.app_version + " · healthy"
    : "Healthy process";
  $("live-wsp").textContent = h.wsp_loaded ? "Ready" : "Off";
  if (metrics && metrics.status === 200) {
    $("live-req").textContent = metrics.data.request_count;
    const ms = Number(metrics.data.avg_latency_ms);
    $("live-lat").textContent = Number.isFinite(ms) ? String(Math.round(ms * 100) / 100) : "—";
  }
}

async function probe(kind) {
  let result;
  if (kind === "health") result = await call("GET", "/health");
  else if (kind === "metrics") result = await call("GET", "/metrics");
  else if (kind === "stats") result = await call("GET", "/stats");
  else if (kind === "models") result = await call("GET", "/models");
  else if (kind === "options") result = await call("GET", "/options");
  else if (kind === "predict") result = await call("POST", "/predict", CAR);
  else if (kind === "explain") result = await call("POST", "/explain", CAR);
  else if (kind === "wsp") result = await call("POST", "/predict", { ...CAR, predictor: "wsp" });
  else if (kind === "suite") {
    const before = await call("GET", "/stats");
    const pred = await call("POST", "/predict", CAR);
    const expl = await call("POST", "/explain", CAR);
    const after = await call("GET", "/stats");
    result = {
      status: pred.status,
      ms: pred.ms + expl.ms,
      path: "suite: /predict → /explain → /stats",
      data: {
        before_request_count: before.data && before.data.request_count,
        predict: pred.data,
        explain_top: expl.data && (expl.data.top_factors || expl.data.factors || expl.data),
        after_request_count: after.data && after.data.request_count,
        avg_latency_ms: after.data && after.data.avg_latency_ms,
      },
    };
  } else return;

  showProbe(result);
  await refreshLive();
}

document.querySelectorAll("[data-api]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try { await probe(btn.getAttribute("data-api")); }
    finally { btn.disabled = false; }
  });
});

refreshLive();
setInterval(refreshLive, 8000);
