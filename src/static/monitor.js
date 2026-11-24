const $ = (id) => document.getElementById(id);
const CAR = { make: "Toyota", model: "Corolla", year: 2016, mileage: 75000 };
const histReq = [];
const histLat = [];

async function getJson(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: {
      Accept: "application/json",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  return { status: res.status, data: await res.json() };
}

function pushHist(arr, value, max = 16) {
  arr.push(Number(value) || 0);
  while (arr.length > max) arr.shift();
}

function sparkPath(values, width = 240, height = 72, pad = 8) {
  if (!values.length) return { line: `M0 ${height / 2} L${width} ${height / 2}`, area: "" };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  const pts = values.map((v, i) => {
    const x = values.length === 1 ? width / 2 : (i / (values.length - 1)) * width;
    const y = pad + (1 - (v - min) / span) * (height - pad * 2);
    return [x, y];
  });
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const area = `${line} L${width} ${height} L0 ${height} Z`;
  return { line, area };
}

function paintSpark(lineId, areaId, values) {
  const { line, area } = sparkPath(values);
  const lineEl = $(lineId);
  const areaEl = $(areaId);
  if (lineEl) lineEl.setAttribute("d", line);
  if (areaEl) areaEl.setAttribute("d", area);
}

function paintDots(el, errors) {
  if (!el) return;
  const n = 28;
  el.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const span = document.createElement("span");
    const hot = errors > 0 && i >= n - Math.min(errors, 8);
    span.className = hot ? "is-hot" : "";
    el.appendChild(span);
  }
}

async function refresh() {
  const [health, stats] = await Promise.all([
    getJson("/health").catch(() => null),
    getJson("/stats").catch(() => null),
  ]);

  const pill = $("mon-health");
  if (!health || health.status !== 200) {
    if (pill) {
      pill.className = "status bad";
      pill.textContent = "API offline";
    }
    return;
  }
  const h = health.data;
  if (pill) {
    pill.className = h.model_loaded ? "status ok" : "status bad";
    pill.textContent = h.model_loaded
      ? `Model ready · ${(h.models_served || []).length} loaded`
      : "API up, no model";
  }

  if (!stats || stats.status !== 200) return;
  const s = stats.data;
  pushHist(histReq, s.request_count);
  pushHist(histLat, s.avg_latency_ms);

  $("d-req").textContent = Number(s.request_count).toLocaleString("en-US");
  $("d-lat").textContent = `${s.avg_latency_ms}`;
  $("d-err").textContent = Number(s.error_count).toLocaleString("en-US");
  $("d-model").textContent = s.model_version || "—";
  $("d-app").textContent = `app ${s.app_version || "—"}`;
  $("d-req-sub").textContent = s.request_count
    ? "Counted API traffic (health/stats excluded)"
    : "Send a /predict to start the counter";
  $("d-lat-sub").textContent = s.request_count ? "Milliseconds, running average" : "Waiting for traffic";
  $("d-err-sub").textContent = s.error_count ? "Investigate recent 5xx" : "No server errors yet";
  const chip = $("d-err-chip");
  if (chip) {
    chip.className = s.error_count ? "dash-chip bad" : "dash-chip";
    chip.textContent = s.error_count ? "Attention" : "5xx";
  }

  $("j-req").textContent = s.request_count;
  $("j-lat").textContent = s.avg_latency_ms;
  $("j-err").textContent = s.error_count;
  $("raw-json").textContent = JSON.stringify(s, null, 2);

  const okPct = s.request_count
    ? Math.max(0, Math.round(100 * (1 - (s.error_count || 0) / s.request_count)))
    : 100;
  if ($("d-ok-pct")) $("d-ok-pct").textContent = okPct + "%";
  if ($("d-ok-sub")) {
    $("d-ok-sub").textContent = s.request_count
      ? `${s.request_count - (s.error_count || 0)} ok · ${s.error_count || 0} errors`
      : "No counted traffic yet";
  }
  const bar = $("d-ok-bar");
  if (bar) {
    bar.style.width = okPct + "%";
    bar.setAttribute("data-pct", okPct + "%");
  }

  paintSpark("sparkA-line", "sparkA-area", histReq);
  paintSpark("sparkB-line", "sparkB-area", histLat);
  paintDots($("d-err-dots"), s.error_count);
}

$("bump-btn")?.addEventListener("click", async () => {
  const btn = $("bump-btn");
  btn.disabled = true;
  try {
    await getJson("/predict", CAR);
  } finally {
    btn.disabled = false;
    await refresh();
  }
});

refresh();
setInterval(refresh, 4000);
