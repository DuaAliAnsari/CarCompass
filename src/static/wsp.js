const $ = (id) => document.getElementById(id);

async function call(method, path, body) {
  const started = performance.now();
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  return { status: res.status, data, ms: Math.round(performance.now() - started) };
}

function fillSelect(el, values, blank) {
  el.innerHTML = "";
  if (blank) el.add(new Option(blank, ""));
  (values || []).forEach((value) => el.add(new Option(value, value)));
}

function carFromForm() {
  const car = {
    make: $("make").value,
    model: $("model").value,
    year: Number($("year").value),
    mileage: Number($("mileage").value),
  };
  if ($("engine_cc").value) car.engine_cc = Number($("engine_cc").value);
  if ($("city").value) car.city = $("city").value;
  return car;
}

function fmtMae(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Math.round(Number(n)).toLocaleString("en-US");
}

async function refreshMetrics() {
  const metrics = await call("GET", "/metrics").catch(() => null);
  if (!metrics || metrics.status !== 200) return;
  $("m-req").textContent = metrics.data.request_count;
  $("m-lat").textContent = metrics.data.avg_latency_ms + " ms";
  $("m-err").textContent = metrics.data.error_count;
}

function paintResult(prefix, body, ms) {
  $(prefix + "-price").textContent = pkr(body.estimated_price_pkr);
  $(prefix + "-lakh").textContent = "about " + lakh(body.estimated_price_pkr);
  $(prefix + "-range").textContent =
    "Range " + pkr(body.range_low_pkr) + " – " + pkr(body.range_high_pkr);
  const bits = [
    body.selected_model ? "model " + plainModel(body.selected_model) : null,
    body.model_version ? "version " + body.model_version : null,
    ms != null ? ms + " ms" : null,
  ].filter(Boolean);
  $(prefix + "-meta").textContent = bits.join(" · ");
}

async function loadPage() {
  const health = await call("GET", "/health").catch(() => null);
  const status = $("wsp-status");
  if (!health || health.status !== 200 || !health.data.model_loaded) {
    status.className = "status bad";
    status.textContent = health ? "API up, no model loaded" : "API not reachable";
    return;
  }

  if (!health.data.wsp_loaded) {
    status.className = "status bad";
    status.textContent = "WSP experiment not loaded";
    $("form-error").textContent = "Run make train-wsp and make export-models, then restart the API.";
  } else {
    status.className = "status ok";
    status.textContent = "WSP ready · " + (health.data.wsp.program || "experiment");
    const wsp = health.data.wsp;
    $("s-program").textContent = wsp.program || "—";
    $("s-note").textContent = wsp.note || "blend experiment";
    $("s-wsp-mae").textContent = fmtMae(wsp.test_mae_pkr);
    $("s-champ-mae").textContent = fmtMae(wsp.champion_test_mae_pkr);
    if (wsp.test_mae_pkr != null && wsp.champion_test_mae_pkr != null) {
      const delta = wsp.champion_test_mae_pkr - wsp.test_mae_pkr;
      $("s-delta").textContent = (delta >= 0 ? "−" : "+") + fmtMae(Math.abs(delta));
      $("s-beat").textContent = wsp.beat_champion
        ? "Better than global champion"
        : "Did not beat global champion";
    }
  }

  await refreshMetrics();

  const options = await call("GET", "/options");
  if (options.status !== 200) return;
  const makes = Object.keys(options.data.models_by_make);
  fillSelect($("make"), makes);
  $("make").value = makes.includes("Toyota") ? "Toyota" : makes[0];
  const models = options.data.models_by_make[$("make").value] || [];
  fillSelect($("model"), models);
  if (models.includes("Corolla")) $("model").value = "Corolla";
  fillSelect($("city"), options.data.city || [], "Not given");

  $("make").addEventListener("change", () => {
    fillSelect($("model"), options.data.models_by_make[$("make").value] || []);
  });
}

async function runCompare(event) {
  event.preventDefault();
  const button = $("wsp-btn");
  button.disabled = true;
  $("form-error").textContent = "";
  $("compare-summary").textContent = "";
  try {
    const car = carFromForm();
    const [wspRes, routedRes] = await Promise.all([
      call("POST", "/predict", { ...car, predictor: "wsp" }),
      call("POST", "/predict", { ...car, predictor: "routed" }),
    ]);

    if (wspRes.status !== 200) {
      $("form-error").textContent = wspRes.status === 503
        ? "WSP experiment model is not loaded on this host."
        : JSON.stringify(wspRes.data && wspRes.data.detail || wspRes.data);
      return;
    }
    if (routedRes.status !== 200) {
      $("form-error").textContent = "Routed predict failed: " + routedRes.status;
      return;
    }

    paintResult("wsp", wspRes.data, wspRes.ms);
    paintResult("routed", routedRes.data, routedRes.ms);
    $("m-wsp-lat").textContent = wspRes.ms + " ms";
    $("m-routed-lat").textContent = routedRes.ms + " ms";

    const delta = wspRes.data.estimated_price_pkr - routedRes.data.estimated_price_pkr;
    $("m-price-delta").textContent = (delta >= 0 ? "+" : "−") + pkr(Math.abs(delta)).replace("PKR ", "");
    $("compare-summary").textContent = delta === 0
      ? "Both predictors returned the same point estimate for this car."
      : `WSP is ${pkr(Math.abs(delta))} ${delta > 0 ? "higher" : "lower"} than the routed champion on this listing.`;
    $("metrics-note").textContent =
      "Latencies are browser round-trips for this comparison. Process averages update below.";
    $("raw-json").textContent = JSON.stringify({
      wsp: { ms: wspRes.ms, body: wspRes.data },
      routed: { ms: routedRes.ms, body: routedRes.data },
    }, null, 2);

    await refreshMetrics();
  } catch (error) {
    $("form-error").textContent = "The API could not be reached.";
  } finally {
    button.disabled = false;
  }
}

$("wsp-form").addEventListener("submit", runCompare);
$("year").max = new Date().getFullYear() + 1;
loadPage();
