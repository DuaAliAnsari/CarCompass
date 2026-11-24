const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  {
    label: "Corolla 2016",
    make: "Toyota",
    model: "Corolla",
    year: 2016,
    mileage: 75000,
    image: "/assets/img/examples/toyota-corolla.jpg",
  },
  {
    label: "Civic 2018",
    make: "Honda",
    model: "Civic",
    year: 2018,
    mileage: 45000,
    image: "/assets/img/examples/honda-civic.jpg",
  },
  {
    label: "Mehran 2005",
    make: "Suzuki",
    model: "Mehran",
    year: 2005,
    mileage: 150000,
    image: "/assets/img/examples/suzuki-mehran.jpg",
  },
];
const FIELD_LABEL = {
  engine_cc: "Engine size",
  transmission: "Transmission",
  fuel: "Fuel",
  city: "City",
  assembly: "Assembly",
};

let optionsCache = null;
const picked = { transmission: "", fuel: "", assembly: "" };
let predictor = "routed";

async function call(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  return { status: res.status, data };
}

function fillSelect(el, values, blank) {
  el.innerHTML = "";
  if (blank) el.add(new Option(blank, ""));
  (values || []).forEach((value) => el.add(new Option(value, value)));
}

function setPills(container, values, key) {
  container.innerHTML = "";
  const choices = ["", ...(values || [])];
  choices.forEach((value) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pill" + (value === "" ? " is-on" : "");
    button.textContent = value || "Any";
    button.addEventListener("click", () => {
      picked[key] = value;
      container.querySelectorAll(".pill").forEach((pill) => pill.classList.remove("is-on"));
      button.classList.add("is-on");
    });
    container.appendChild(button);
  });
}

function updateModels(preferred) {
  const models = optionsCache.models_by_make[$("make").value] || [];
  fillSelect($("model"), models);
  if (preferred && models.includes(preferred)) $("model").value = preferred;
  paintHeading();
}

function paintHeading() {
  const make = $("make").value || "Your car";
  const model = $("model").value || "";
  $("result-title").textContent = make;
  const year = $("year").value;
  const mileage = Number($("mileage").value || 0).toLocaleString("en-US");
  $("result-sub").textContent = [model, year, mileage ? mileage + " km" : ""].filter(Boolean).join(" · ");
  paintIdlePreview();
}

function paintIdlePreview() {
  const panel = $("result-panel");
  if (!panel || panel.classList.contains("has-result")) return;
  const year = Number($("year").value) || 2010;
  const mileage = Number($("mileage").value) || 0;
  const now = new Date().getFullYear();
  const yearScore = Math.min(1, Math.max(0, (year - 1995) / Math.max(1, now - 1995)));
  const mileScore = 1 - Math.min(1, Math.max(0, mileage / 250000));
  const score = 0.65 * yearScore + 0.35 * mileScore;
  const center = 18 + score * 64;
  const width = 18 + score * 10;
  const mark = $("idle-mark");
  const band = $("idle-band");
  const hint = $("idle-hint");
  if (mark) mark.style.left = "calc(" + center + "% - 2px)";
  if (band) {
    band.style.left = Math.max(6, center - width / 2) + "%";
    band.style.right = Math.max(6, 100 - (center + width / 2)) + "%";
  }
  if (hint) {
    hint.textContent = score > 0.62
      ? "Preview leans newer / lower km"
      : score < 0.38
        ? "Preview leans older / higher km"
        : "Live preview · not a price yet";
  }
  const priceEl = $("result-price");
  const lakhEl = $("result-lakh");
  const priceWrap = priceEl && priceEl.parentElement;
  if (priceWrap) priceWrap.classList.add("is-waiting");
  if (priceEl) priceEl.textContent = "····";
  if (lakhEl) lakhEl.textContent = "Waiting for Estimate price";
}

function setIdleStep(step) {
  document.querySelectorAll("[data-idle-step]").forEach((el) => {
    el.classList.toggle("is-on", Number(el.getAttribute("data-idle-step")) === step);
  });
}

function markResultReady() {
  const panel = $("result-panel");
  if (panel) panel.classList.add("has-result");
  const priceWrap = $("result-price") && $("result-price").parentElement;
  if (priceWrap) priceWrap.classList.remove("is-waiting");
}

function carFromForm() {
  const car = {
    make: $("make").value,
    model: $("model").value,
    year: Number($("year").value),
    mileage: Number($("mileage").value),
    predictor,
  };
  if ($("engine_cc").value) car.engine_cc = Number($("engine_cc").value);
  if ($("city").value) car.city = $("city").value;
  for (const key of ["transmission", "fuel", "assembly"]) {
    if (picked[key]) car[key] = picked[key];
  }
  return car;
}

function setPredictor(next) {
  predictor = next;
  document.querySelectorAll("#predictor-pills .pill").forEach((pill) => {
    pill.classList.toggle("is-on", pill.getAttribute("data-predictor") === next);
  });
  const note = $("predictor-note");
  if (!note) return;
  note.textContent = next === "wsp"
    ? "Experiment: WSP beat blend (try it against the routed champion)."
    : "Production route: LightGBM default + 2010+ specialist.";
}

function showRange(low, high, price) {
  const pad = (high - low) * 0.15 || 1;
  const min = low - pad;
  const span = high + pad - min;
  const pos = (value) => ((value - min) / span) * 100;
  $("range").hidden = false;
  $("range-band").style.left = pos(low) + "%";
  $("range-band").style.right = (100 - pos(high)) + "%";
  $("range-mark").style.left = "calc(" + pos(price) + "% - 1px)";
  $("range-lo").textContent = pkr(low);
  $("range-hi").textContent = pkr(high);
}

function showFactors(factors) {
  if (!factors || !factors.length) {
    $("factors").innerHTML = "";
    return;
  }
  const top = Math.max(...factors.map((factor) => factor.importance), 1e-9);
  $("factors").innerHTML = factors.map((factor) => {
    const width = Math.max(4, (factor.importance / top) * 100);
    const name = factor.feature.replace(/^cat__/, "").replace(/^num__/, "").replaceAll("_", " ");
    return `<div class="factor"><span>${name}</span><div class="track"><span style="width:${width}%"></span></div><b>${factor.importance.toFixed(2)}</b></div>`;
  }).join("");
}

async function showComparison(car, price) {
  const newer = car.year <= new Date().getFullYear() - 3;
  const other = {
    ...car,
    year: newer ? car.year + 3 : Math.max(1980, car.year - 4),
    mileage: newer ? Math.max(5000, Math.round(car.mileage * 0.45)) : Math.min(1000000, Math.round(car.mileage * 1.6)),
  };
  const response = await call("POST", "/predict", other);
  const line = $("compare-line");
  if (response.status !== 200) {
    line.textContent = "";
    return;
  }
  const delta = response.data.estimated_price_pkr - price;
  const direction = delta >= 0 ? "higher" : "lower";
  line.textContent = `A ${other.year} ${car.model} with ${other.mileage.toLocaleString("en-US")} km is about ${pkr(Math.abs(delta))} ${direction}.`;
}

async function estimate(event) {
  event.preventDefault();
  const button = $("estimate-btn");
  button.disabled = true;
  $("form-error").textContent = "";
  try {
    const car = carFromForm();
    const response = await call("POST", "/explain", car);
    if (response.status !== 200) {
      const detail = response.data && response.data.detail
        ? JSON.stringify(response.data.detail)
        : "The estimate did not come back.";
      $("form-error").textContent = response.status === 503
        ? "No model is loaded. Train one, or run the image that contains the model files."
        : detail;
      return;
    }
    const data = response.data;
    setIdleStep(2);
    markResultReady();
    $("result-price").textContent = pkr(data.estimated_price_pkr);
    $("result-lakh").textContent = "about " + lakh(data.estimated_price_pkr);
    $("result-model").textContent = data.selected_model === "wsp_experiment"
      ? "WSP experiment"
      : plainModel(data.selected_model);
    $("result-model").classList.add("ok");
    showRange(data.range_low_pkr, data.range_high_pkr, data.estimated_price_pkr);
    const assumed = (data.assumed_fields || []).map((field) => FIELD_LABEL[field] || field);
    $("assumed").innerHTML = assumed.length
      ? assumed.map((field) => `<span class="chip">${field} assumed</span>`).join("")
      : `<span class="chip ok">Nothing assumed</span>`;
    $("known").textContent = data.known_model ? "Known make and model" : "Uncommon make or model";
    setIdleStep(3);
    showFactors(data.top_factors);
    $("factor-note").textContent = data.note || "";
    await showComparison(car, data.estimated_price_pkr);
  } catch (error) {
    $("form-error").textContent = "The API could not be reached.";
  } finally {
    button.disabled = false;
  }
}

function applyExample(example) {
  $("make").value = example.make;
  updateModels(example.model);
  $("year").value = example.year;
  $("mileage").value = example.mileage;
  paintHeading();
}

async function loadOptions() {
  const health = await call("GET", "/health").catch(() => null);
  const status = $("health-pill");
  if (!health || health.status !== 200 || !health.data.model_loaded) {
    status.className = "status bad";
    status.textContent = health ? "API up, no model loaded" : "API not reachable";
    $("form-error").textContent = health && health.data && !health.data.model_loaded
      ? "Server is running, but models/ was empty on this host. Redeploy with the served .joblib files."
      : "The form fills in once a model is loaded.";
    return;
  }
  status.className = "status ok";
  status.textContent = "Model ready · " + health.data.models_served.length + " loaded";

  const wspPill = $("wsp-pill");
  if (wspPill && health.data.wsp_loaded) {
    wspPill.disabled = false;
    wspPill.title = health.data.wsp && health.data.wsp.beat_champion
      ? "WSP beat the global LightGBM on test MAE"
      : "Try the WSP experiment predictor";
    wspPill.addEventListener("click", () => setPredictor("wsp"));
  }
  document.querySelectorAll("#predictor-pills .pill[data-predictor='routed']").forEach((pill) => {
    pill.addEventListener("click", () => setPredictor("routed"));
  });

  const response = await call("GET", "/options");
  if (response.status !== 200) return;
  optionsCache = response.data;
  const makes = Object.keys(response.data.models_by_make);
  fillSelect($("make"), makes);
  $("make").value = makes.includes("Toyota") ? "Toyota" : makes[0];
  updateModels("Corolla");
  fillSelect($("city"), response.data.city || [], "Not given");
  setPills($("transmission-pills"), response.data.transmission, "transmission");
  setPills($("fuel-pills"), response.data.fuel, "fuel");
  setPills($("assembly-pills"), response.data.assembly, "assembly");

  const box = $("examples");
  EXAMPLES.forEach((example) => {
    const models = response.data.models_by_make[example.make] || [];
    if (!models.includes(example.model)) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rec";
    button.innerHTML = `
      <div class="rec-visual">
        <img src="${example.image}" alt="${example.make} ${example.model}" loading="lazy" width="640" height="360">
      </div>
      <div class="rec-body">
        <span>${example.make}</span>
        <b>${example.model}</b>
        <em>${example.year} · ${example.mileage.toLocaleString("en-US")} km</em>
      </div>`;
    button.addEventListener("click", () => {
      applyExample(example);
      document.getElementById("estimate").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    box.appendChild(button);
  });
}

$("make").addEventListener("change", () => updateModels());
["model", "year", "mileage"].forEach((id) => $(id).addEventListener("input", paintHeading));
document.querySelectorAll("[data-idle-step]").forEach((el) => {
  el.addEventListener("click", () => setIdleStep(Number(el.getAttribute("data-idle-step"))));
});
$("car-form").addEventListener("submit", estimate);
$("year").max = new Date().getFullYear() + 1;
paintIdlePreview();
loadOptions();
