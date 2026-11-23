const $ = (id) => document.getElementById(id);

function years(value) {
  if (!value || typeof value === "string") return value || "All years";
  if (value.year_min) return value.year_min + " and later";
  if (value.year_max) return "up to " + value.year_max;
  return "All years";
}

async function loadSystem() {
  const health = await fetch("/health").then((res) => res.json()).catch(() => null);
  const pill = $("health-pill");
  if (health && health.model_loaded) {
    pill.className = "status ok";
    pill.textContent = health.models_served.length + " models loaded · " + (health.app_version || "dev");
  } else {
    pill.className = "status bad";
    pill.textContent = health ? "API up, no model loaded" : "API not reachable";
  }

  const response = await fetch("/models").catch(() => null);
  if (!response || response.status !== 200) {
    $("live-models").innerHTML = "<tr><td>Start the API with the model files to see the loaded set.</td></tr>";
    return;
  }
  const body = await response.json();
  const rows = Object.entries(body.models || {}).map(([name, model]) => {
    const mark = name === body.default ? " default" : "";
    return `<tr><td>${plainModel(name)}${mark}</td><td>${years(model.years)}</td><td>${model.run_name || "—"}</td><td class="num">${pkr(model.test_mae_pkr || 0)}</td></tr>`;
  });
  $("live-models").innerHTML = "<tr><th>Model</th><th>Years</th><th>Run</th><th class='num'>Test MAE</th></tr>" + rows.join("");
}

loadSystem();
