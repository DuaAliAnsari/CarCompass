const NAV = [
  ["/", "Estimate"],
  ["/experiments", "Experiments"],
  ["/system", "Models"],
  ["/pipeline", "Pipeline"],
  ["/demo", "Demo"],
];

const path = location.pathname.replace(/\/$/, "") || "/";

const nav = document.createElement("header");
nav.className = "site-nav";
nav.innerHTML = `
  <nav class="nav-left" aria-label="Primary">
    ${NAV.map(([href, label]) =>
      `<a href="${href}" data-nav="${href}">${label}</a>`).join("")}
  </nav>
  <a class="site-brand" href="/">CarCompass</a>
  <div class="nav-right">
    <span id="nav-health" class="status">Checking…</span>
    <a class="btn" href="${path === "/" ? "#estimate" : "/"}">Get a price →</a>
  </div>
`;
document.body.prepend(nav);
document.querySelectorAll(".nav, .topbar, .rail").forEach((node) => node.remove());
document.body.classList.remove("with-rail");

nav.querySelectorAll("[data-nav]").forEach((link) => {
  if (link.dataset.nav === path) link.classList.add("is-active");
});

fetch("/health")
  .then((res) => res.json())
  .then((data) => {
    const el = document.getElementById("nav-health");
    if (!el) return;
    if (data.model_loaded) {
      el.className = "status ok";
      el.textContent = "Model ready";
    } else {
      el.className = "status bad";
      el.textContent = "No model loaded";
    }
  })
  .catch(() => {
    const el = document.getElementById("nav-health");
    if (el) {
      el.className = "status bad";
      el.textContent = "API offline";
    }
  });

function pkr(n) {
  return "PKR " + Math.round(n).toLocaleString("en-US");
}

function lakh(n) {
  return (n / 1e5).toFixed(1) + " lakh";
}

function plainModel(name) {
  if (!name) return "—";
  if (name.includes("2010plus") || name.includes("2010")) return "2010 and newer";
  if (name.includes("pre2010") || name.includes("old")) return "Older cars";
  if (name.includes("global")) return "All years";
  if (name.includes("dataset_a")) return "Dataset A only";
  if (name.includes("dataset_b")) return "Dataset B only";
  return name.replaceAll("_", " ");
}
