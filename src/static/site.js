const NAV = [
  ["/", "Estimate"],
  ["/experiments", "Experiments"],
  ["/system", "Models"],
];

const GUIDE_PATHS = new Set([
  "/guide",
  "/guide/milestones",
  "/guide/pipeline",
  "/guide/wsp",
  "/guide/checks",
  "/guide/monitor",
  "/milestones",
  "/stats",
  "/metrics",
]);

const path = location.pathname.replace(/\/$/, "") || "/";
const onGuide = GUIDE_PATHS.has(path) || path.startsWith("/guide/");
const onProduct = path === "/" || path === "/experiments" || path === "/system";
if (onGuide || path === "/wsp" || path === "/pipeline" || path === "/demo") {
  document.body.classList.add("guide-shell");
}
if (onProduct) {
  document.body.classList.add("product-shell");
}

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

if (onGuide || path === "/wsp" || path === "/pipeline" || path === "/demo") {
  const sub = document.createElement("nav");
  sub.className = "guide-subnav";
  sub.setAttribute("aria-label", "Project guide");
  const links = [
    ["/guide", "Overview"],
    ["/guide/milestones", "Milestones"],
    ["/guide/pipeline", "Pipeline"],
    ["/guide/monitor", "Monitoring"],
    ["/guide/checks", "Checks"],
    ["/guide/wsp", "WSP"],
  ];
  sub.innerHTML = links.map(([href, label]) => {
    const active =
      path === href ||
      (href === "/guide/milestones" && path === "/milestones") ||
      (href === "/guide/pipeline" && path === "/pipeline") ||
      (href === "/guide/wsp" && path === "/wsp") ||
      (href === "/guide/checks" && path === "/demo") ||
      (href === "/guide/monitor" && (path === "/stats" || path === "/metrics"));
    return `<a href="${href}" class="${active ? "is-active" : ""}">${label}</a>`;
  }).join("");
  nav.after(sub);
}

document.querySelectorAll("footer.foot").forEach((old) => old.remove());
const foot = document.createElement("footer");
foot.className = "foot site-foot";
foot.innerHTML = `
  <div class="foot-panel">
    <div class="foot-copy">
      <p class="foot-disclaimer">Estimates reflect historical asking prices. They are a guide, not a valuation.</p>
      <p class="foot-meta-line">
        <span>CarCompass</span>
        <span class="foot-dot" aria-hidden="true"></span>
        <span>MLOps · Fall 2026</span>
      </p>
    </div>
    <a class="foot-more-btn foot-guide-btn" href="/guide">
      Project guide
      <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 8h9M8 3l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </a>
  </div>
`;
const main = document.querySelector("main.wrap") || document.body;
main.appendChild(foot);

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
      el.title = "API started, but models/manifest.json or *.joblib was missing on this host";
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
