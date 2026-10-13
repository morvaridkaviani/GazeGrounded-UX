/* GazeGrounded UX — project website. Reads data/site.json exported by `gazegrounded.cli export-site`. */
"use strict";

const REPO_URL = "";  // set to the GitHub repository URL
const GAZE_RGB = [235, 104, 52];
const MODEL_RGB = [42, 120, 214];
const TARGET = "#0ca30c";
const MODEL = "#2a78d6";
const GAZE = "#eb6834";
const RF = "#1baf7a";

const $ = (s, root = document) => root.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const human = (s) => String(s ?? "").replace(/_/g, " ");
/** Machine identifiers such as "icon_freeze" become "icon freeze"; normal labels are kept. */
const nice = (s) => (/^[a-z0-9]+(_[a-z0-9]+)+$/.test(String(s ?? "")) ? human(s) : String(s ?? ""));
const short = (s, n = 42) => { const t = String(s ?? "").replace(/\s+/g, " ").trim(); return t.length > n ? t.slice(0, n - 1) + "…" : t; };
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);

let DATA = null;
const byId = {};

// ---------------------------------------------------------------------------------------------
// Images and maps
// ---------------------------------------------------------------------------------------------

const imgCache = new Map();
function loadImage(src) {
  if (!imgCache.has(src)) {
    imgCache.set(src, new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    }));
  }
  return imgCache.get(src);
}

const mapCache = new Map();
/** Colour a greyscale map PNG: transparent at 0, opaque single hue at 1 (display gamma 0.6). */
function colourMap(src, rgb) {
  const key = src + rgb.join(",");
  if (!mapCache.has(key)) {
    mapCache.set(key, loadImage(src).then((img) => {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, c.width, c.height);
      for (let i = 0; i < d.data.length; i += 4) {
        const v = d.data[i] / 255;
        d.data[i] = rgb[0]; d.data[i + 1] = rgb[1]; d.data[i + 2] = rgb[2];
        d.data[i + 3] = Math.round(255 * 0.8 * Math.pow(v, 0.7));  // smooth ramp, no hard edge
      }
      ctx.putImageData(d, 0, 0);
      return c;
    }));
  }
  return mapCache.get(key);
}

// ---------------------------------------------------------------------------------------------
// Viewer: screenshot + layers on a responsive canvas, with hover tooltips
// ---------------------------------------------------------------------------------------------

const tooltip = () => $("#tooltip");
class Viewer {
  constructor(container) {
    this.container = container;
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "img");
    container.appendChild(this.canvas);
    this.spec = null;
    this.hits = [];
    new ResizeObserver(() => this.draw()).observe(container);
    this.canvas.addEventListener("mousemove", (e) => this.hover(e));
    this.canvas.addEventListener("mouseleave", () => tooltip().classList.remove("show"));
  }

  async set(spec) {
    this.spec = spec;
    const token = (this.token = Symbol());
    const [image, ...maps] = await Promise.all([
      loadImage(spec.image),
      ...(spec.maps || []).map((m) => colourMap(m.src, m.rgb)),
    ]);
    if (token !== this.token) return;
    this.image = image;
    this.maps = maps;
    this.canvas.setAttribute("aria-label", spec.alt || "Interface screenshot with overlays");
    this.draw();
  }

  draw() {
    if (!this.spec || !this.image) return;
    const { width: W, height: H } = this.spec;
    const cssW = this.container.clientWidth;
    if (!cssW) return;
    const dpr = window.devicePixelRatio || 1;
    const scale = cssW / W;
    this.scale = scale;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(H * scale * dpr);
    this.canvas.style.height = `${H * scale}px`;
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(this.image, 0, 0, W, H);
    if ((this.maps || []).length || (this.spec.fixations || []).length) {
      ctx.fillStyle = "rgba(255,255,255,0.28)";
      ctx.fillRect(0, 0, W, H);
    }
    for (const m of this.maps || []) ctx.drawImage(m, 0, 0, W, H);
    this.hits = [];
    const px = 1 / scale;  // one CSS pixel in image units

    for (const b of this.spec.boxes || []) {
      const [x0, y0, x1, y1] = [b.box[0] * W, b.box[1] * H, b.box[2] * W, b.box[3] * H];
      ctx.save();
      ctx.lineWidth = (b.strong ? 3 : 1.6) * px;
      ctx.strokeStyle = b.color;
      ctx.setLineDash(b.dash ? [6 * px, 4 * px] : []);
      if (b.fill) { ctx.fillStyle = b.fill; ctx.fillRect(x0, y0, x1 - x0, y1 - y0); }
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.restore();
      this.hits.push({ kind: "box", x0, y0, x1, y1, area: (x1 - x0) * (y1 - y0), tip: b.tip });
      if (b.badge) {
        const r = 10 * px;
        const cx = b.badgeAt === "right" ? x1 : x0, cy = y0;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = b.color; ctx.fill();
        ctx.lineWidth = 2 * px; ctx.strokeStyle = "#fff"; ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.font = `700 ${11 * px}px system-ui, sans-serif`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(b.badge, cx, cy + 0.5 * px);
        this.hits.push({ kind: "badge", cx, cy, r: r * 1.3, tip: b.tip });
      }
    }

    const fx = this.spec.fixations || [];
    if (fx.length && this.spec.scanpath) {
      ctx.save();
      ctx.strokeStyle = "rgba(235,104,52,0.75)"; ctx.lineWidth = 1.6 * px;
      ctx.beginPath();
      fx.forEach((f, i) => (i ? ctx.lineTo(f[0], f[1]) : ctx.moveTo(f[0], f[1])));
      ctx.stroke(); ctx.restore();
    }
    fx.forEach((f, i) => {
      const r = (4 + Math.sqrt(f[2]) * 0.5) * px;
      ctx.beginPath(); ctx.arc(f[0], f[1], r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(235,104,52,0.62)"; ctx.fill();
      ctx.lineWidth = 1.4 * px; ctx.strokeStyle = "#fff"; ctx.stroke();
      if (this.spec.scanpath) {
        ctx.fillStyle = "#fff"; ctx.font = `700 ${10 * px}px system-ui, sans-serif`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(i + 1), f[0], f[1] + 0.5 * px);
      }
      this.hits.push({
        kind: "fix", cx: f[0], cy: f[1], r: Math.max(r, 8 * px),
        tip: `<b>Fixation ${i + 1}</b>at ${(f[3] / 1000).toFixed(2)} s · ${Math.round(f[2])} ms<br>${f[4] ? "AOI: " + esc(human(f[4])) : "outside AOIs"}`,
      });
    });
  }

  hover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / this.scale, y = (e.clientY - rect.top) / this.scale;
    let best = null;
    for (const h of this.hits) {
      if ((h.kind === "fix" || h.kind === "badge") && Math.hypot(x - h.cx, y - h.cy) <= h.r) { best = h; break; }
    }
    if (!best) {
      const boxes = this.hits.filter((h) => h.kind === "box" && h.tip && x >= h.x0 && x <= h.x1 && y >= h.y0 && y <= h.y1);
      boxes.sort((a, b) => a.area - b.area);
      best = boxes[0];
    }
    const tt = tooltip();
    if (best && best.tip) {
      tt.innerHTML = best.tip;
      tt.classList.add("show");
      const pad = 14;
      let left = e.clientX + pad, top = e.clientY + pad;
      if (left + 300 > window.innerWidth) left = e.clientX - 300 - pad;
      tt.style.left = `${Math.max(8, left)}px`; tt.style.top = `${top}px`;
    } else tt.classList.remove("show");
  }
}

function modelBoxes(task, k = 3) {
  return task.model.elements.slice(0, k).map((e, i) => ({
    box: e.box, color: MODEL, strong: i === 0, badge: String(i + 1),
    fill: i === 0 ? "rgba(42,120,214,0.08)" : null, dash: i > 0,
    tip: `<b>Model #${i + 1}: ${esc(short(nice(e.label), 60))}</b>score ${fmt(e.score)} · relevance ${fmt(e.relevance)}<br>${esc(short(e.reason, 120))}`,
  }));
}
function targetBoxes(task) {
  const ui = byId.ui[task.ui];
  return ui.aois.filter((a) => task.targets.includes(a.id)).map((a) => ({
    box: a.box, color: TARGET, dash: true, strong: true, badge: "T", badgeAt: "right",
    fill: "rgba(12,163,12,0.07)", tip: `<b>Task target</b>${esc(a.label || human(a.id))}`,
  }));
}

// ---------------------------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------------------------

const CONDITION_BADGE = { clean: "green", friction: "orange", single: "", free_viewing: "blue" };
const FRICTION_BADGE = { low: "green", medium: "orange", high: "red" };
const uiLabel = (ui) => `${ui.domain ? ui.domain[0].toUpperCase() + ui.domain.slice(1) + " · " : ""}${ui.name}`;
const badge = (text, cls = "") => `<span class="badge ${cls}">${esc(text)}</span>`;
const metric = (label, value, help, unit = "") =>
  `<div class="metric"><span>${esc(label)}${help ? ` <abbr class="help" title="${esc(help)}">?</abbr>` : ""}</span><b>${value}${unit ? ` <small>${unit}</small>` : ""}</b></div>`;

function fillSelect(sel, items, value) {
  sel.innerHTML = items.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join("");
  if (value !== undefined && items.some(([v]) => v === value)) sel.value = value;
}

function segmented(container, options, value, onChange) {
  container.innerHTML = options.map(([v, t]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${v === value}">${esc(t)}</button>`).join("");
  container.onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    container.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    onChange(b.dataset.v);
  };
}

const FEATURE_NAMES = {
  fixation_count: "Fixation count", completion_time_ms: "Completion time", ttff_censored_ms: "Time to first look at target",
  revisit_count: "Revisits", aoi_transition_count: "AOI transitions", mean_fixation_duration_ms: "Mean fixation duration",
  fixation_rate_per_s: "Fixation rate", scanpath_length_norm: "Scanpath length", task_relevant_attention: "Attention on target",
  task_relevant_aoi_coverage: "Target coverage", target_fixated: "Target fixated", click_count: "Clicks", error_count: "Errors",
  task_success: "Task success", vlm_human_similarity: "Gaze–model similarity", attention_divergence: "Divergence (whole trial)",
  attention_divergence_early: "Early divergence", aoi_attention_divergence: "AOI divergence",
  vlm_relevant_dwell_share: "Dwell on model-relevant regions", task_relevant_region_count: "Model: relevant regions",
  max_task_relevance: "Model: max relevance", mean_task_relevance: "Model: mean relevance", has_feedback: "Has feedback", text: "Feedback text",
};
const MODALITY_NAMES = { interaction: "Interaction", gaze: "Gaze", vlm: "Model relevance", text: "Feedback text", visual: "Visual features", multimodal: "All combined" };
const modalityName = (m) => MODALITY_NAMES[m] || human(m);
const featureName = (f) => FEATURE_NAMES[f] || (f.startsWith("text_p_") ? `Feedback: ${human(f.slice(7))}` : human(f));

// ---------------------------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------------------------

const LAYERS = [["gaze", "Gaze heatmap"], ["fix", "Fixations"], ["scan", "Scanpath"], ["modelmap", "Model relevance"], ["boxes", "Model elements"], ["target", "Task target"]];
const explorerState = { layers: new Set(["gaze", "target"]) };

function initExplorer() {
  const viewer = new Viewer($("#viewer-explorer"));
  const selUi = $("#sel-ui"), selTask = $("#sel-task"), selP = $("#sel-participant");
  fillSelect(selUi, DATA.uis.map((u) => [u.id, `${uiLabel(u)} (${human(u.condition)})`]), "shop_checkout_friction");
  const layerBox = $("#layers");
  layerBox.innerHTML = LAYERS.map(([k, t]) => `<button type="button" class="chip" data-k="${k}" aria-pressed="${explorerState.layers.has(k)}">${t}</button>`).join("");
  layerBox.onclick = (e) => {
    const b = e.target.closest(".chip"); if (!b) return;
    const k = b.dataset.k;
    explorerState.layers.has(k) ? explorerState.layers.delete(k) : explorerState.layers.add(k);
    b.setAttribute("aria-pressed", explorerState.layers.has(k));
    render();
  };

  function syncTasks() {
    const tasks = DATA.tasks.filter((t) => t.ui === selUi.value);
    fillSelect(selTask, tasks.map((t) => [t.id, t.description]), selTask.value);
    syncParticipants();
  }
  function syncParticipants() {
    const ps = DATA.trials.filter((t) => t.task === selTask.value).map((t) => t.participant).sort();
    fillSelect(selP, ps.map((p) => [p, p]), selP.value);
    render();
  }
  function render() {
    const trial = DATA.trials.find((t) => t.task === selTask.value && t.participant === selP.value);
    const task = byId.task[selTask.value], ui = byId.ui[selUi.value];
    if (!trial) return;
    const L = explorerState.layers;
    const maps = [];
    if (L.has("modelmap")) maps.push({ src: task.modelMap, rgb: MODEL_RGB });
    if (L.has("gaze")) maps.push({ src: trial.heatmap, rgb: GAZE_RGB });
    viewer.set({
      image: ui.image, width: ui.width, height: ui.height, maps,
      fixations: L.has("fix") || L.has("scan") ? trial.fixations : [], scanpath: L.has("scan"),
      boxes: [...(L.has("boxes") ? modelBoxes(task) : []), ...(L.has("target") ? targetBoxes(task) : [])],
      alt: `Screenshot of ${uiLabel(ui)} with gaze and model overlays for participant ${trial.participant}`,
    });
    renderTrialPanel(trial, task, ui);
  }
  selUi.onchange = syncTasks; selTask.onchange = syncParticipants; selP.onchange = render;
  syncTasks();
}

function renderTrialPanel(trial, task, ui) {
  const match = trial.predicted && trial.predicted === trial.label;
  const contrib = trial.contributions || [];
  const maxAbs = Math.max(0.01, ...contrib.map((c) => Math.abs(c[1])));
  $("#trial-panel").innerHTML = `
    <p class="task-title">“${esc(task.description)}”</p>
    <div class="badges">
      ${badge(uiLabel(ui))}${badge(human(trial.condition) + " variant", CONDITION_BADGE[trial.condition])}
      ${badge(trial.participant, "blue")}${trial.success ? badge("completed", "green") : badge("failed", "red")}
    </div>
    <div class="metrics">
      ${metric("First look at target", trial.ttffS === null ? "never" : fmt(trial.ttffS, 1), "Time to first fixation on the task-relevant AOI", trial.ttffS === null ? "" : "s")}
      ${metric("Completion time", fmt(trial.completionS, 1), "", "s")}
      ${metric("Fixations", trial.fixationCount)}
      ${metric("Revisits", trial.revisits, "Returns to an AOI after looking elsewhere")}
      ${metric("Gaze–model similarity", fmt(trial.similarity), "Overlap of the gaze map and the model map (1 = identical)")}
      ${metric("Early divergence", fmt(trial.divergenceEarly), "Jensen–Shannon divergence between the first fixations and the model map (0 = same, 1 = no overlap)")}
    </div>
    <div class="panel-title">Friction level</div>
    <div class="badges">
      ${badge("label: " + (trial.label || "n/a"), FRICTION_BADGE[trial.label])}
      ${badge("predicted: " + (trial.predicted || "n/a"), FRICTION_BADGE[trial.predicted])}
      ${trial.predicted ? badge(match ? "match" : "mismatch", match ? "green" : "red") : ""}
    </div>
    <div class="panel-title">Participant feedback</div>
    ${trial.feedback ? `<blockquote>${esc(trial.feedback)}</blockquote>` : `<p style="color:var(--muted);margin:0 0 8px">No feedback given.</p>`}
    ${trial.nlp ? `<div class="badges">${badge("text category: " + human(trial.nlp.category), "blue")}${badge("confidence " + fmt(trial.nlp.confidence))}</div>` : ""}
    ${contrib.length ? `<div class="panel-title">Why this prediction</div>
    <div class="bars diverge">${contrib.map(([f, v]) => {
      const w = (Math.abs(v) / maxAbs) * 50;
      return `<div class="bar-row" title="${esc(featureName(f))}: ${fmt(v)}"><span class="name">${esc(featureName(f))}</span>
        <span class="bar-track"><span class="bar-fill" style="${v >= 0 ? `left:50%;width:${w}%;background:#e34948` : `right:50%;width:${w}%;background:${MODEL}`}"></span></span>
        <span class="val">${v >= 0 ? "+" : ""}${fmt(v)}</span></div>`;
    }).join("")}</div>
    <p style="font-size:12px;color:var(--muted);margin:8px 0 0">Red pushes towards the predicted level, blue away from it. Model trained without this participant.</p>` : ""}`;
}

function initCompare() {
  const vh = new Viewer($("#viewer-human")), vm = new Viewer($("#viewer-model"));
  const sel = $("#sel-compare");
  fillSelect(sel, DATA.tasks.map((t) => [t.id, `${uiLabel(byId.ui[t.ui])} (${human(byId.ui[t.ui].condition)}) — ${t.description}`]), "shop_checkout_friction__change_address");
  function render() {
    const task = byId.task[sel.value], ui = byId.ui[task.ui];
    const base = { image: ui.image, width: ui.width, height: ui.height };
    vh.set({ ...base, maps: [{ src: task.humanMap, rgb: GAZE_RGB }], boxes: targetBoxes(task), alt: "Pooled human gaze heatmap" });
    vm.set({ ...base, maps: [{ src: task.modelMap, rgb: MODEL_RGB }], boxes: [...modelBoxes(task), ...targetBoxes(task)], alt: "Model relevance map" });
    const m = task.metrics;
    $("#compare-metrics").innerHTML =
      metric("Correlation (CC)", fmt(m.cc), "Pearson correlation of the two maps; higher = closer") +
      metric("Overlap (SIM)", fmt(m.sim), "Histogram intersection; 1 = identical") +
      metric("KL (human ‖ model)", fmt(m.kl), "Lower = the model covers where people looked") +
      metric("Divergence (JSD)", fmt(m.jsd), "0 = same distribution, 1 = no overlap");
  }
  sel.onchange = render;
  render();

  const base = DATA.results.baselines || [];
  const draw = (metricKey) => {
    const lowerBetter = metricKey === "kl";
    const rows = [...base].sort((a, b) => (lowerBetter ? 1 : -1) * (a[`${metricKey}_mean`] - b[`${metricKey}_mean`]));
    const vals = rows.flatMap((r) => [r[`${metricKey}_min`], r[`${metricKey}_max`]]);
    const lo = Math.min(0, ...vals);
    const raw = (Math.max(...vals) - lo) / 4, mag = 10 ** Math.floor(Math.log10(raw)), nrm = raw / mag;
    const step = (nrm < 1.5 ? 1 : nrm < 3 ? 2 : nrm < 7 ? 5 : 10) * mag;
    const hi = Math.ceil(Math.max(...vals) / step) * step;
    const W = 760, rowH = 40, left = 190, right = 70, H = rows.length * rowH + 30;
    const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
    const ticks = []; for (let t = lo; t <= hi + 1e-9; t += step) ticks.push(t);
    $("#baseline-chart").innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Baseline comparison">
      ${ticks.map((t) => `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H - 24}"/><text x="${x(t)}" y="${H - 6}" text-anchor="middle">${t.toFixed(2)}</text>`).join("")}
      ${rows.map((r, i) => {
        const y = i * rowH + 20, isModel = r.baseline.startsWith("D");
        const label = r.baseline.replace(/^[A-D] · /, "");
        return `<g><title>${esc(label)}: mean ${fmt(r[`${metricKey}_mean`])}, range ${fmt(r[`${metricKey}_min`])}–${fmt(r[`${metricKey}_max`])}, n = ${r.n} tasks</title>
          <text x="${left - 12}" y="${y + 4}" text-anchor="end" style="font-weight:${isModel ? 700 : 400}">${esc(label)}</text>
          <line x1="${x(r[`${metricKey}_min`])}" x2="${x(r[`${metricKey}_max`])}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-width="8" stroke-linecap="round"/>
          <circle cx="${x(r[`${metricKey}_mean`])}" cy="${y}" r="8" fill="${isModel ? MODEL : "var(--ink-2)"}" stroke="var(--surface)" stroke-width="2"/>
          <text x="${x(r[`${metricKey}_max`]) + 12}" y="${y + 4}">${fmt(r[`${metricKey}_mean`])} <tspan fill="var(--muted)">n=${r.n}</tspan></text></g>`;
      }).join("")}</svg>`;
  };
  segmented($("#baseline-metric"), [["cc", "Correlation"], ["sim", "Overlap"], ["nss", "NSS"], ["kl", "KL ↓"]], "cc", draw);
  draw("cc");
}

function initModels() {
  const rows = DATA.results.modalities || [];
  const draw = (split) => {
    const r = rows.filter((x) => x.split === split);
    const chance = (r.find((x) => x.model === "majority") || {}).macroF1Mean;
    const mods = [...new Set(r.filter((x) => x.model !== "majority").map((x) => x.modality))];
    const models = [["logistic_regression", "Logistic regression", MODEL], ["random_forest", "Random forest", RF]];
    $("#modality-chart").innerHTML = `<div class="bars">${mods.map((m) => `
      <div class="modality-row">
        <span class="name" style="font-size:14px;font-weight:${m === "multimodal" ? 700 : 400}">${esc(modalityName(m))}</span>
        <div style="display:grid;gap:3px">${models.map(([key, name, color]) => {
          const x = r.find((y) => y.modality === m && y.model === key); if (!x) return "";
          return `<div class="bar-row" style="grid-template-columns:1fr 46px" title="${name} · ${modalityName(m)}: ${fmt(x.macroF1Mean)} ± ${fmt(x.macroF1Sd)} (mean ± SD over folds)">
            <span class="bar-track">${chance ? `<span style="position:absolute;left:${chance * 100}%;top:-4px;bottom:-4px;border-left:2px dashed var(--muted)"></span>` : ""}
              <span class="bar-fill" style="left:0;width:${x.macroF1Mean * 100}%;background:${color}"></span></span>
            <span class="val">${fmt(x.macroF1Mean)}</span></div>`;
        }).join("")}</div></div>`).join("")}</div>`;
    $("#modality-legend").innerHTML = models.map(([, n, c]) => `<span><span class="dot" style="background:${c}"></span>${n}</span>`).join("") +
      (chance ? `<span><span style="display:inline-block;width:14px;border-top:2px dashed var(--muted)"></span>majority-class reference (${fmt(chance)})</span>` : "");
  };
  segmented($("#split-toggle"), [["participant_id", "New participants"], ["ui_id", "New interfaces"]], "participant_id", draw);
  draw("participant_id");

  const full = rows.find((x) => x.split === "participant_id" && x.model === "logistic_regression" && x.modality === "multimodal");
  $("#ablation").innerHTML = `<table><thead><tr><th>Without</th><th class="num">Macro-F1</th><th class="num">Change</th></tr></thead><tbody>${
    (DATA.results.ablation || []).map((a) => {
      let d = full ? a.macroF1 - full.macroF1 : null;
      if (d !== null && Math.abs(d) < 0.005) d = 0;
      const txt = d === null ? "—" : d === 0 ? "0.00" : (d > 0 ? "+" : "") + fmt(d);
      return `<tr><td>${esc(modalityName(a.without))}</td><td class="num">${fmt(a.macroF1)}</td><td class="num" style="color:${d < 0 ? "var(--bad)" : "var(--ink-2)"}">${txt}</td></tr>`;
    }).join("")}</tbody></table>`;

  const cm = DATA.results.confusion;
  if (cm) {
    const n = cm.labels.length;
    const cells = [`<div class="h"></div>`, ...cm.labels.map((l) => `<div class="h">${esc(l)}</div>`)];
    cm.matrix.forEach((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      cells.push(`<div class="h" style="text-align:right">${esc(cm.labels[i])}</div>`);
      row.forEach((v) => {
        const share = v / tot;
        cells.push(`<div style="background:rgba(42,120,214,${0.08 + share * 0.85});color:${share > 0.55 ? "#fff" : "var(--ink)"}" title="${v} trials (${pct(share)} of this label)">${v}</div>`);
      });
    });
    $("#confusion").innerHTML = `<div class="cm" style="grid-template-columns:70px repeat(${n},1fr)">${cells.join("")}</div>`;
  }

  const imp = DATA.results.importance || [];
  const maxImp = Math.max(0.001, ...imp.map((x) => x[1]));
  $("#importance").innerHTML = `<div class="bars">${imp.slice(0, 8).map(([f, v]) => `
    <div class="bar-row" style="grid-template-columns:130px 1fr 40px" title="${esc(featureName(f))}: ${fmt(v, 3)}">
      <span class="name">${esc(featureName(f))}</span>
      <span class="bar-track"><span class="bar-fill" style="left:0;width:${Math.max(0, v / maxImp) * 100}%;background:${MODEL}"></span></span>
      <span class="val">${fmt(v)}</span></div>`).join("")}</div>`;
}

function initFailures() {
  const list = $("#failure-list");
  const fails = DATA.results.failures || [];
  if (!fails.length) { list.innerHTML = `<div class="card"><p>No failure cases: the model's top element matched the target for every task.</p></div>`; return; }
  list.innerHTML = "";
  for (const f of fails) {
    const task = byId.task[f.task], ui = byId.ui[task.ui];
    const top = task.model.elements[0];
    const targetLabels = ui.aois.filter((a) => task.targets.includes(a.id)).map((a) => a.label || human(a.id)).join(", ");
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <p class="task-title">“${esc(task.description)}”</p>
      <div class="badges">${badge(uiLabel(ui))}${badge(human(ui.condition) + " variant", CONDITION_BADGE[ui.condition])}
        ${task.designed && task.designed !== "no_friction" ? badge("designed: " + human(task.designed), "violet") : ""}</div>
      <div class="case">
        <div><div class="viewer"></div>
          <div class="legend"><span><span class="dot" style="background:var(--gaze)"></span>Where people looked</span>
          <span><span class="dot" style="background:var(--model)"></span>Model's top 3 (numbered)</span>
          <span><span class="dot" style="background:var(--target)"></span>Task target (T)</span></div></div>
        <div>
          <div class="panel-title" style="margin-top:0">What went wrong</div>
          <p class="verdict">The model ranked <strong>${esc(short(nice(top.label)))}</strong> first (score ${fmt(top.score)}), but the task needs <strong>${esc(targetLabels)}</strong>.</p>
          <div class="panel-title">Where people actually looked</div>
          <div class="bars">
            <div class="bar-row" style="grid-template-columns:130px 1fr 46px"><span class="name">Task target</span><span class="bar-track"><span class="bar-fill" style="left:0;width:${(f.targetShare || 0) * 100}%;background:${TARGET}"></span></span><span class="val">${pct(f.targetShare)}</span></div>
            <div class="bar-row" style="grid-template-columns:130px 1fr 46px"><span class="name">Model's top choice</span><span class="bar-track"><span class="bar-fill" style="left:0;width:${(f.modelTopShare || 0) * 100}%;background:${MODEL}"></span></span><span class="val">${pct(f.modelTopShare)}</span></div>
          </div>
          <p style="font-size:13px;color:var(--muted);margin:8px 0 0">${(f.targetShare ?? 0) >= (f.modelTopShare ?? 0)
            ? "People attended to the target more than to the model's choice — a likely model error rather than a design problem."
            : "People also looked at the model's choice — it may be a genuine distractor in the design."}</p>
          <div class="panel-title">Model ranking <span style="font-weight:400;color:var(--muted)">score = relevance × attention priority</span></div>
          <table><thead><tr><th>Element</th><th class="num">Score</th><th class="num">Target</th></tr></thead><tbody>
            ${task.model.elements.slice(0, 4).map((e, i) => `<tr><td>${i + 1}. ${esc(short(nice(e.label), 30))}</td><td class="num">${fmt(e.score)}</td><td class="num">${task.targets.includes(e.matched) ? "✓" : ""}</td></tr>`).join("")}
          </tbody></table>
        </div>
      </div>`;
    list.appendChild(card);
    new Viewer(card.querySelector(".viewer")).set({
      image: ui.image, width: ui.width, height: ui.height, maps: [{ src: task.humanMap, rgb: GAZE_RGB }],
      boxes: [...modelBoxes(task), ...targetBoxes(task)], alt: `Failure case: ${task.description}`,
    });
  }
}

function initChrome() {
  const m = DATA.meta;
  if (m.notice) $("#data-notice").hidden = false;
  const who = m.dataKind === "synthetic_demo" ? "simulated participants" : "participants";
  $("#stats").innerHTML = [[m.interfaces, "interfaces"], [m.tasks, "tasks"], [m.participants, who], [m.trials, "trials"]]
    .map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
  $("#year").textContent = new Date().getFullYear();
  if (REPO_URL) {
    ["#repo-link", "#repo-link-2"].forEach((s) => ($(s).href = REPO_URL));
    $("#clone-cmd").textContent = `git clone ${REPO_URL}.git`;
  } else {
    ["#repo-link", "#repo-link-2"].forEach((s) => ($(s).hidden = true));
  }
  document.querySelectorAll(".copy").forEach((b) => b.addEventListener("click", async () => {
    const text = b.parentElement.querySelector("pre").innerText.replace(/\s+#.*$/gm, "");
    try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; } catch { b.textContent = "Select & copy"; }
    setTimeout(() => (b.textContent = "Copy"), 1500);
  }));
  // Theme toggle (remembered per browser when storage is available).
  const root = document.documentElement;
  try { const saved = localStorage.getItem("gg-theme"); if (saved) root.dataset.theme = saved; } catch { /* storage unavailable */ }
  $("#theme-toggle").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("gg-theme", root.dataset.theme); } catch { /* ignore */ }
  });
  // Highlight the section in view.
  const links = [...document.querySelectorAll(".nav a")];
  const obs = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === `#${e.target.id}`));
  }), { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("section.block").forEach((s) => obs.observe(s));
}

async function main() {
  try {
    const res = await fetch("data/site.json");
    if (!res.ok) throw new Error(res.statusText);
    DATA = await res.json();
  } catch (err) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="container loading">Could not load data/site.json (${esc(err.message)}). Serve this folder over HTTP, e.g. <code>python -m http.server</code>.</div>`);
    return;
  }
  byId.ui = Object.fromEntries(DATA.uis.map((u) => [u.id, u]));
  byId.task = Object.fromEntries(DATA.tasks.map((t) => [t.id, t]));
  initChrome();
  initExplorer();
  initCompare();
  initModels();
  initFailures();
}
main();
