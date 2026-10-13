/* GazeGrounded UX — browser study with WebGazer.js.
 *
 * Privacy: the webcam stream is processed by WebGazer inside this tab only. No video frames or
 * images are stored or transmitted. Only estimated gaze coordinates, clicks, timings and the
 * participant's answers are kept in memory until the participant downloads them.
 *
 * Output format "gazegrounded-study-session" v1 is read by
 * gazegrounded.data.adapters.WebGazerStudyAdapter.
 */
"use strict";

const WEBGAZER_VERSION = "3.5.3";
const FACE_MESH_PATH =
  `https://cdn.jsdelivr.net/npm/webgazer@${WEBGAZER_VERSION}/dist/mediapipe/face_mesh`;
const CALIBRATION_CLICKS = 5;

const state = {
  config: null,
  participant: null,
  sessionId: randomId(),
  group: null,
  plan: [],
  index: 0,
  env: {},
  calibration: { attempts: 0, validations: [] },
  trials: [],
  current: null,
  recording: null, // "trial" | "validation" | null
  validationBuffer: [],
};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

function $(id) { return document.getElementById(id); }

function show(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
  $(id).classList.remove("hidden");
}

function randomId() {
  const a = new Uint8Array(8);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
}

function hashString(s) {
  let h = 2166136261;
  for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(arr, rand) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function aoiAt(ui, nx, ny) {
  // Smallest AOI containing the normalised point (same rule as the Python pipeline).
  let best = null, bestArea = Infinity;
  for (const a of ui.aois) {
    if (nx >= a.x_min && nx <= a.x_max && ny >= a.y_min && ny <= a.y_max) {
      const area = (a.x_max - a.x_min) * (a.y_max - a.y_min);
      if (area < bestArea) { best = a.aoi_id; bestArea = area; }
    }
  }
  return best;
}

// ---------------------------------------------------------------------------------------------
// Consent, code, set-up
// ---------------------------------------------------------------------------------------------

async function loadConfig() {
  const res = await fetch("tasks.json");
  if (!res.ok) throw new Error("tasks.json not found — run `python -m gazegrounded.cli build-study`");
  state.config = await res.json();
}

$("consent-form").addEventListener("submit", (e) => {
  e.preventDefault();
  state.consentedAt = new Date().toISOString();
  show("screen-code");
});

$("code-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const code = $("participant-code").value.trim();
  if (!/^[A-Za-z0-9_-]{2,12}$/.test(code) || code.includes("@")) return;
  state.participant = code;
  const urlGroup = new URLSearchParams(location.search).get("group");
  state.group = urlGroup === "A" || urlGroup === "B" ? urlGroup : (hashString(code) % 2 ? "B" : "A");
  const uis = new Set(state.config.groups[state.group]);
  const tasks = state.config.tasks.filter((t) => uis.has(t.ui_id));
  state.plan = shuffle(tasks, mulberry32(hashString(code + state.config.study_version)));
  state.env = {
    viewport: { width: window.innerWidth, height: window.innerHeight },
    screen: { width: screen.width, height: screen.height },
    device_pixel_ratio: window.devicePixelRatio,
  };
  show("screen-setup");
});

$("btn-start-webgazer").addEventListener("click", async () => {
  $("setup-status").textContent = "Starting camera…";
  try {
    if (document.documentElement.requestFullscreen) {
      await document.documentElement.requestFullscreen().catch(() => {});
    }
    webgazer.params.faceMeshSolutionPath = FACE_MESH_PATH;
    webgazer.saveDataAcrossSessions(false); // never persist calibration data
    webgazer.applyKalmanFilter(true);
    webgazer.setRegression("ridge");
    webgazer.setGazeListener(onGaze);
    await webgazer.begin(() => {
      $("setup-status").textContent = "Camera not available. Please allow camera access.";
    });
    await webgazer.clearData();
    webgazer.showVideoPreview(true);
    webgazer.showFaceOverlay(true);
    webgazer.showFaceFeedbackBox(true);
    webgazer.showPredictionPoints(true);
    state.env.viewport = { width: window.innerWidth, height: window.innerHeight };
    startCalibration();
  } catch (err) {
    $("setup-status").textContent = `Could not start eye tracking: ${err}`;
  }
});

// ---------------------------------------------------------------------------------------------
// Gaze stream
// ---------------------------------------------------------------------------------------------

function onGaze(data) {
  const now = performance.now();
  if (state.recording === "validation") {
    state.validationBuffer.push({ t: now, x: data ? data.x : null, y: data ? data.y : null });
    return;
  }
  if (state.recording !== "trial" || !state.current) return;
  const c = state.current;
  const r = c.rect;
  const sample = { t: Math.round((now - c.onset) * 100) / 100, x: null, y: null, valid: 0 };
  if (data && Number.isFinite(data.x) && Number.isFinite(data.y)) {
    sample.x = Math.round(((data.x - r.left) * c.natural.width / r.width) * 10) / 10;
    sample.y = Math.round(((data.y - r.top) * c.natural.height / r.height) * 10) / 10;
    sample.valid = 1;
  }
  c.gaze.push(sample);
}

// ---------------------------------------------------------------------------------------------
// Calibration and validation
// ---------------------------------------------------------------------------------------------

function startCalibration() {
  state.calibration.attempts += 1;
  show("screen-calibration");
  $("calibration-instructions").classList.remove("hidden");
  const area = $("calibration-area");
  area.innerHTML = "";
  const pos = [0.08, 0.5, 0.92];
  let remaining = 0;
  for (const y of pos) {
    for (const x of pos) {
      const dot = document.createElement("div");
      dot.className = "calib-dot";
      dot.style.left = `${x * 100}%`;
      dot.style.top = `${y * 100}%`;
      let clicks = 0;
      remaining += 1;
      dot.addEventListener("click", () => {
        clicks += 1;
        dot.style.opacity = String(0.35 + 0.13 * clicks);
        if (clicks >= CALIBRATION_CLICKS) {
          dot.classList.add("done");
          remaining -= 1;
          if (remaining === 0) runValidation();
        }
      });
      area.appendChild(dot);
    }
  }
}

async function runValidation() {
  $("calibration-instructions").textContent = "Now just LOOK at each blue dot (no clicking).";
  const area = $("calibration-area");
  area.innerHTML = "";
  webgazer.showPredictionPoints(false);
  const points = [[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];
  const results = [];
  for (const [px, py] of points) {
    const dot = document.createElement("div");
    dot.className = "target-dot";
    dot.style.left = `${px * 100}%`;
    dot.style.top = `${py * 100}%`;
    area.appendChild(dot);
    await new Promise((r) => setTimeout(r, 1000)); // let the eyes settle
    state.validationBuffer = [];
    state.recording = "validation";
    await new Promise((r) => setTimeout(r, 1000));
    state.recording = null;
    const tx = px * window.innerWidth, ty = py * window.innerHeight;
    const valid = state.validationBuffer.filter((s) => s.x !== null);
    const n = state.validationBuffer.length;
    let accuracy = null, precision = null;
    if (valid.length) {
      const mx = valid.reduce((a, s) => a + s.x, 0) / valid.length;
      const my = valid.reduce((a, s) => a + s.y, 0) / valid.length;
      accuracy = Math.hypot(mx - tx, my - ty);
      precision = Math.sqrt(valid.reduce((a, s) => a + (s.x - mx) ** 2 + (s.y - my) ** 2, 0) / valid.length);
    }
    results.push({ target: [Math.round(tx), Math.round(ty)], n_samples: n,
      valid_ratio: n ? valid.length / n : 0, accuracy_px: accuracy, precision_px: precision });
    area.removeChild(dot);
  }
  const acc = results.filter((r) => r.accuracy_px !== null);
  const summary = {
    attempt: state.calibration.attempts,
    mean_accuracy_px: acc.length ? acc.reduce((a, r) => a + r.accuracy_px, 0) / acc.length : null,
    mean_precision_px: acc.length ? acc.reduce((a, r) => a + r.precision_px, 0) / acc.length : null,
    valid_ratio: results.reduce((a, r) => a + r.valid_ratio, 0) / results.length,
    points: results,
  };
  state.calibration.validations.push(summary);
  const s = state.config.settings;
  const ok = summary.mean_accuracy_px !== null && summary.mean_accuracy_px <= s.max_validation_error_px
    && summary.valid_ratio >= s.min_valid_sample_ratio;
  const canRetry = state.calibration.attempts < s.max_calibration_attempts;
  $("validation-text").textContent = summary.mean_accuracy_px === null
    ? "We could not detect your gaze. Please check lighting and that your face is visible."
    : `Average error ≈ ${Math.round(summary.mean_accuracy_px)} px, `
      + `${Math.round(100 * summary.valid_ratio)}% of samples valid. `
      + (ok ? "That is good enough to continue." : "This is less accurate than we hoped.");
  $("btn-recalibrate").classList.toggle("hidden", ok || !canRetry);
  $("btn-begin-tasks").textContent = ok || !canRetry ? "Begin tasks" : "Continue anyway";
  show("screen-validation-result");
}

$("btn-recalibrate").addEventListener("click", async () => {
  await webgazer.clearData();
  webgazer.showPredictionPoints(true);
  $("calibration-instructions").textContent = "Look at each dot and click it 5 times while looking at it.";
  startCalibration();
});

$("btn-begin-tasks").addEventListener("click", () => {
  // From here on the participant should not see themselves or the gaze estimate.
  webgazer.showVideoPreview(false);
  webgazer.showFaceOverlay(false);
  webgazer.showFaceFeedbackBox(false);
  webgazer.showPredictionPoints(false);
  nextInstruction();
});

// ---------------------------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------------------------

function uiById(id) { return state.config.interfaces.find((u) => u.ui_id === id); }

function nextInstruction() {
  if (state.index >= state.plan.length) return finish();
  const task = state.plan[state.index];
  $("instruction-progress").textContent = `Task ${state.index + 1} of ${state.plan.length}`;
  $("instruction-task").textContent = task.description;
  show("screen-instruction");
}

$("btn-show-stimulus").addEventListener("click", () => {
  const task = state.plan[state.index];
  const ui = uiById(task.ui_id);
  const img = $("stimulus");
  img.onload = () => {
    const r = img.getBoundingClientRect();
    state.current = {
      trial_id: `${state.participant}_T${String(state.index + 1).padStart(2, "0")}`,
      order: state.index + 1, task, ui,
      rect: { left: r.left, top: r.top, width: r.width, height: r.height },
      natural: { width: img.naturalWidth, height: img.naturalHeight },
      onset: performance.now(), gaze: [], clicks: [], errors: 0,
    };
    state.recording = "trial";
    state.current.timer = setTimeout(() => endTrial(false, "timeout"),
      state.config.settings.task_timeout_s * 1000);
  };
  $("stimulus-task").textContent = task.description;
  show("screen-stimulus");
  img.src = `${ui.image}?v=${state.config.study_version}`;
});

$("stimulus").addEventListener("click", (e) => {
  const c = state.current;
  if (!c || state.recording !== "trial") return;
  const nx = (e.clientX - c.rect.left) / c.rect.width;
  const ny = (e.clientY - c.rect.top) / c.rect.height;
  const aoi = aoiAt(c.ui, nx, ny);
  const hit = aoi !== null && c.task.target_aoi_ids.includes(aoi);
  c.clicks.push({ t: Math.round(performance.now() - c.onset), x: Math.round(nx * c.natural.width),
    y: Math.round(ny * c.natural.height), aoi_id: aoi, hit_target: hit });
  if (hit) endTrial(true, "target_clicked");
  else c.errors += 1;
});

$("btn-give-up").addEventListener("click", () => endTrial(false, "gave_up"));

function endTrial(success, reason) {
  const c = state.current;
  if (!c || state.recording !== "trial") return;
  state.recording = null;
  clearTimeout(c.timer);
  const valid = c.gaze.filter((s) => s.valid).length;
  state.trials.push({
    trial_id: c.trial_id, order: c.order, task_id: c.task.task_id, ui_id: c.ui.ui_id,
    condition: c.ui.condition, task_description: c.task.description,
    image_rect: c.rect, natural_size: c.natural,
    completion_time_ms: Math.round(performance.now() - c.onset),
    task_success: success, end_reason: reason, error_count: c.errors,
    click_count: c.clicks.length, clicks: c.clicks,
    quality: { n_samples: c.gaze.length, valid_ratio: c.gaze.length ? valid / c.gaze.length : 0 },
    gaze: c.gaze,
  });
  showPostTask();
}

function showPostTask() {
  const seq = $("seq");
  seq.innerHTML = "";
  $("feedback").value = "";
  $("btn-next").disabled = true;
  for (let i = 1; i <= 7; i++) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = String(i);
    b.setAttribute("role", "radio"); b.setAttribute("aria-checked", "false");
    b.addEventListener("click", () => {
      seq.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", "false"));
      b.setAttribute("aria-checked", "true");
      state.trials[state.trials.length - 1].ease_rating = i;
      $("btn-next").disabled = false;
    });
    seq.appendChild(b);
  }
  show("screen-post");
}

$("btn-next").addEventListener("click", () => {
  state.trials[state.trials.length - 1].feedback_text = $("feedback").value.trim() || null;
  state.index += 1;
  state.current = null;
  nextInstruction();
});

// ---------------------------------------------------------------------------------------------
// End and export
// ---------------------------------------------------------------------------------------------

function sessionDocument() {
  return {
    format: "gazegrounded-study-session",
    format_version: 1,
    study_version: state.config.study_version,
    participant_code: state.participant,
    session_id: state.sessionId,
    group: state.group,
    consented_at: state.consentedAt,
    webgazer_version: WEBGAZER_VERSION,
    environment: state.env,
    calibration: state.calibration,
    trials: state.trials,
  };
}

function finish() {
  try { webgazer.end(); } catch (e) { /* camera already stopped */ }
  const ratios = state.trials.map((t) => t.quality.valid_ratio);
  const mean = ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0;
  const last = state.calibration.validations.at(-1);
  $("quality-summary").textContent = `${state.trials.length} tasks · `
    + `${Math.round(100 * mean)}% valid gaze samples · calibration error ≈ `
    + `${last && last.mean_accuracy_px !== null ? Math.round(last.mean_accuracy_px) : "n/a"} px`;
  show("screen-end");
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}

$("btn-download-json").addEventListener("click", () => {
  download(`gazegrounded_${state.participant}_${state.sessionId}.json`,
    JSON.stringify(sessionDocument()), "application/json");
});

$("btn-download-csv").addEventListener("click", () => {
  const rows = ["participant_id,session_id,trial_id,ui_id,timestamp_ms,gaze_x,gaze_y,validity"];
  for (const t of state.trials) {
    for (const s of t.gaze) {
      rows.push([state.participant, state.sessionId, t.trial_id, t.ui_id, s.t,
        s.x ?? "", s.y ?? "", s.valid].join(","));
    }
  }
  download(`gazegrounded_${state.participant}_${state.sessionId}_gaze.csv`, rows.join("\n"),
    "text/csv");
});

loadConfig().catch((err) => {
  document.body.innerHTML = `<p style="padding:24px">Study configuration could not be loaded: ${err}</p>`;
});
