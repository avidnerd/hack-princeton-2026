// Spoonful UI: one WebSocket for state (10 Hz), POSTs for every action. No build step.
"use strict";
const $ = (id) => document.getElementById(id);
let S = null;            // latest snapshot
let pick = null;         // calibration click mode: label string, or "window"
let winStart = null;

const REASON_TEXT = {
  "": "Clear to approach",
  no_face: "No one in view",
  face_too_brief: "Checking you're there",
  looking_away: "Waiting for you to face it",
  mouth_closed: "Open when you're ready",
  mouth_outside_window: "Move a little toward the centre",
  frame_stale: "Camera lagging",
  paused: "Paused",
  stop_latched: "Stopped",
};
const STATE_TEXT = {
  BOOT: "Starting", HOMING: "Moving home", IDLE: "Idle", LISTENING: "Ready for a bite",
  SCOOPING: "Scooping", LIFTING: "Lifting", PRE_DELIVERY_HOLD: "Waiting for you", DELIVERING: "Coming to you",
  MOUTH_HOLD: "Holding still", RETREAT: "Moving away", PAUSED: "Paused", FAULT: "Taking a break", STOPPED: "STOPPED",
};

async function post(path, body = {}, method = "POST") {
  try {
    const r = await fetch("/api/" + path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) toast(j.detail || r.statusText);
    return j;
  } catch (e) { toast("server unreachable"); return {}; }
}
async function get(path) { const r = await fetch("/api/" + path); return r.json(); }
let toastUntil = 0;
function toast(msg) { const b = $("fault-banner"); b.textContent = msg; b.classList.remove("hidden"); toastUntil = Date.now() + 4000; }

// ------------------------------------------------------------------ websocket
function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { $("cam").src = "/stream?t=" + Date.now(); };   // re-attach the MJPEG stream after a restart
  ws.onmessage = (e) => { S = JSON.parse(e.data); render(); };
  ws.onclose = () => { $("verdict").textContent = "Reconnecting"; $("verdict").className = "verdict stop"; setTimeout(connect, 1000); };
}

function render() {
  if (!S) return;
  const f = S.fsm, st = f.state;
  $("state-badge").textContent = STATE_TEXT[st] || st;
  const v = $("verdict");
  if (st === "STOPPED") { v.textContent = "STOPPED"; v.className = "verdict stop"; }
  else if (st === "FAULT") { v.textContent = "Taking a break"; v.className = "verdict stop"; }
  else if (["PRE_DELIVERY_HOLD", "DELIVERING", "MOUTH_HOLD"].includes(st)) {
    v.textContent = REASON_TEXT[f.gate.reason] ?? f.gate.reason; v.className = "verdict " + (f.gate.ok ? "ok" : "wait");
  } else { v.textContent = STATE_TEXT[st] || st; v.className = "verdict"; }
  $("explain").textContent = f.explain;
  const cd = $("countdown");
  if (f.countdown != null) { cd.textContent = f.countdown.toFixed(1) + " s"; cd.classList.remove("hidden"); } else cd.classList.add("hidden");

  const banner = $("fault-banner");
  if (st === "FAULT") { banner.textContent = "Fault: " + f.fault + ". Arm parked. Check it, then Reset."; banner.classList.remove("hidden"); }
  else if (st === "STOPPED") { banner.textContent = f.stop_release_armed ? "Button pressed. Press Reset to confirm."
      : "Stopped. Press the physical STOP button (or spacebar), then Reset."; banner.classList.remove("hidden"); }
  else if (f.calibration_mode) { banner.textContent = "CALIBRATION MODE: nobody in the chair"; banner.classList.remove("hidden"); }
  else if (Date.now() > toastUntil) banner.classList.add("hidden");

  $("btn-reset").classList.toggle("hidden", !(st === "FAULT" || st === "STOPPED"));
  $("btn-pause").textContent = st === "PAUSED" ? "RESUME" : "PAUSE";
  $("btn-bite").disabled = st !== "LISTENING";
  $("btn-session").textContent = f.session.active ? "End session" : "Start session";
  $("bites").textContent = f.session.active ? f.session.taken : (S.fsm.session.bites || 0);
  $("pace").textContent = S.pace.label;

  const sel = $("profile");
  if (sel.options.length !== S.profiles.length) {
    sel.innerHTML = S.profiles.map((p) => `<option>${p}</option>`).join("");
  }
  if (document.activeElement !== sel) sel.value = S.profile.name;

  $("timeline").innerHTML = (S.timeline || []).slice().reverse().map((e) => {
    const t = new Date(e.ts * 1000).toLocaleTimeString([], { hour12: false });
    let txt;
    if (e.kind === "fault") txt = `<span class="fault">fault ${esc(compact(e.detail))}</span>`;
    else if (e.kind === "bite") txt = `bite: <b>${esc(e.outcome || "")}</b>`;
    else if (e.type === "state") txt = `${e.from} → <b>${e.to}</b> <span class="muted">${esc(e.detail?.why || "")}</span>`;
    else txt = `${esc(e.type || e.kind)} ${esc(compact(e.detail))}`;
    return `<li><time>${t}</time><span>${txt}</span></li>`;
  }).join("");

  if (!$("drawer").classList.contains("hidden")) renderDrawer();
}
const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const compact = (d) => !d ? "" : Object.entries(d).filter(([k]) => k !== "why").map(([k, v]) => `${k.replace("detail_", "")}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ");

// ------------------------------------------------------------------ main buttons
$("btn-stop").onclick = () => post("stop", { source: "ui" });
$("btn-pause").onclick = () => post(S && S.fsm.state === "PAUSED" ? "resume" : "pause");
$("btn-bite").onclick = () => post("bite");
$("btn-session").onclick = () => post(S && S.fsm.session.active ? "session/end" : "session/start");
$("btn-reset").onclick = () => confirmModal(S.fsm.state === "STOPPED"
  ? "Release the stop? The arm will turn torque on and move slowly home." : "Reset the fault? The arm will move slowly home.",
  () => post("reset", { confirm: true }));
$("profile").onchange = (e) => post("profile", { name: e.target.value });

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); post("stop", { source: "key" }); }
  else if (e.key === "o") toggleDrawer();
  else if (e.key === "b") post("bite");
  else if (e.key === "p") $("btn-pause").click();
  else if (e.key === "Escape") { pick = null; updatePick(); }
});

function confirmModal(text, ok) {
  $("modal-text").textContent = text; $("modal").classList.remove("hidden");
  $("modal-ok").onclick = () => { $("modal").classList.add("hidden"); ok(); };
  $("modal-cancel").onclick = () => $("modal").classList.add("hidden");
}

// ------------------------------------------------------------------ operator drawer
function toggleDrawer(open) {
  const d = $("drawer");
  const show = open ?? d.classList.contains("hidden");
  d.classList.toggle("hidden", !show);
  if (show) { loadTraj(); loadCfg(); }
}
$("btn-operator").onclick = () => toggleDrawer();
$("drawer-close").onclick = () => toggleDrawer(false);
$("tabs").onclick = (e) => {
  const t = e.target.dataset.tab; if (!t) return;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  document.querySelectorAll("[data-pane]").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== t));
  if (t === "log") get("log/tail?n=80").then((rows) => $("log").textContent = rows.map((r) =>
    `${new Date(r.ts * 1000).toLocaleTimeString([], { hour12: false })} ${r.kind.padEnd(6)} ${r.text}`).join("\n"));
  if (t === "health") get("health").then((h) => $("health").textContent = JSON.stringify(h, null, 1));
};

function renderDrawer() {
  const a = S.arm;
  $("joints").innerHTML = Object.entries(a.deg).map(([j, d]) =>
    `<div>${j}<b>${d.toFixed(1)}°</b>${a.units[j] ?? ""}u err ${a.error[j] ?? "-"}</div>`).join("") +
    `<div>tip (mm)<b>${a.tip_mm ? a.tip_mm.join(", ") : "-"}</b>${a.in_face_zone ? "IN face zone" : "outside face zone"}</div>` +
    `<div>arm<b>${a.mode}</b>torque ${a.torque ? "on" : "off"} · ${a.voltage ?? "?"} V</div>` +
    `<div>teach<b>${S.teach.recording ? "REC " + S.teach.samples : "-"}</b>${S.teach.name || ""}</div>`;
  const labels = ["nominal", "left", "right", "up", "down", "away", "toward"];
  const have = new Set(S.calib.samples.map((s) => s.label));
  $("calib-labels").innerHTML = labels.map((l) => `<button data-label="${l}" class="${have.has(l) ? "primary" : ""}">${l}</button>`).join("");
}
$("calib-labels").onclick = (e) => { const l = e.target.dataset.label; if (l) { pick = l; updatePick(); } };

const JOINTS = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"];
$("jog").innerHTML = JOINTS.map((j) => `<span>${j}</span>` + [-5, -1, 1, 5].map((d) =>
  `<button data-j="${j}" data-d="${d}">${d > 0 ? "+" : ""}${d}°</button>`).join("")).join("");
$("jog").onclick = (e) => { const b = e.target; if (b.dataset.j) post("jog", { joint: b.dataset.j, deg: +b.dataset.d }); };

$("teach-start").onclick = () => post("teach/start", { name: $("teach-name").value, kind: "path" });
$("teach-stop").onclick = () => post("teach/stop").then((r) => { if (r.ok) { $("plan").textContent = `saved ${r.name}: ${r.waypoints} waypoints`; loadTraj(); } });
$("pose-save").onclick = () => post("teach/pose", { name: $("pose-name").value }).then((r) => { if (r.ok) { $("plan").textContent = `saved ${r.name}: ${JSON.stringify(r.deg)}`; loadTraj(); } });
$("torque-off").onclick = () => post("torque", { on: false });
$("torque-on").onclick = () => post("torque", { on: true });
$("replay-dry").onclick = () => post("teach/replay", { name: $("replay-name").value, speed: +$("replay-speed").value, dry_run: true })
  .then((r) => { if (r.plan) $("plan").textContent = r.plan; });
$("replay-live").onclick = () => confirmModal(`Replay "${$("replay-name").value}" on the arm? Clear the area.`, () =>
  post("teach/replay", { name: $("replay-name").value, speed: +$("replay-speed").value, dry_run: false }));
async function loadTraj() {
  const t = await get("trajectories");
  $("replay-name").innerHTML = t.items.filter((i) => i.ok).map((i) => `<option>${i.name}</option>`).join("");
  const have = new Set(t.items.filter((i) => i.ok).map((i) => i.name));
  $("traj").innerHTML = "<tr><th>name</th><th>kind</th><th>wps</th><th>speed</th><th>recorded</th></tr>" +
    t.items.map((i) => `<tr><td>${i.name}</td><td>${i.kind || ""}</td><td>${i.waypoints ?? ""}</td><td>${i.speed || ""}</td><td>${i.ok ? (i.recorded_at || "") : "INVALID: " + esc(i.error)}</td></tr>`).join("") +
    t.required.filter((n) => !have.has(n)).map((n) => `<tr><td>${n}</td><td colspan="4">not taught</td></tr>`).join("");
}

$("calib-on").onclick = () => post("calib/mode", { on: true });
$("calib-off").onclick = () => post("calib/mode", { on: false });
$("calib-fit").onclick = () => post("calib/fit", { measured_depth_mm: +$("calib-depth").value || null }).then((r) => $("calib-out").textContent = JSON.stringify(r, null, 1));
$("calib-clear").onclick = () => post("calib/clear");
$("window-draw").onclick = () => { pick = "window"; updatePick(); };

function updatePick() {
  $("cam-wrap").classList.toggle("picking", !!pick);
  $("calib-hint").classList.toggle("hidden", !pick);
  $("calib-hint").textContent = pick === "window" ? "Drag the delivery window" : `Click the spoon tip (${pick})`;
}
function camCoords(e, w, h) {
  const r = $("cam").getBoundingClientRect();
  return [(e.clientX - r.left) / r.width * w, (e.clientY - r.top) / r.height * h];
}
$("cam").addEventListener("mousedown", (e) => { if (pick === "window") { winStart = camCoords(e, 1280, 720); e.preventDefault(); } });
$("cam").addEventListener("mouseup", (e) => {
  if (pick === "window" && winStart) {
    const end = camCoords(e, 1280, 720);
    post("calib/window", { x: [winStart[0], end[0]], y: [winStart[1], end[1]] });
    winStart = null; pick = null; updatePick();
  }
});
$("cam").addEventListener("click", (e) => {
  if (!pick || pick === "window" || !S) return;
  const px = camCoords(e, S.camera.w, S.camera.h);
  post("calib/sample", { label: pick, px }); pick = null; updatePick();
});

async function loadCfg() {
  const c = await get("config");
  $("cfg").innerHTML = Object.entries(c).filter(([, v]) => typeof v !== "object").map(([k, v]) =>
    `<label>${k}<input data-path="${k}" value="${v}" ${typeof v === "boolean" ? 'type="checkbox"' + (v ? " checked" : "") : ""}></label>`).join("");
}
$("cfg").onchange = (e) => {
  const i = e.target; const path = i.dataset.path; if (!path) return;
  const value = i.type === "checkbox" ? i.checked : Number(i.value);
  post("config", { path, value }, "PATCH");
};
$("cfg-save").onclick = () => post("config/save");

const simSend = () => post("sim/face", { present: $("sim-present").checked, mouth_open: $("sim-open").checked,
  yaw: +$("sim-yaw").value, dx: +$("sim-dx").value, dz: +$("sim-dz").value });
["sim-present", "sim-open", "sim-yaw", "sim-dx", "sim-dz"].forEach((id) => $(id).addEventListener("input", simSend));
$("sim-block").onclick = () => post("sim/jam", { joint: "elbow", on: true });
$("sim-unblock").onclick = () => post("sim/jam", { joint: "elbow", on: false });

if (location.hash.includes("operator")) { toggleDrawer(true); if (location.hash.includes("calibrate")) $("tabs").querySelector('[data-tab="calibrate"]').click(); }
connect();
