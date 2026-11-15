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
  MOUTH_HOLD: "Holding still", RETREAT: "Moving away", PAUSED: "Paused", FAULT: "Taking a break", STOPPED: "Stopped",
};

async function post(path, body = {}, method = "POST") {
  try {
    const r = await fetch("/api/" + path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) toast(j.detail || r.statusText);
    return j;
  } catch (e) { toast("Can't reach Spoonful. Is it still running?"); return {}; }
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
  if (st === "STOPPED") { v.textContent = "Stopped"; v.className = "verdict stop"; }
  else if (st === "FAULT") { v.textContent = "Taking a break"; v.className = "verdict stop"; }
  else if (["PRE_DELIVERY_HOLD", "DELIVERING", "MOUTH_HOLD"].includes(st)) {
    v.textContent = REASON_TEXT[f.gate.reason] ?? f.gate.reason; v.className = "verdict " + (f.gate.ok ? "ok" : "wait");
  } else { v.textContent = STATE_TEXT[st] || st; v.className = "verdict"; }
  $("explain").textContent = f.explain;
  renderPlate();
  const cd = $("countdown");
  if (f.countdown != null) { cd.textContent = f.countdown.toFixed(1) + " s"; cd.classList.remove("hidden"); } else cd.classList.add("hidden");

  const banner = $("fault-banner");
  if (st === "FAULT") { banner.textContent = "Fault: " + f.fault + ". Arm parked. Check it, then Reset."; banner.classList.remove("hidden"); }
  else if (st === "STOPPED") { banner.textContent = f.stop_release_armed ? "Button pressed. Press Reset to confirm."
      : "Stopped. Press the physical STOP button (or spacebar), then Reset."; banner.classList.remove("hidden"); }
  else if (f.calibration_mode) { banner.textContent = "Calibration mode is on. Nobody in the chair."; banner.classList.remove("hidden"); }
  else if (Date.now() > toastUntil) banner.classList.add("hidden");

  $("btn-reset").classList.toggle("hidden", !(st === "FAULT" || st === "STOPPED"));
  $("btn-pause").textContent = st === "PAUSED" ? "Resume" : "Pause";
  $("btn-bite").disabled = st !== "LISTENING";
  $("btn-session").textContent = f.session.active ? "End session" : "Start session";
  $("bites").textContent = f.session.active ? f.session.taken : (S.fsm.session.bites || 0);
  $("pace").textContent = cap(S.pace.label.replace(/^(learned )?pace: /, "$1"));

  const sel = $("profile");
  if (sel.options.length !== S.profiles.length) {
    sel.innerHTML = S.profiles.map((p) => `<option>${p}</option>`).join("");
  }
  if (document.activeElement !== sel) sel.value = S.profile.name;

  $("timeline").innerHTML = (S.timeline || []).slice().reverse().map((e) => {
    const txt = describe(e);
    if (!txt) return "";
    const t = new Date(e.ts * 1000).toLocaleTimeString([], { hour12: false });
    return `<li><time>${t}</time><span>${txt}</span></li>`;
  }).join("");

  if (!$("drawer").classList.contains("hidden")) renderDrawer();
}
// The activity list says what happened in plain words. Everything else stays in the Log tab.
function describe(e) {
  const d = e.detail || {};
  if (e.kind === "fault") return `<span class="fault">Fault: ${esc(String(d.kind || "").replace(/_/g, " "))}</span>`;
  if (e.kind === "bite") return `Bite ${esc(e.outcome || "")}`;
  if (e.type === "state") return e.to === "FAULT" || e.to === "BOOT" ? "" : esc(STATE_TEXT[e.to] || e.to);
  if (e.type === "speech" && !d.silent) return `Said "${esc(d.detail_line || "")}"`;
  if (e.type === "food_selected") return `Chose ${esc(d.food || "")}`;
  if (e.type === "attack_plan") return `${esc(STRATEGY_TEXT[d.strategy] || "Scoop")}: ${esc(d.why || "")}`;
  if (e.type === "bowl_refused") return `Not scooping: ${esc(d.why || "")}`;
  if (e.type === "photo_map") return "Table mapped";
  if (e.type === "setup_mouth") return "Mouth position saved";
  if (e.type === "face_center") return "Looking direction saved";
  return "";
}
const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

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
  // GO: Enter, or either button of a wireless presenter clicker (they send PageDown / PageUp)
  else if (e.key === "Enter" || e.key === "PageDown" || e.key === "PageUp") { e.preventDefault(); post("go", { source: "key" }); }
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
if (location.hash === "#setup") toggleDrawer(true);
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
  renderSetup();
  const labels = ["nominal", "left", "right", "up", "down", "away", "toward"];
  const have = new Set(S.calib.samples.map((s) => s.label));
  $("calib-labels").innerHTML = labels.map((l) => `<button data-label="${l}" class="${have.has(l) ? "primary" : ""}">${l}</button>`).join("");
}
$("calib-labels").onclick = (e) => { const l = e.target.dataset.label; if (l) { pick = l; updatePick(); } };

const JOINTS = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"];
$("jog").innerHTML = JOINTS.map((j) => `<span>${j}</span>` + [-5, -1, 1, 5].map((d) =>
  `<button data-j="${j}" data-d="${d}">${d > 0 ? "+" : ""}${d}°</button>`).join("")).join("");
$("jog").onclick = (e) => { const b = e.target; if (b.dataset.j) post("jog", { joint: b.dataset.j, deg: +b.dataset.d }); };

// ------------------------------------------------------------------ setup (Setup tab)
const STRATEGY_TEXT = { skim: "Skim", dig: "Dig", stab_and_saw: "Stab and saw", scrape: "Scrape", ladle: "Ladle" };
const AMOUNT_TEXT = { full: "full", half: "half full", low: "nearly empty", empty: "empty" };
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const cap = (t) => t ? t[0].toUpperCase() + t.slice(1) : "";
let fileBusy = false;
let photoKey = "";              // which photo the overlay was built for
const picked = new Set();       // bowl ids ticked in the photo

function chip(id, text, done) {
  const c = $(id);
  c.textContent = text;
  c.closest(".step").classList.toggle("done", !!done);
}
// The photo with every bowl outlined. Selecting an outline (or its row) adds or removes that bowl.
function drawBowls(t) {
  const cands = t.candidates;
  const [w, h] = t.photo_size || [1, 1];
  const svg = $("photo-svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = cands.map((c) => {
    const cls = !c.ok ? "off" : picked.has(c.id) ? "on" : "";
    const pts = c.poly.map((p) => p.join(",")).join(" ");
    const ty = Math.max(30, c.box[1] - 12);
    return `<g class="bowl ${cls}" data-id="${c.id}" ${c.ok ? 'tabindex="0" role="checkbox" aria-checked="' + picked.has(c.id) + '"' : ""}>` +
      `<polygon points="${pts}"></polygon><text x="${c.box[0]}" y="${ty}">${esc(cap(c.food))}</text></g>`;
  }).join("");
  $("s-bowls").innerHTML = cands.map((c) => {
    const detail = !c.ok ? c.why
      : c.strategy && c.used ? `${STRATEGY_TEXT[c.strategy] || ""} · ${AMOUNT_TEXT[c.amount] || ""}`
      : c.sure ? (AMOUNT_TEXT[c.amount] || "") : "Not sure this is food";
    const why = c.used && c.strategy_why ? `<span class="why">${esc(cap(c.strategy_why))}</span>` : "";
    return `<li class="${c.ok ? "" : "skipped"}"><label><input type="checkbox" data-id="${c.id}" ${picked.has(c.id) ? "checked" : ""} ${c.ok ? "" : "disabled"}>` +
      `<b>${esc(cap(c.food))}</b></label><span>${esc(cap(detail))}</span>${why}</li>`;
  }).join("");
  const used = cands.filter((c) => c.used).map((c) => c.id).sort().join();
  const now = [...picked].sort().join();
  $("s-use-row").classList.toggle("hidden", !cands.some((c) => c.ok));
  $("s-use").disabled = !picked.size || S.fsm.state !== "IDLE" || used === now;
  $("s-use").textContent = used && used === now ? "Serving these bowls" : "Serve these bowls";
}
function togglePick(id) {
  const c = S.setup.table.candidates.find((x) => x.id === id);
  if (!c || !c.ok) return;
  if (picked.has(id)) picked.delete(id); else picked.add(id);
  drawBowls(S.setup.table);
}
function renderSetup() {
  const u = S.setup;
  if (!u) return;
  const idle = S.fsm.state === "IDLE";

  chip("s-mouth-chip", u.mouth.set ? (u.mouth.at ? `Saved ${clock(u.mouth.at)}` : "Saved earlier") : "Not set", u.mouth.set);
  $("s-release").disabled = !idle || u.mouth.released;
  $("s-mouth-save").disabled = !idle;
  $("s-mouth-hint").textContent = !idle ? "End the session to change this."
    : u.mouth.released ? "The arm is released. Support it." : "";

  const t = u.table;
  const bowls = t.bowls;
  chip("s-table-chip", bowls.length ? `Serving ${bowls.length} bowl${bowls.length > 1 ? "s" : ""}` : t.candidates.length ? "Choose bowls" : "No photo yet", bowls.length);
  $("photo-pose").textContent = t.posed ? "Spoon is in place" : "Pose for photo";
  $("photo-pose").disabled = !idle || t.posed;
  $("photo-pose").classList.toggle("primary", !t.posed);
  $("phone-qr").classList.toggle("hidden", !t.posed);
  $("phone-qr-wait").classList.toggle("hidden", t.posed);
  $("phone-url").classList.toggle("hidden", !t.posed);
  const ph = u.phone;
  if (ph) {
    const qr = $("phone-qr");
    if (qr.dataset.url !== ph.url) {
      qr.dataset.url = ph.url;
      qr.src = "/api/phone/qr.svg?u=" + encodeURIComponent(ph.url);
      $("phone-url").textContent = ph.url.replace("http://", "");
    }
    if (!fileBusy) {
      const out = $("photo-out");
      out.textContent = { waiting: "Waiting for a photo", finding: "Photo received. Finding the bowls." }[ph.status] || ph.message;
      out.classList.toggle("bad", ph.status === "problem");
    }
  }
  if (t.photo_at) {
    const key = String(t.photo_at) + t.candidates.length;
    if (key !== photoKey) {                 // a new photo (or its bowls just arrived): start from the suggested choice
      if (!photoKey.startsWith(String(t.photo_at))) $("photo-img").src = "/api/phone/photo.jpg?t=" + t.photo_at;
      photoKey = key;
      picked.clear();
      const used = t.candidates.filter((c) => c.used);
      (used.length ? used : t.candidates.filter((c) => c.ok && c.sure)).forEach((c) => picked.add(c.id));
      $("photo-wrap").classList.remove("hidden");
    }
    const drawn = JSON.stringify([t.candidates.map((c) => [c.id, c.used, c.strategy]), [...picked], S.fsm.state === "IDLE"]);
    if (drawn !== $("photo-svg").dataset.drawn) { $("photo-svg").dataset.drawn = drawn; drawBowls(t); }
  }

  chip("s-look-chip", u.looking.at ? `Saved ${clock(u.looking.at)}` : "Not set", u.looking.at);
  const ready = u.mouth.set && bowls.length && u.looking.at;
  chip("s-ready-chip", ready ? "Ready" : "Finish the steps above", ready);
  if (bowls.length) {
    $("s-ready-text").textContent = `Say "Spoonful, give me the ${bowls[0].food}", then press GO. ` +
      "Press GO again when the spoon is waiting in front of you.";
  }
}

$("s-release").onclick = () => post("torque", { on: false });
$("s-mouth-save").onclick = () => post("setup/mouth/save").then((r) => { if (r.ok) loadTraj(); });
$("s-done").onclick = () => toggleDrawer(false);
$("s-use").onclick = () => post("setup/bowls", { ids: [...picked] }).then((r) => { if (r.ok) loadTraj(); });
$("photo-svg").onclick = (e) => { const g = e.target.closest(".bowl"); if (g) togglePick(+g.dataset.id); };
$("photo-svg").onkeydown = (e) => {
  const g = e.target.closest(".bowl");
  if (g && (e.key === " " || e.key === "Enter")) { e.preventDefault(); e.stopPropagation(); togglePick(+g.dataset.id); }
};
$("s-bowls").onchange = (e) => { if (e.target.dataset.id) togglePick(+e.target.dataset.id); };
$("photo-pose").onclick = () => post("photo/pose");
$("photo-go").onclick = () => {
  const f = $("photo-file").files[0];
  if (!f) { toast("Choose a photo first."); return; }
  fileBusy = true;
  $("photo-out").textContent = "Finding the bowls.";
  $("photo-out").classList.remove("bad");
  const rd = new FileReader();
  rd.onload = () => post("photo/upload", { image: rd.result }).then((r) => {
    fileBusy = false;
    if (r.ok) $("photo-out").textContent = "Choose which bowls to serve.";
    else { $("photo-out").textContent = r.detail || "Couldn't read the photo."; $("photo-out").classList.add("bad"); }
  });
  rd.readAsDataURL(f);
};
for (const n of [1, 2]) {
  $(`setup-b${n}`).onclick = () => post("setup/bowl", { bowl: `bowl${n}` }).then((r) => {
    if (r.ok) $("setup-out").textContent = `Bowl ${n} saved. ` + (r.seen ? `The camera sees ${r.food || "a bowl"}.` : "Name its food in the Bowls tab.");
  });
}
$("setup-build").onclick = () => post("setup/build").then((r) => {
  if (r.ok) { $("setup-out").textContent = `Moves built for ${r.bowls.length} bowl${r.bowls.length > 1 ? "s" : ""}.`; loadTraj(); }
});
$("teach-start").onclick = () => post("teach/start", { name: $("teach-name").value, kind: "path" });
$("teach-stop").onclick = () => post("teach/stop").then((r) => { if (r.ok) { $("plan").textContent = `Saved ${r.name} (${r.waypoints} points)`; loadTraj(); } });
$("pose-save").onclick = () => post("teach/pose", { name: $("pose-name").value }).then((r) => { if (r.ok) { $("plan").textContent = `Saved ${r.name}`; loadTraj(); } });
$("spoon-touch").onclick = () => post("spoon/touch").then((r) => {
  if (r.ok) $("spoon-out").textContent = `${r.n} of 3 touches saved`;
});
$("spoon-clear").onclick = () => post("spoon/clear").then(() => { $("spoon-out").textContent = ""; });
$("spoon-fit").onclick = () => post("spoon/fit").then((r) => {
  if (r.ok) $("spoon-out").textContent = `Saved: spoon tip is ${r.tool_mm} mm from the wrist (fit error ${r.rms_mm} mm).`;
});
$("face-center").onclick = async () => {
  const btn = $("face-center");
  btn.disabled = true;
  await post("speak", { text: "Look at the spoon" });
  for (let n = 5; n > 0; n--) {                       // time to turn from the laptop to the spoon
    btn.textContent = `Look at the spoon: ${n}`;
    await new Promise((r) => setTimeout(r, 1000));
  }
  btn.textContent = "Hold still";
  await post("face/center");
  btn.textContent = "Start countdown";
  btn.disabled = false;
};
setInterval(() => {
  const f = S && S.face;
  if (!f) return;
  $("face-live").textContent = !f.present ? "No face in view"
    : `Head turn ${(+f.yaw_deg).toFixed(0)}°, tilt ${(+f.pitch_deg).toFixed(0)}°, mouth open ${f.jaw_open >= 0 ? (+f.jaw_open).toFixed(2) : "?"}`;
}, 300);
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


// ---------------------------------------------------------------- plate
let plateKey = "";
function renderPlate() {
  const p = S && S.plate;
  const line = $("plate-line");
  if (!p || !p.enabled) { line.classList.add("hidden"); return; }
  const foods = p.bowls.filter((b) => b.food).map((b) => cap(b.food) + (b.empty ? " (finished)" : ""));
  const next = p.bowls.find((b) => b.name === (S.fsm.bowl || S.fsm.last_bowl));
  line.textContent = (foods.length ? foods.join(" · ") : "No food seen yet") +
    (next && next.food ? `   ${S.fsm.bowl ? "Now" : "Next"}: ${cap(next.food)}` : "");
  line.classList.remove("hidden");
  $("plate-status").textContent = p.error ? `Vision problem: ${p.error}`
    : "Check aim with camera lines the spoon up over each bowl. Show aim hovers it there; the arrows move it 5 mm.";
  const idle = S.fsm.state === "IDLE";
  const aim = (S.setup && S.setup.aim) || { running: null, results: {} };
  const key = JSON.stringify([p.bowls, aim]) + S.fsm.state;
  if (key === plateKey) return;            // don't rebuild while someone is typing
  const active = document.activeElement;
  if (active && (active.classList.contains("bowl-food") || active.classList.contains("bowl-tex"))) return;
  plateKey = key;
  $("plate-bowls").innerHTML = p.bowls.map((b) => `<div class="bowl-row">
      <b class="${b.empty ? "empty" : ""}">${esc(cap(b.food) || "Unknown food")}</b>
      <span class="meta">${esc([AMOUNT_TEXT[b.amount], STRATEGY_TEXT[b.attack && b.attack.strategy]].filter(Boolean).join(" · "))}</span>
      <input class="bowl-food" data-bowl="${esc(b.name)}" placeholder="Rename" size="12" aria-label="Food name">
      <button data-set="${esc(b.name)}">Save name</button>
      <select class="bowl-tex" data-bowl="${esc(b.name)}" aria-label="Texture">
        <option value="">Texture: ${esc(b.texture)}</option>
        ${["liquid", "soft", "thick", "firm"].map((t) => `<option value="${t}" ${b.texture_why === "operator" && b.texture === t ? "selected" : ""}>${t}</option>`).join("")}
      </select>
      <button data-bowlbite="${esc(b.name)}" ${S.fsm.state === "LISTENING" ? "" : "disabled"}>Bite of this</button>
      <span class="aim" data-aim="${esc(b.name)}">
        <button data-camera="1" class="primary" ${idle && !aim.running ? "" : "disabled"}>Check aim with camera</button>
        <button data-step="0,0" ${idle ? "" : "disabled"}>Show aim</button>
        <button data-step="-5,0" ${idle ? "" : "disabled"} title="Scoop 5 mm closer to the arm">Closer</button>
        <button data-step="5,0" ${idle ? "" : "disabled"} title="Scoop 5 mm further from the arm">Further</button>
        <button data-step="0,5" ${idle ? "" : "disabled"} title="Scoop 5 mm to the arm's left">Left</button>
        <button data-step="0,-5" ${idle ? "" : "disabled"} title="Scoop 5 mm to the arm's right">Right</button>
        <span class="status ${(aim.results[b.name] || {}).state === "problem" ? "bad" : ""}">${esc((aim.results[b.name] || {}).message || "")}</span>
      </span>
    </div>`).join("");
}
$("ask-go").onclick = () => {
  const t = $("ask-text").value.trim();
  $("ask-text").blur();                       // give the keyboard back: the GO button types PageDown
  if (t) post("ask", { text: t }).then(() => { $("ask-text").value = ""; });
};
$("ask-text").onkeydown = (e) => { if (e.key === "Enter") { e.stopPropagation(); $("ask-go").click(); } };
$("plate-bowls").onclick = (e) => {
  const t = e.target;
  if (t.dataset.set) {
    const inp = document.querySelector(`.bowl-food[data-bowl="${t.dataset.set}"]`);
    post("plate/label", { bowl: t.dataset.set, food: inp.value.trim() }).then(() => { inp.value = ""; plateKey = ""; });
  } else if (t.dataset.bowlbite) post("plate/bite", { bowl: t.dataset.bowlbite });
  else if (t.dataset.camera) post("setup/aim/camera", { bowl: t.closest(".aim").dataset.aim });
  else if (t.dataset.step) {
    const [further_mm, left_mm] = t.dataset.step.split(",").map(Number);
    post("setup/aim", { bowl: t.closest(".aim").dataset.aim, further_mm, left_mm });
  }
};
$("plate-bowls").onchange = (e) => {
  const t = e.target;
  if (t.classList.contains("bowl-tex")) post("plate/label", { bowl: t.dataset.bowl, food: "", texture: t.value }).then(() => { plateKey = ""; });
};
$("plate-check").onclick = () => post("plate/check").then(() => { plateKey = ""; $("plate-img").src = "/api/plate/image?t=" + Date.now(); });
$("plate-img").onclick = (e) => {
  if (!$("table-mode").checked) return;
  const img = e.target, r = img.getBoundingClientRect();
  const u = (e.clientX - r.left) / r.width * 1280, v = (e.clientY - r.top) / r.height * 720;
  post("plate/table/point", { u, v }).then((j) => {
    if (j.ok) { toast(`Table spot ${j.points} saved (${j.mm[0]}, ${j.mm[1]} mm)` + (j.ready ? ` — fit error ${j.rms_mm} mm` : ` — ${j.needed - j.points} more`)); }
    $("plate-img").src = "/api/plate/image?t=" + Date.now();
  });
};
$("table-auto").onclick = () => post("plate/table/auto");
$("table-auto-stop").onclick = () => post("plate/table/auto/stop");
$("table-clear").onclick = () => post("plate/table/clear").then(() => { $("plate-img").src = "/api/plate/image?t=" + Date.now(); });
setInterval(() => {
  const p = S && S.plate; if (!p || !p.table || !$("table-status")) return;
  const t = p.table;
  const sh = Object.entries(p.shifts || {}).map(([k, s]) => `${k}: ${s ? `moved ${(s[0] / 10).toFixed(1)}, ${(s[1] / 10).toFixed(1)} cm` : "at its spot / not found"}`).join(" · ");
  const a = p.autocal || {};
  $("table-status").textContent = (a.running ? `Calibrating: ${a.msg} (${a.points} seen). ` : (a.msg ? `Last calibration: ${a.msg}. ` : "")) +
    (t.ready ? `Table calibrated from ${t.points} spots (fit error ${t.rms_mm} mm). ${sh}`
             : `Table not calibrated: ${t.points} of ${t.needed} spots.`);
}, 500);
$("plate-img-refresh").onclick = () => { $("plate-img").src = "/api/plate/image?t=" + Date.now(); };
