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
  renderPlate();
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

const setupLog = (t) => { $("setup-out").textContent = t; };
for (const n of [1, 2]) {
  $(`setup-b${n}`).onclick = () => post("setup/bowl", { bowl: `bowl${n}` }).then((r) => {
    if (r.ok) setupLog(`Bowl ${n} saved` + (r.seen ? ` — camera sees ${r.food || "a bowl"}` : " — camera couldn't see the bowl (type its food in the Plate tab later)"));
  });
}
$("setup-mouth").onclick = () => post("setup/mouth").then((r) => { if (r.ok) setupLog("Mouth saved. Torque on, then Build moves."); });
$("setup-build").onclick = () => post("setup/build").then((r) => {
  if (r.ok) { setupLog(`Built: ${r.moves.join(", ")}. Bowls: ${r.bowls.join(", ")}. Try a bite!`); loadTraj(); }
});
$("photo-pose").onclick = () => post("photo/pose").then((r) => { if (r.ok) $("photo-out").textContent = "Spoon in place. Take the photo from straight above."; });
$("photo-go").onclick = () => {
  const f = $("photo-file").files[0];
  if (!f) { toast("choose the photo first"); return; }
  $("photo-out").textContent = "Reading the photo...";
  const rd = new FileReader();
  rd.onload = () => post("photo/upload", { image: rd.result }).then((r) => {
    if (r.ok) {
      $("photo-out").textContent = r.report.join(" | ");
      $("photo-img").src = "/api/photo/annotated?t=" + Date.now(); $("photo-img").style.display = "block";
      loadTraj();
    } else $("photo-out").textContent = "";
  });
  rd.readAsDataURL(f);
};
$("setup-clear").onclick = () => post("setup/clear").then(() => setupLog("Cleared."));
$("teach-start").onclick = () => post("teach/start", { name: $("teach-name").value, kind: "path" });
$("teach-stop").onclick = () => post("teach/stop").then((r) => { if (r.ok) { $("plan").textContent = `saved ${r.name}: ${r.waypoints} waypoints`; loadTraj(); } });
$("pose-save").onclick = () => post("teach/pose", { name: $("pose-name").value }).then((r) => { if (r.ok) { $("plan").textContent = `saved ${r.name}: ${JSON.stringify(r.deg)}`; loadTraj(); } });
$("spoon-touch").onclick = () => post("spoon/touch").then((r) => {
  if (r.ok) $("spoon-out").textContent = `${r.n} touch${r.n > 1 ? "es" : ""} (this one tilted ${r.tilt_deg} deg)`;
});
$("spoon-clear").onclick = () => post("spoon/clear").then(() => { $("spoon-out").textContent = "cleared"; });
$("spoon-fit").onclick = () => post("spoon/fit").then((r) => {
  if (r.ok) $("spoon-out").textContent = `Spoon: ${r.tool_mm} mm long, tip ${r.tool_offset_mm} mm off the gripper line ` +
    `(was ${r.was[0]} / ${r.was[1]}), fit error ${r.rms_mm} mm. Saved.`;
});
$("face-center").onclick = async () => {
  const btn = $("face-center"); btn.disabled = true;
  await post("speak", { text: "Look at the spoon" });
  for (let n = 5; n > 0; n--) {                       // time to turn from the laptop to the spoon
    btn.textContent = `Look at the spoon... ${n}`;
    await new Promise((r) => setTimeout(r, 1000));
  }
  btn.textContent = "Hold still...";
  const r = await post("face/center");
  btn.textContent = "I'm looking at the arm"; btn.disabled = false;
  if (r.ok) toast(`Saved: looking at the arm = head turned ${r.yaw_center_deg} deg, tilted ${r.pitch_offset_deg} deg`);
};
setInterval(() => {
  const f = S && S.face; if (!f || !$("face-live")) return;
  $("face-live").textContent = f.present ? `now: turn ${(+f.yaw_deg).toFixed(0)} deg, tilt ${(+f.pitch_deg).toFixed(0)} deg, ` +
    `mouth ${f.jaw_open >= 0 ? "jaw " + (+f.jaw_open).toFixed(2) : "ratio " + (+f.mouth_open_ratio).toFixed(2)}` : "no face";
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
  const foods = p.bowls.map((b) => b.food ? (b.empty ? `${b.food} (finished)` : b.food) : null).filter(Boolean);
  const now = S.fsm.bowl || S.fsm.last_bowl;
  const cur = p.bowls.find((b) => b.name === now);
  line.textContent = (foods.length ? "On the plate: " + foods.join(" · ") : "On the plate: (not seen yet)") +
    (cur && cur.food ? `   → ${S.fsm.bowl ? "now" : "next"}: ${cur.food} (${cur.texture} scoop)` : "");
  line.classList.remove("hidden");
  $("plate-status").textContent = `vision: ${p.vision}` + (p.error ? ` — ${p.error}` : "") +
    (p.last_check_age_s != null ? ` — checked ${p.last_check_age_s}s ago` : "");
  const key = JSON.stringify(p.bowls) + S.fsm.state;
  if (key === plateKey) return;            // don't rebuild while someone is typing
  if (document.activeElement && (document.activeElement.classList.contains("bowl-food") ||
      document.activeElement.classList.contains("bowl-tex"))) return;
  plateKey = key;
  $("plate-bowls").innerHTML = p.bowls.map((b) => `<div class="bowl-row">
      <b>${esc(b.name)}</b>
      <span class="${b.empty ? "empty" : ""}">${esc(b.food || "?")}${b.amount ? " (" + esc(b.amount) + ")" : ""}</span>
      <small>[${esc(b.source || "unknown")}; scoop: ${esc(b.scoop)}]</small>
      <select class="bowl-tex" data-bowl="${esc(b.name)}" title="how hard to scoop (${esc(b.texture_why)})">
        <option value="">auto: ${esc(b.texture)}</option>
        ${["liquid", "soft", "thick", "firm"].map((t) => `<option value="${t}" ${b.texture_why === "operator" && b.texture === t ? "selected" : ""}>${t}</option>`).join("")}
      </select>
      <input class="bowl-food" data-bowl="${esc(b.name)}" placeholder="type food to override" size="14">
      <button data-set="${esc(b.name)}">Set</button>
      <button data-bowlbite="${esc(b.name)}" ${S.fsm.state === "LISTENING" ? "" : "disabled"}>Bite from ${esc(b.name)}</button>
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
