"""REST endpoints. Every button POSTs here; every endpoint is idempotent (posting STOP twice = one STOP).

Motion endpoints (jog, replay, sim hand) only act in IDLE, and still go through every arm guard.
"""
from __future__ import annotations

import re

import csv
import io
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from ..arm import kinematics as K
from ..arm.teach import save_current_pose
from ..arm.trajectory import plan as make_plan
from ..config import ARM_JOINTS, LIVE_EDITABLE
from ..brain.fsm import TRANSITIONS, S

router = APIRouter(prefix="/api")


def _sys(req: Request):
    return req.app.state.sys


def _need_idle(sys) -> None:
    if sys.fsm.state != S.IDLE:
        raise HTTPException(409, f"only in IDLE (now {sys.fsm.state.value})")


class Src(BaseModel):
    source: str = "ui"


# ------------------------------------------------------------------ diner controls
@router.post("/stop")
def stop(req: Request, b: Src = Src()):
    _sys(req).fsm.post("stop", source=b.source)
    return {"ok": True}


@router.post("/pause")
def pause(req: Request):
    _sys(req).fsm.post("pause", source="ui")
    return {"ok": True}


@router.post("/resume")
def resume(req: Request):
    _sys(req).fsm.post("resume", source="ui")
    return {"ok": True}


@router.post("/go")
def go(req: Request, b: Src = Src()):
    _sys(req).fsm.post("go", source=b.source)
    return {"ok": True}


@router.post("/bite")
def bite(req: Request):
    _sys(req).fsm.post("bite", source="ui")
    return {"ok": True}


class Confirm(BaseModel):
    confirm: bool = False


@router.post("/reset")
def reset(req: Request, b: Confirm):
    if not b.confirm:
        raise HTTPException(400, "confirm=true required")
    _sys(req).fsm.post("reset", source="ui")
    return {"ok": True}


@router.post("/session/start")
def session_start(req: Request):
    _sys(req).fsm.post("session_start", source="ui")
    return {"ok": True}


@router.post("/session/end")
def session_end(req: Request):
    _sys(req).fsm.post("session_end", source="ui")
    return {"ok": True}


class Name(BaseModel):
    name: str


@router.post("/profile")
def profile(req: Request, b: Name):
    _sys(req).fsm.post("profile", name=b.name)
    return {"ok": True}


@router.post("/profile/update")
def profile_update(req: Request, b: dict):
    sys = _sys(req)
    allowed = {"approach_speed", "hold_s", "bite_size", "preferred_food", "speech", "auto_pace", "pace_scale"}
    p = sys.profiles.update(**{k: v for k, v in b.items() if k in allowed})
    return p.__dict__


@router.post("/voice/ptt")
def ptt(req: Request, b: dict):
    _sys(req).voice.set_ptt(bool(b.get("on")))
    return {"ok": True}


# ------------------------------------------------------------------ teach
class TeachStart(BaseModel):
    name: str
    kind: str = "path"


@router.post("/teach/start")
def teach_start(req: Request, b: TeachStart):
    sys = _sys(req)
    _need_idle(sys)
    sys.recorder.profile = sys.profiles.current.name
    sys.recorder.start(b.name, b.kind)
    sys.log.event("teach_start", detail_name=b.name, kind=b.kind)
    return {"ok": True}


@router.post("/teach/stop")
def teach_stop(req: Request):
    sys = _sys(req)
    if not sys.recorder.recording:
        return {"ok": False, "error": "not recording"}
    try:
        tf = sys.recorder.stop()
    except Exception as e:
        raise HTTPException(422, str(e))
    sys.lib.reload()
    sys.log.event("teach_saved", detail_name=tf.name, waypoints=len(tf.waypoints))
    return {"ok": True, "name": tf.name, "waypoints": len(tf.waypoints)}


@router.post("/teach/pose")
def teach_pose(req: Request, b: Name):
    sys = _sys(req)
    _need_idle(sys)
    try:
        tf = save_current_pose(sys.cfg, sys.arm, sys.lib.dir, b.name)
    except Exception as e:
        raise HTTPException(422, str(e))
    sys.lib.reload()
    return {"ok": True, "name": tf.name, "deg": tf.waypoints[0]["deg"]}


class Replay(BaseModel):
    name: str
    speed: float = 0.8
    dry_run: bool = True


@router.post("/teach/replay")
def teach_replay(req: Request, b: Replay):
    sys = _sys(req)
    _need_idle(sys)
    if not sys.lib.has(b.name):
        raise HTTPException(404, f"no trajectory {b.name}")
    tf = sys.lib.items[b.name]
    p = make_plan(sys.cfg, sys.arm.current(), [(tf.poses, tf.speed, b.name)], max(0.1, min(1.0, b.speed)), b.name)
    if b.dry_run:
        return {"ok": True, "dry_run": True, "plan": p.describe(hz=4)}
    sys.arm.run(p)
    return {"ok": True, "dry_run": False, "duration_s": round(p.duration, 2)}


@router.get("/trajectories")
def trajectories(req: Request):
    sys = _sys(req)
    sys.lib.reload()
    return {"items": sys.lib.summary(), "required": list(sys.lib.PATHS) + list(sys.lib.POSES)}


class Torque(BaseModel):
    on: bool


@router.post("/torque")
def torque(req: Request, b: Torque):
    sys = _sys(req)
    _need_idle(sys)
    sys.arm.torque(b.on)
    return {"ok": True}


class Jog(BaseModel):
    joint: str
    deg: float


@router.post("/jog")
def jog(req: Request, b: Jog):
    sys = _sys(req)
    _need_idle(sys)
    if b.joint not in ARM_JOINTS or abs(b.deg) > 10:
        raise HTTPException(400, "joint must be an arm joint and |deg| <= 10")
    cur = sys.arm.current()
    tgt = dict(cur)
    tgt[b.joint] += b.deg
    speed = "near_face" if K.in_face_zone(sys.cfg.kinematics, tgt) else "normal"
    try:
        sys.arm.goto(tgt, speed, 0.5, name="jog")
    except Exception as e:
        raise HTTPException(422, str(e))
    return {"ok": True, "target": {j: round(v, 1) for j, v in tgt.items()}}


@router.post("/home")
def home(req: Request):
    sys = _sys(req)
    _need_idle(sys)
    sys.arm.goto(sys.cfg.arm.home_pose, "normal", 0.8, name="home")
    return {"ok": True}


# ------------------------------------------------------------------ calibration
class Mode(BaseModel):
    on: bool


@router.post("/calib/mode")
def calib_mode(req: Request, b: Mode):
    _sys(req).fsm.post("calibration_mode", on=b.on)
    return {"ok": True}


class Sample(BaseModel):
    label: str
    px: tuple[float, float]


@router.post("/calib/sample")
def calib_sample(req: Request, b: Sample):
    return _sys(req).calib_add(b.label, b.px)


@router.post("/calib/clear")
def calib_clear(req: Request):
    _sys(req).calib_samples.clear()
    return {"ok": True}


class FitReq(BaseModel):
    measured_depth_mm: Optional[float] = None   # tape: camera lens to spoon tip at nominal


@router.post("/calib/fit")
def calib_fit(req: Request, b: Optional[FitReq] = None):
    try:
        m = _sys(req).calib_fit(b.measured_depth_mm if b else None)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"ok": True, "px_per_mm": round(m.px_per_mm, 3), "residual_deg": round(m.residual_deg, 2),
            "nominal_px": m.nominal_px, "depth": m.has_depth, "focal_px": round(m.focal_px, 1),
            "nominal_depth_mm": round(m.nominal_depth_mm), "warnings": m.warnings}


class WindowBox(BaseModel):
    x: tuple[float, float]
    y: tuple[float, float]


@router.post("/calib/window")
def calib_window(req: Request, b: WindowBox):
    sys = _sys(req)
    sys.cfg.set_path("face.delivery_window_px", {"x": sorted(b.x), "y": sorted(b.y)})
    sys.log.event("config", detail_path="face.delivery_window_px", value=b.model_dump())
    return {"ok": True}


# ------------------------------------------------------------------ simulation helpers (sim backend / --sim-face only)
@router.post("/sim/face")
def sim_face(req: Request, b: dict):
    sys = _sys(req)
    if not sys.face.sim_enabled:
        raise HTTPException(409, "start with --sim-face")
    for k in ("present", "mouth_open", "yaw", "pitch", "dx", "dy", "dz"):
        if k in b:
            sys.face.sim[k] = b[k]
    return sys.face.sim


@router.post("/sim/hand")
def sim_hand(req: Request, b: Jog):
    sys = _sys(req)
    if not sys.is_sim:
        raise HTTPException(409, "sim backend only")
    sid = sys.cfg.arm.servo_map[b.joint]
    sys.backend.hand_move(sid, sys.arm.to_units(b.joint, b.deg))
    return {"ok": True}


@router.post("/sim/jam")
def sim_jam(req: Request, b: dict):
    """Simulate contact: {joint, on} stalls that servo."""
    sys = _sys(req)
    if not sys.is_sim:
        raise HTTPException(409, "sim backend only")
    sys.backend.jam(sys.cfg.arm.servo_map[b["joint"]], bool(b.get("on")))
    return {"ok": True}


# ------------------------------------------------------------------ config
@router.get("/config")
def config_get(req: Request):
    sys = _sys(req)
    return {p: sys.cfg.get_path(p) for p in LIVE_EDITABLE}


class SetCfg(BaseModel):
    path: str
    value: Any


@router.patch("/config")
def config_set(req: Request, b: SetCfg):
    sys = _sys(req)
    if b.path not in LIVE_EDITABLE:
        raise HTTPException(403, f"{b.path} is not live-editable")
    try:
        sys.cfg.set_path(b.path, b.value)
    except Exception as e:
        raise HTTPException(422, str(e))
    sys.log.event("config", detail_path=b.path, value=b.value)
    return {"ok": True, "value": sys.cfg.get_path(b.path)}


class Speak(BaseModel):
    text: str


@router.post("/speak")
def speak(req: Request, b: Speak):
    _sys(req).speaker.say_text(b.text[:120])
    return {"ok": True}


class Ask(BaseModel):
    text: str


@router.post("/ask")
def ask(req: Request, b: Ask):
    """Same as saying "Spoonful, <text>" (voice-select mode): plate question, food name, or an open request."""
    sys = _sys(req)
    sys.on_voice(None, "Spoonful, " + b.text, 1.0)
    return {"ok": True}


@router.get("/plan/last")
def plan_last(req: Request):
    return _sys(req).planner.last


@router.post("/photo/pose")
def photo_pose(req: Request):
    """Hold the spoon tip at the known spot for the overhead photo."""
    from ..brain.table_auto import pose_at
    from ..perception.photo_map import PHOTO_TIP_MM
    from ..arm.trajectory import check_limits
    sys = _sys(req)
    _need_idle(sys)
    ref = sys.lib.pose("rest") if sys.lib.has("rest") else dict(sys.cfg.arm.home_pose)

    def ok(q):
        try:
            check_limits(sys.cfg, q)
        except Exception:
            return False
        return not K.in_face_zone(sys.cfg.kinematics, q)
    q = pose_at(sys.cfg.kinematics, ref, *PHOTO_TIP_MM, ok)
    if q is None:
        raise HTTPException(409, "can't reach the photo spot")
    sys.arm.goto(q, "normal", 0.4, name="photo_pose")
    return {"ok": True, "tip_mm": PHOTO_TIP_MM}


class PhotoUpload(BaseModel):
    image: str          # data URL from the browser


@router.post("/photo/upload")
def photo_upload(req: Request, b: PhotoUpload):
    """An overhead photo chosen as a file on the laptop (the phone link posts to /api/phone/photo instead)."""
    import cv2
    import numpy as np
    from ..brain.photo_setup import SetupProblem, find_bowls
    from ..perception.photo_map import decode_upload
    sys = _sys(req)
    img = cv2.imdecode(np.frombuffer(decode_upload(b.image), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "That isn't an image.")
    try:
        return {"ok": True, "candidates": find_bowls(sys, img)}
    except SetupProblem as e:
        raise HTTPException(409, str(e))


@router.post("/setup/mouth/save")
def setup_mouth_save(req: Request):
    """The spoon is at the diner's lips: hold the arm there and save the mouth, waiting pose and face zone."""
    from ..brain.photo_setup import SetupProblem, save_mouth
    sys = _sys(req)
    _need_idle(sys)
    try:
        r = save_mouth(sys)
    except SetupProblem as e:
        raise HTTPException(409, str(e))
    sys.arm.torque(True)
    return {"ok": True, **r}


@router.get("/surface")
def surface(req: Request):
    return _sys(req).food_surface.status()


class SetupBowl(BaseModel):
    bowl: str = "bowl1"


def _setup(sys):
    if not hasattr(sys, "setup_touch"):
        sys.setup_touch = {"dips": {}, "mouth": None, "boxes": {}}
    return sys.setup_touch


@router.post("/setup/bowl")
def setup_bowl(req: Request, b: SetupBowl):
    """Spoon is dipped in this bowl right now (moved there by hand): remember the pose, and ask the camera which
    container the spoon is in (that becomes the bowl's picture box)."""
    sys = _sys(req)
    if not re.fullmatch(r"bowl[1-6]", b.bowl):
        raise HTTPException(400, "bowl must be bowl1..bowl6")
    st = _setup(sys)
    st["dips"][b.bowl] = {j: sys.arm.state.deg[j] for j in ARM_JOINTS}
    box = sys.plate.box_of_touched_bowl()
    if box:
        st["boxes"][b.bowl] = box
    x, y, z = K.fk(sys.cfg.kinematics, st["dips"][b.bowl])
    sys.log.event("setup_bowl", detail_bowl=b.bowl, tip_mm=[round(x), round(y), round(z)], box=box)
    return {"ok": True, "bowl": b.bowl, "seen": bool(box), "food": box and box["food"], "tip_mm": [round(x), round(y), round(z)]}


@router.post("/setup/mouth")
def setup_mouth(req: Request):
    sys = _sys(req)
    st = _setup(sys)
    st["mouth"] = {j: sys.arm.state.deg[j] for j in ARM_JOINTS}
    x, y, z = K.fk(sys.cfg.kinematics, st["mouth"])
    sys.log.event("setup_mouth", detail_tip_mm=[round(x), round(y), round(z)])
    return {"ok": True, "tip_mm": [round(x), round(y), round(z)]}


@router.post("/setup/build")
def setup_build(req: Request):
    """Generate and save every move from the touches; set the face zone around the mouth and the bowls' boxes."""
    from ..brain.quick_setup import SetupError, build_all, save_all
    from ..config import BowlCfg
    sys = _sys(req)
    _need_idle(sys)
    st = _setup(sys)
    if st["mouth"] is None and sys.lib.has("mouth_nominal"):
        st["mouth"] = dict(sys.lib.pose("mouth_nominal"))        # saved in step 1 of Setup
    if st["mouth"] is None:
        raise HTTPException(409, "Set your mouth position first.")
    rest = sys.lib.pose("rest") if sys.lib.has("rest") else dict(sys.cfg.arm.home_pose)
    try:
        built = build_all(sys.cfg, st["dips"], st["mouth"], rest)
    except Exception as e:
        raise HTTPException(409, f"couldn't build the moves: {e}")
    saved = save_all(sys.cfg, sys.lib.dir, built)
    sys.cfg.kinematics.face_cyl = built["face_cyl"]
    bowls = []
    for name in sorted(st["dips"]):
        old = next((b for b in sys.cfg.plate.bowls if b.name == name), None)
        box = st["boxes"].get(name)
        roi = tuple(box["roi"]) if box else (old.roi if old else (0, 0, 1280, 720))
        bowls.append(BowlCfg(name=name, roi=roi, label="" if box else (old.label if old else ""), watch=bool(box)))
    sys.cfg.plate.bowls = bowls
    sys.cfg.save()
    sys.lib.reload()
    sys.plate.reset_bowls()
    sys.log.event("setup_built", detail_moves=saved, bowls=[b.name for b in bowls])
    return {"ok": True, "moves": saved, "bowls": [b.name for b in bowls],
            "foods": {k: v.get("food") for k, v in st["boxes"].items()}}


@router.post("/setup/clear")
def setup_clear(req: Request):
    _sys(req).setup_touch = {"dips": {}, "mouth": None, "boxes": {}}
    return {"ok": True}


@router.post("/spoon/touch")
def spoon_touch(req: Request):
    """The spoon tip is resting on the table right now (moved there by hand, torque off): record the pose."""
    sys = _sys(req)
    if not hasattr(sys, "spoon_touches"):
        sys.spoon_touches = []
    q = {j: sys.arm.state.deg[j] for j in ARM_JOINTS}
    sys.spoon_touches.append(q)
    s, z0 = sys.cfg.kinematics.sign, sys.cfg.kinematics.zero_deg
    tilt = sum(s[j] * (q[j] - z0[j]) for j in ("shoulder", "elbow", "wrist_pitch"))
    sys.log.event("spoon_touch", detail_n=len(sys.spoon_touches), tilt_deg=round(tilt, 1))
    return {"ok": True, "n": len(sys.spoon_touches), "tilt_deg": round(tilt, 1)}


@router.post("/spoon/clear")
def spoon_clear(req: Request):
    _sys(req).spoon_touches = []
    return {"ok": True}


@router.post("/spoon/fit")
def spoon_fit(req: Request):
    sys = _sys(req)
    t = getattr(sys, "spoon_touches", [])
    if len(t) < 3:
        raise HTTPException(409, f"need 3 touches at different tilts (have {len(t)})")
    kin = sys.cfg.kinematics
    L, o, rms, spread = K.fit_spoon(kin, t)
    if spread < 20:
        raise HTTPException(409, f"the touches are too similar (tilts differ by {spread:.0f} deg; need 20+): touch once "
                                 "with the spoon steep, once medium, once nearly flat")
    if not (120 <= L <= 400 and -80 <= o <= 80) or rms > 8:
        raise HTTPException(409, f"that doesn't fit a spoon (length {L:.0f} mm, offset {o:.0f} mm, error {rms:.1f} mm): "
                                 "make sure the TIP touches the table each time, then clear and redo")
    old = (kin.tool_mm, kin.tool_offset_mm)
    sys.cfg.set_path("kinematics.tool_mm", round(L, 1))
    sys.cfg.set_path("kinematics.tool_offset_mm", round(o, 1))
    sys.cfg.set_path("kinematics.spoon_measured", True)
    sys.cfg.save()
    sys.log.event("spoon_fit", detail_tool_mm=round(L, 1), offset_mm=round(o, 1), rms_mm=round(rms, 1),
                  was=[round(v, 1) for v in old])
    return {"ok": True, "tool_mm": round(L, 1), "tool_offset_mm": round(o, 1), "rms_mm": round(rms, 1),
            "was": [round(v, 1) for v in old]}


@router.post("/face/center")
def face_center(req: Request):
    """Diner looks at the spoon (arm) for ~1 s: that head direction becomes "looking". Saved to config.yaml."""
    import time as _t
    sys = _sys(req)
    sys.speaker.say_text("Hold still")
    ys, ps = [], []
    t_end = _t.time() + 1.5
    while _t.time() < t_end:
        f = sys.face.latest() if hasattr(sys.face, "latest") else sys.face.state
        if f and f.present:
            ys.append(f.yaw_deg)
            ps.append(f.pitch_deg)
        _t.sleep(0.05)
    if len(ys) < 5:
        sys.speaker.say_text("I couldn't see your face")
        raise HTTPException(409, "no face seen: sit in place, look at the spoon, and try again")
    ys.sort(); ps.sort()
    yaw, pitch = ys[len(ys) // 2], ps[len(ps) // 2]
    sys.cfg.set_path("face.yaw_center_deg", round(yaw, 1))
    sys.cfg.set_path("face.pitch_offset_deg", round(pitch, 1))
    sys.cfg.save()
    sys.setup_state["looking_at"] = _t.time()
    sys.log.event("face_center", detail_yaw=round(yaw, 1), pitch=round(pitch, 1))
    sys.speaker.say_text("Got it")
    return {"ok": True, "yaw_center_deg": round(yaw, 1), "pitch_offset_deg": round(pitch, 1)}


@router.post("/config/save")
def config_save(req: Request):
    _sys(req).cfg.save()
    return {"ok": True}


# ------------------------------------------------------------------ logs, caregiver, stats
@router.get("/log/tail")
def log_tail(req: Request, n: int = 60):
    return _sys(req).log.tail(n)


@router.get("/safety")
def safety():
    return [{"from": f if isinstance(f, str) else f.value, "to": t.value, "trigger": w} for f, t, w in TRANSITIONS]


@router.get("/sessions")
def sessions(req: Request):
    sys = _sys(req)
    out = sys.log.sessions(50)
    for s in out:
        s["bite_rows"] = sys.log.query("SELECT outcome, COUNT(*) n FROM bites WHERE session_id=? GROUP BY outcome",
                                       (s["id"],))
    return {"sessions": out, "outbox": sys.summary_outbox[-10:]}


@router.get("/bites.csv", response_class=PlainTextResponse)
def bites_csv(req: Request):
    rows = _sys(req).log.query("SELECT * FROM bites ORDER BY ts_request")
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return PlainTextResponse(buf.getvalue(), headers={"Content-Disposition": "attachment; filename=bites.csv"})


class SendSummary(BaseModel):
    session_id: Optional[int] = None


@router.post("/summary/send")
def summary_send(req: Request, b: SendSummary):
    sys = _sys(req)
    item = None
    if b.session_id is not None:
        rows = sys.log.query("SELECT id, summary FROM sessions WHERE id=?", (b.session_id,))
        if rows and rows[0]["summary"]:
            item = {"session_id": rows[0]["id"], "text": rows[0]["summary"], "sent": False, "detail": "",
                    "ts": time.time()}
    elif sys.summary_outbox:
        item = sys.summary_outbox[-1]
    if item is None:
        raise HTTPException(404, "no finished session summary yet")
    return sys.send_summary(item)


@router.get("/stats")
def stats(req: Request):
    from ..stats import compute
    return compute(_sys(req).log)


@router.get("/health")
def health(req: Request):
    sys = _sys(req)
    return {"fsm": sys.fsm.state.value, "arm": sys.arm.status(), "camera_fps": sys.camera.fps,
            "face_fps": sys.face.fps, "trajectories": sys.lib.summary(), "offset_map": bool(sys.offset_map),
            "esp32": sys.esp32.status(), "voice": sys.voice.status(), "watchdog_trips": sys.watchdog.trips[-5:]}


# ---------------------------------------------------------------- plate (which food is in which bowl)
@router.get("/plate")
def plate_status(req: Request):
    return _sys(req).plate.status()


@router.get("/plate/image")
def plate_image(req: Request):
    """Plate camera frame with each bowl's box and label drawn: use it to set plate.bowls[].roi."""
    import cv2
    sys = _sys(req)
    img = sys.plate.frame_fn()
    if img is None:
        raise HTTPException(503, "no plate camera frame")
    img = img.copy()
    h, w = img.shape[:2]
    sx, sy = w / 1280.0, h / 720.0
    st = {b["name"]: b for b in sys.plate.status()["bowls"]}
    for b in sys.cfg.plate.bowls:
        x, y, bw, bh = b.roi
        p0, p1 = (int(x * sx), int(y * sy)), (int((x + bw) * sx), int((y + bh) * sy))
        cv2.rectangle(img, p0, p1, (0, 200, 255), 3)
        label = f"{b.name}: {st[b.name]['food'] or '?'}"
        cv2.putText(img, label, (p0[0] + 6, max(24, p0[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    for p in sys.plate.table.points:                                   # table calibration spots: green
        c = (int(p["px"][0] * sx), int(p["px"][1] * sy))
        cv2.circle(img, c, 7, (60, 200, 60), -1)
    shifts = sys.plate.status().get("shifts", {})
    for name, b in sys.plate.bowls.items():                            # where each bowl was found: magenta
        if b.get("pos_px"):
            c = (int(b["pos_px"][0] * sx), int(b["pos_px"][1] * sy))
            cv2.circle(img, c, 14, (200, 60, 200), 3)
            s = shifts.get(name)
            txt = f"{name} " + (f"moved {s[0] / 10:+.0f},{s[1] / 10:+.0f} cm" if s else "found")
            cv2.putText(img, txt, (c[0] + 18, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 60, 200), 2)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpg.tobytes(), media_type="image/jpeg")


class TablePoint(BaseModel):
    u: float            # pixel in 1280x720 reference coordinates
    v: float


@router.post("/plate/table/point")
def plate_table_point(req: Request, b: TablePoint):
    """Spoon tip is touching the table at the spot just clicked in the plate picture: pair the two."""
    sys = _sys(req)
    x, y, z = K.fk(sys.cfg.kinematics, sys.arm.state.deg)
    if z > 60:
        raise HTTPException(409, f"the spoon tip is {z:.0f} mm up: touch it to the table first, then click the spot")
    sys.plate.table.add((b.u, b.v), (x, y))
    sys.log.event("table_point", detail_px=[round(b.u), round(b.v)], mm=[round(x), round(y)])
    return {"ok": True, "mm": [round(x), round(y)], **sys.plate.table.status()}


@router.post("/plate/table/auto")
def plate_table_auto(req: Request):
    """The arm calibrates the plate camera by itself (cup removed, IDLE)."""
    try:
        _sys(req).table_auto.start()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@router.post("/plate/table/auto/stop")
def plate_table_auto_stop(req: Request):
    _sys(req).table_auto.abort()
    return {"ok": True}


@router.post("/plate/table/clear")
def plate_table_clear(req: Request):
    _sys(req).plate.table.clear()
    return {"ok": True}


class PlateLabel(BaseModel):
    bowl: str
    food: str = ""
    texture: Optional[str] = None      # liquid|soft|thick|firm, "" = automatic, None = leave as is


@router.post("/plate/label")
def plate_label(req: Request, b: PlateLabel):
    try:
        _sys(req).plate.set_label(b.bowl, b.food, b.texture)
    except KeyError:
        raise HTTPException(404, f"no bowl {b.bowl}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    _sys(req).log.event("plate_label", detail_bowl=b.bowl, food=b.food, texture=b.texture)
    return {"ok": True}


@router.post("/plate/check")
def plate_check(req: Request):
    p = _sys(req).plate
    return {"ok": p.check_now(), "error": p.error}


class PlateBite(BaseModel):
    bowl: str


@router.post("/plate/bite")
def plate_bite(req: Request, b: PlateBite):
    _sys(req).fsm.post("bite", source="ui", bowl=b.bowl)
    return {"ok": True}
