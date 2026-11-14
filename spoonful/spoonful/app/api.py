"""REST endpoints. Every button POSTs here; every endpoint is idempotent (posting STOP twice = one STOP).

Motion endpoints (jog, replay, sim hand) only act in IDLE, and still go through every arm guard.
"""
from __future__ import annotations

import csv
import io
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
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
