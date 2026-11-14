"""The Spoonful state machine. Runs in its own thread at fsm.tick_hz; everything else posts events to it.

Safety contract (R2): the FSM installs `permission()` on the Arm. The arm thread calls it for every setpoint.
Inside the face zone, any setpoint that brings the spoon tip closer to the mouth target is refused unless the
FSM is DELIVERING and the gate verdict from the last tick is a permit (or the operator has explicitly entered
calibration mode with no diner). Moving away is always allowed.

The transition table below is the only list of legal transitions; `_go()` refuses anything else and
scripts/make_safety_table.py renders it into SAFETY.md.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..arm import kinematics as K
from ..arm.arm import Arm, Motion, MotionDenied
from ..arm.contact import Contact
from ..arm.trajectory import Library, Pose, lerp_pose, plan
from ..config import ARM_JOINTS, Config
from ..log import Log
from ..perception.face import FaceState
from ..perception.gate import Gate, Verdict
from ..perception.offsets import OffsetTracker
from .pace import Pace
from .profiles import ProfileStore
from .session import Session
from .speech import Speaker


class S(str, Enum):
    BOOT = "BOOT"
    HOMING = "HOMING"
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SCOOPING = "SCOOPING"
    LIFTING = "LIFTING"
    PRE_DELIVERY_HOLD = "PRE_DELIVERY_HOLD"
    DELIVERING = "DELIVERING"
    MOUTH_HOLD = "MOUTH_HOLD"
    RETREAT = "RETREAT"
    PAUSED = "PAUSED"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


ANY = "*"
ACTIVE = {S.LISTENING, S.SCOOPING, S.LIFTING, S.PRE_DELIVERY_HOLD, S.DELIVERING, S.MOUTH_HOLD, S.RETREAT}
BITE_STATES = {S.SCOOPING, S.LIFTING, S.PRE_DELIVERY_HOLD, S.DELIVERING, S.MOUTH_HOLD, S.RETREAT}

# (from, to, trigger). ANY = from every state except the target itself.
TRANSITIONS: list[tuple[str, S, str]] = [
    (S.BOOT, S.HOMING, "arm connected, trajectories validated"),
    (S.HOMING, S.IDLE, "home pose reached"),
    (S.IDLE, S.LISTENING, "session start"),
    (S.LISTENING, S.IDLE, "session end / \"that's enough\""),
    (S.LISTENING, S.SCOOPING, "bite request: voice \"bite\", button, UI, or pace offer (auto-pace opted in)"),
    (S.SCOOPING, S.LIFTING, "scoop path done (outside face zone, no permit needed)"),
    (S.LIFTING, S.PRE_DELIVERY_HOLD, "lift path done, spoon at pre_delivery (outside face zone)"),
    (S.PRE_DELIVERY_HOLD, S.DELIVERING, "gate PERMIT, plus a GO press within go_window_s if gate.require_go (one press per approach)"),
    (S.PRE_DELIVERY_HOLD, S.RETREAT, "no permit for pre_delivery_timeout_s, or retreat_after_s after a back-off"),
    (S.DELIVERING, S.PRE_DELIVERY_HOLD, "permit lost: back off backoff_mm, wait again"),
    (S.DELIVERING, S.RETREAT, "permit lost twice in one bite, or contact near the face"),
    (S.DELIVERING, S.MOUTH_HOLD, "spoon tip within mouth_arrive_mm of the (re-targeted) mouth"),
    (S.MOUTH_HOLD, S.RETREAT, "mouth closed mouth_closed_taken_s (taken), hold_s elapsed, or diner turned away"),
    (S.RETREAT, S.LISTENING, "retreat done, bite logged"),
    (S.RETREAT, S.IDLE, "retreat done after \"that's enough\""),
    (ANY, S.PAUSED, "voice \"pause\" / UI PAUSE (from an active state): hold position"),
    (S.PAUSED, S.LISTENING, "resume with the spoon at rest"),
    (S.PAUSED, S.RETREAT, "resume mid-bite: retreat first"),
    (ANY, S.STOPPED, "STOP switch off or unreadable / STOP button / spacebar / voice \"stop\" / UI STOP: hold, torque off after 1 s (latched)"),
    (S.STOPPED, S.HOMING, "STOP press (button or spacebar) or STOP switch back on, THEN UI confirm within stop_release_window_s; never while the switch is off; GO never releases"),
    (ANY, S.FAULT, "contact outside face zone, HID errors, camera lost > camera_fault_s, watchdog trip: park"),
    (S.FAULT, S.HOMING, "operator reset in the UI"),
]


def allowed(src: S, dst: S) -> bool:
    for f, t, _ in TRANSITIONS:
        if t == dst and (f == src or (f == ANY and src != dst)):
            return True
    return False


@dataclass
class Bite:
    source: str
    ts_request: float = field(default_factory=time.time)
    ts_delivered: Optional[float] = None
    holds: int = 0
    aborts: int = 0
    outcome: str = ""
    ready_spoken: bool = False
    here_spoken: bool = False
    mouth_closed_since: Optional[float] = None
    seen_closed: bool = False
    reached_face_side: bool = False
    hold_since: Optional[float] = None


class FSM:
    def __init__(self, cfg: Config, arm: Arm, lib: Library, gate: Gate, offsets: OffsetTracker,
                 speaker: Speaker, pace: Pace, profiles: ProfileStore, session: Session, log: Log,
                 face_fn: Callable[[], FaceState], camera_age_fn: Callable[[], float] = lambda: 0.0):
        self.cfg, self.arm, self.lib, self.gate, self.offsets = cfg, arm, lib, gate, offsets
        self.speaker, self.pace, self.profiles, self.session, self.log = speaker, pace, profiles, session, log
        self.face_fn, self.camera_age_fn = face_fn, camera_age_fn
        self.state = S.BOOT
        self.since = time.monotonic()
        self.prev = S.BOOT
        self.verdict = Verdict(False, "no_face")
        self.verdict_t = 0.0
        self.face = FaceState()
        self.bite: Optional[Bite] = None
        self.motion: Optional[Motion] = None
        self.motion_step = 0
        self.events: "collections.deque[tuple[str, dict]]" = collections.deque()
        self.fault_reason = ""
        self.stop_release_armed_at: Optional[float] = None
        self.go_at: Optional[float] = None     # last GO press while holding at pre_delivery (gate.require_go)
        self.stop_switch_off = False           # rocker STOP switch at O (or unreadable): no release while set
        self.end_after_bite = False
        self.calibration_mode = False
        self.last_tick = time.monotonic()
        self.tick_ms = 0.0
        self.mouth_target: Optional[Pose] = None
        self.backoff_pose: Optional[Pose] = None
        self.last_offer_check = 0.0
        self.last_gate_log = 0.0
        self.last_verdict_logged: Optional[Verdict] = None
        self.last_said_last = False
        self.food = None
        self.on_summary: Optional[Callable[[int, str], None]] = None
        self.session_last_id: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.listeners: list[Callable[[str, str, str], None]] = []
        arm.permission = self.permission
        arm.on_fault = lambda kind, d: self.post("fault", kind=kind, **d)
        speaker.muted_fn = lambda: not self.profiles.current.speech

    # ================================================================== public API (thread-safe)
    def post(self, kind: str, **kw) -> None:
        self.events.append((kind, kw))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="fsm", daemon=True)
        self._thread.start()

    def stop_thread(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _loop(self) -> None:
        dt = 1.0 / self.cfg.fsm.tick_hz
        self.arm.require_heartbeat(True)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick(t0)
            except Exception as e:  # a bug in a handler must not leave the arm moving
                self.log.fault("fsm_exception", error=repr(e))
                self.arm.hold("fsm exception")
                self._go(S.FAULT, f"fsm_exception {e!r}", force=True)
            self.tick_ms = (time.monotonic() - t0) * 1000
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    # ================================================================== permission (arm thread)
    def mouth_point(self) -> tuple[float, float, float]:
        q = self.mouth_target or (self.lib.pose("mouth_nominal") if self.lib.has("mouth_nominal") else None)
        if q is None:
            b = self.cfg.kinematics.face_zone
            return ((b.x[0] + b.x[1]) / 2, (b.y[0] + b.y[1]) / 2, (b.z[0] + b.z[1]) / 2)
        return K.fk(self.cfg.kinematics, q)

    def permission(self, cur: Pose, nxt: Pose, name: str) -> Optional[str]:
        st = self.state
        if st in (S.STOPPED, S.PAUSED, S.BOOT):
            return f"state_{st.value}"
        kin = self.cfg.kinematics
        if not K.in_face_zone(kin, nxt):
            return None
        m = self.mouth_point()
        approaching = K.dist(K.fk(kin, nxt), m) < K.dist(K.fk(kin, cur), m) - 1e-3
        if not approaching:
            return None
        if self.calibration_mode and st == S.IDLE:
            return None
        if st != S.DELIVERING:
            return "approach_outside_delivering"
        if not self.verdict.ok:
            return f"gate_{self.verdict.reason}"
        if time.monotonic() - self.verdict_t > 0.2:
            return "gate_stale"
        return None

    # ================================================================== tick
    def tick(self, now: float) -> None:
        self.last_tick = now
        self.arm.heartbeat()
        self.face = self.face_fn()
        self.verdict = self.gate.permit(self.face, self.state.value, now)
        self.verdict_t = now
        self._log_gate(now)
        while self.events:
            kind, kw = self.events.popleft()
            self._event(kind, kw, now)
        if self.state in BITE_STATES | {S.LISTENING} and self.camera_age_fn() > self.cfg.fsm.camera_fault_s:
            self._fault("camera_lost", now)
        handler = getattr(self, f"_in_{self.state.value.lower()}", None)
        if handler:
            handler(now)

    def _log_gate(self, now: float) -> None:
        v = self.verdict
        if v != self.last_verdict_logged or now - self.last_gate_log > 0.2:
            f = self.face
            age = (now - f.ts) * 1000 if f.ts else -1
            self.log.gate(v.ok, v.reason, round(f.yaw_deg, 1), round(f.pitch_deg, 1),
                          round(f.mouth_open_ratio, 3),
                          tuple(round(c) for c in f.mouth_center_px) if f.mouth_center_px else None, round(age, 1))
            self.last_verdict_logged, self.last_gate_log = v, now

    # ================================================================== transitions
    def _go(self, dst: S, why: str = "", force: bool = False) -> bool:
        src = self.state
        if src == dst:
            return False
        if not force and not allowed(src, dst):
            self.log.event("illegal_transition", src.value, dst.value, why=why)
            return False
        self.prev, self.state, self.since = src, dst, time.monotonic()
        self.motion, self.motion_step = None, 0
        self.log.event("state", src.value, dst.value, why=why)
        for fn in self.listeners:
            try:
                fn(src.value, dst.value, why)
            except Exception:
                pass
        enter = getattr(self, f"_enter_{dst.value.lower()}", None)
        if enter:
            enter()
        return True

    def _event(self, kind: str, kw: dict, now: float) -> None:
        st = self.state
        src = kw.get("source", "")
        if kind not in ("piezo", "stop_switch"):     # stop_switch repeats every heartbeat: logged on change below
            self.log.event("input", detail_kind=kind, **{k: v for k, v in kw.items() if k != "kind"})
        if kind == "boot":
            if st == S.BOOT:
                self._go(S.HOMING, "boot ok")
        elif kind == "stop":
            if st == S.STOPPED:
                if src in ("button", "key"):
                    self.stop_release_armed_at = now
                    self.log.event("stop_release_armed", detail_source=src)
            else:
                self._go(S.STOPPED, f"stop ({src})")
        elif kind == "button":                 # ESP32 press: STOP everywhere; arms release when stopped
            self._event("stop", {"source": "button"}, now)
        elif kind == "stop_switch":            # rocker level, every change and every heartbeat
            off, was_off = bool(kw.get("off")), self.stop_switch_off
            self.stop_switch_off = off
            if off != was_off:
                self.log.event("input", detail_kind="stop_switch", off=off, lost=bool(kw.get("lost")))
            if off and st != S.STOPPED:
                self._go(S.STOPPED, "stop (STOP switch " + ("unreadable" if kw.get("lost") else "off") + ")")
            elif was_off and not off and st == S.STOPPED:
                self.stop_release_armed_at = now   # flipped back to I: the physical half of a release
                self.log.event("stop_release_armed", detail_source="switch")
        elif kind == "go":                     # GO press: ask for a bite, or confirm the approach while holding
            if st == S.LISTENING:
                self._event("bite", {"source": src or "go"}, now)
            elif st == S.PRE_DELIVERY_HOLD:
                self.go_at = now
                self.log.event("go_pressed", detail_source=src)
        elif kind == "reset":
            if st == S.FAULT:
                self._go(S.HOMING, "operator reset")
            elif st == S.STOPPED:
                armed = self.stop_release_armed_at
                if self.stop_switch_off:
                    self.log.event("stop_release_refused", detail_why="STOP switch is off")
                elif armed and now - armed <= self.cfg.fsm.stop_release_window_s:
                    self.stop_release_armed_at = None
                    self._go(S.HOMING, "stop released: button + UI confirm")
                else:
                    self.log.event("stop_release_refused", detail_why="press the STOP button or spacebar first")
        elif kind == "fault":
            self._fault(kw.get("kind", "fault"), now, kw)
        elif kind == "pause":
            if st in ACTIVE:
                self._go(S.PAUSED, f"pause ({src})")
        elif kind == "resume":
            if st == S.PAUSED:
                if self.bite and self.bite.reached_face_side:
                    self._go(S.RETREAT, "resume mid-bite")
                else:
                    if self.bite:
                        self._finish_bite("aborted")
                    self._go(S.LISTENING, "resume")
        elif kind == "session_start":
            if st == S.IDLE:
                self.session.start(self.profiles.current.name)
                self._refit_pace()
                self._go(S.LISTENING, "session start")
        elif kind in ("session_end", "enough"):
            if st == S.LISTENING:
                self._end_session()
                self._go(S.IDLE, kind)
            elif st in BITE_STATES or st == S.PAUSED:
                self.end_after_bite = True
        elif kind == "bite":
            if st == S.LISTENING:
                self._start_bite(src or "ui", now)
        elif kind in ("slower", "faster"):
            p = self.profiles.current
            k = 1.25 if kind == "slower" else 0.8
            self.profiles.update(pace_scale=round(min(3.0, max(0.4, p.pace_scale * k)), 2),
                                 approach_speed=round(min(1.0, max(0.4, p.approach_speed / k)), 2))
            self._refit_pace()
        elif kind == "profile":
            if st in (S.IDLE, S.LISTENING):
                self.profiles.select(kw["name"])
                self._refit_pace()
        elif kind == "piezo":
            if self.pace.on_piezo(kw.get("envelope", 0.0)):
                self.log.swallow(self.profiles.current.name, kw.get("envelope", 0.0))
        elif kind == "calibration_mode":
            self.calibration_mode = bool(kw.get("on")) and st == S.IDLE
            self.log.event("calibration_mode", detail_on=self.calibration_mode)

    def _fault(self, kind: str, now: float, detail: Optional[dict] = None) -> None:
        if self.state in (S.FAULT, S.STOPPED):
            return
        self.fault_reason = kind
        self.log.fault(kind, state=self.state.value, **(detail or {}))
        if self.session.active:
            self.session.faults += 1
        self._go(S.FAULT, kind)

    # ================================================================== bite bookkeeping
    def _start_bite(self, source: str, now: float) -> None:
        self.bite = Bite(source)
        self.go_at = None
        self.offsets.reset()
        self.mouth_target = None
        self._go(S.SCOOPING, f"bite ({source})")

    def _finish_bite(self, outcome: str) -> None:
        b = self.bite
        if not b:
            return
        b.outcome = b.outcome or outcome
        done = time.time()
        self.log.bite(session_id=self.session.id, profile=self.profiles.current.name, ts_request=b.ts_request,
                      ts_delivered=b.ts_delivered, ts_done=done, outcome=b.outcome, holds=b.holds,
                      aborts=b.aborts, offset_mm=round(self.offsets.offset_mm, 1), source=b.source,
                      depth_mm=round(self.offsets.depth_mm, 1))
        if self.session.active:
            self.session.record_bite(b.outcome, done)
        self.bite = None
        self._refit_pace()

    def _refit_pace(self) -> None:
        prof = self.profiles.current
        iv = self.pace.refit(self.log.bites_for(prof.name), prof)
        if self.pace.learned and abs(prof.learned_interval_s - iv) > 0.5:
            self.profiles.update(learned_interval_s=round(iv, 1))

    def _end_session(self) -> None:
        self.end_after_bite = False
        self.session_last_id = self.session.id
        text = self.session.end(self.pace.status()["label"])
        if text and self.on_summary:
            self.on_summary(self.session_last_id, text)

    # ================================================================== motion helpers
    def _need(self, *names: str) -> Optional[str]:
        for n in names:
            if not self.lib.has(n):
                return n
        return None

    def _run_paths(self, items: list[tuple[str, Optional[str]]], speed_factor: float = 1.0, name: str = "") -> bool:
        """items: [(trajectory name, speed override)]. Paths and poses mix freely."""
        segs = []
        for n, sp in items:
            tf = self.lib.items[n]
            segs.append((tf.poses, sp or tf.speed, n))
        try:
            p = plan(self.cfg, self.arm.current(), segs, speed_factor, name or "+".join(n for n, _ in items))
        except Exception as e:
            self._fault("plan_error", time.monotonic(), {"error": str(e)})
            return False
        self.motion = self.arm.run(p)
        return True

    def _motion_result(self) -> Optional[str]:
        """None while running; 'ok'; or an error kind."""
        m = self.motion
        if m is None or not m.done.is_set():
            return None
        if m.error is None:
            return "ok"
        if isinstance(m.error, Contact):
            return "contact"
        if isinstance(m.error, MotionDenied):
            return f"denied:{m.error}"
        return f"error:{m.error!r}"

    def _handle_motion_failure(self, res: str, now: float) -> None:
        if res == "contact":
            c = self.arm.last_contact
            if c and c.in_face_zone:
                if self.bite:
                    self.bite.outcome = "contact"
                self._go(S.RETREAT, "contact near face")
            else:
                self._fault("contact", now, {"joint": c.joint if c else "?"})
        else:
            self._fault("motion_failed", now, {"result": res})

    def _home_plan_items(self) -> list[tuple[list[Pose], str, str]]:
        cur = self.arm.current()
        items = []
        if K.in_face_zone(self.cfg.kinematics, cur) and self.lib.has("pre_delivery"):
            items.append(([self.lib.pose("pre_delivery")], "near_face", "away"))
        items.append(([self.cfg.arm.home_pose | {j: cur[j] for j in ARM_JOINTS if j not in self.cfg.arm.home_pose}],
                      "normal", "home"))
        return items

    # ================================================================== state handlers
    def _enter_homing(self) -> None:
        try:
            if not self.arm.torque_on:
                self.arm.torque(True)
            p = plan(self.cfg, self.arm.current(), self._home_plan_items(), 0.8, "homing")
            self.motion = self.arm.run(p)
        except Exception as e:
            self._fault("homing_failed", time.monotonic(), {"error": str(e)})
            return
        if self.cfg.arm.gripper_close_deg is not None:
            try:
                self.arm.gripper(self.cfg.arm.gripper_close_deg)
            except Exception as e:
                self.log.fault("gripper", error=str(e))

    def _in_homing(self, now: float) -> None:
        r = self._motion_result()
        if r == "ok":
            self._go(S.IDLE, "home")
        elif r:
            self._handle_motion_failure(r, now)

    def _in_listening(self, now: float) -> None:
        prof = self.profiles.current
        if not (prof.auto_pace and self.cfg.pace.auto_offer):
            return
        if now - self.last_offer_check < 0.5:
            return
        self.last_offer_check = now
        t = self.pace.suggest_next_offer_time(self.session.last_bite_done)
        if t and time.time() >= t:
            self._start_bite("pace", now)

    def _enter_scooping(self) -> None:
        prof = self.profiles.current
        scoop = prof.scoop_path if self.lib.has(prof.scoop_path) else "scoop"
        missing = self._need(scoop, "lift", "pre_delivery")
        if missing:
            self._fault("missing_trajectory", time.monotonic(), {"name": missing})
            return
        self._run_paths([(scoop, "normal")], name="scoop")

    def _in_scooping(self, now: float) -> None:
        r = self._motion_result()
        if r == "ok":
            self._go(S.LIFTING, "scooped")
        elif r:
            self._handle_motion_failure(r, now)

    def _enter_lifting(self) -> None:
        self._run_paths([("lift", "normal"), ("pre_delivery", "normal")], name="lift")

    def _in_lifting(self, now: float) -> None:
        r = self._motion_result()
        if r == "ok":
            if self.bite:
                self.bite.reached_face_side = True
            self._go(S.PRE_DELIVERY_HOLD, "at pre_delivery")
        elif r:
            self._handle_motion_failure(r, now)

    def _enter_pre_delivery_hold(self) -> None:
        if self.backoff_pose is not None:
            self.motion = self.arm.goto(self.backoff_pose, "near_face", name="backoff")
            self.backoff_pose = None
        else:
            self.arm.hold("pre_delivery")
        b = self.bite
        if b:
            b.holds += 1
            if not b.ready_spoken:
                b.ready_spoken = self.speaker.say("ready") or True

    def _in_pre_delivery_hold(self, now: float) -> None:
        r = self._motion_result()
        if r and r != "ok":
            self._handle_motion_failure(r, now)
            return
        if self.motion and r is None:
            return  # still backing off
        b = self.bite
        waited = now - self.since
        limit = self.cfg.fsm.pre_delivery_timeout_s
        if b and b.aborts:
            limit = min(limit, self.cfg.gate.retreat_after_s)
        if self.verdict.ok and self._go_fresh(now):
            self.go_at = None                  # one press per approach: a back-off needs a new press
            self._go(S.DELIVERING, "permit" + (" + GO" if self.cfg.gate.require_go else ""))
        elif waited >= limit:
            if b:
                b.outcome = "timeout"
            self.speaker.say("wait")
            self._go(S.RETREAT, f"no permit for {waited:.0f} s")

    def _go_fresh(self, now: float) -> bool:
        if not self.cfg.gate.require_go:
            return True
        return self.go_at is not None and now - self.go_at <= self.cfg.gate.go_window_s

    def _delivery_target(self, now: float) -> Pose:
        base = self.lib.pose("mouth_nominal") if self.lib.has("mouth_nominal") else self.lib.path("present")[-1]
        dq = self.offsets.update(self.face, now) if self.face.present else self.offsets.joint_delta()
        t = {j: base[j] + dq.get(j, 0.0) for j in ARM_JOINTS}
        for j in ARM_JOINTS:
            lo, hi = self.cfg.arm.soft_limits_deg[j]
            t[j] = min(hi, max(lo, t[j]))
        return t

    def _enter_delivering(self) -> None:
        if self._need("mouth_nominal") and self._need("present"):
            self._fault("missing_trajectory", time.monotonic(), {"name": "mouth_nominal"})
            return
        b = self.bite
        if b and b.ts_delivered is None:
            b.ts_delivered = time.time()
        if b and not b.here_spoken:
            b.here_spoken = self.speaker.say("here") or True
        self.arm.last_contact = None

    def _in_delivering(self, now: float) -> None:
        b = self.bite
        if self.arm.last_contact is not None:
            self._handle_motion_failure("contact", now)
            return
        if not self.verdict.ok:
            self.arm.hold("permit lost")
            if b:
                b.aborts += 1
            if b and b.aborts >= self.cfg.fsm.max_aborts_per_bite:
                b.outcome = "aborted"
                self._go(S.RETREAT, f"permit lost twice ({self.verdict.reason})")
            else:
                self.backoff_pose = self._backoff_pose()
                self._go(S.PRE_DELIVERY_HOLD, f"permit lost ({self.verdict.reason})")
            return
        tgt = self._delivery_target(now)
        self.mouth_target = tgt
        self.arm.servo_toward(tgt, "near_face", self.profiles.current.approach_speed)
        if K.tip_distance(self.cfg.kinematics, self.arm.setpoint, tgt) <= self.cfg.arm.mouth_arrive_mm:
            self._go(S.MOUTH_HOLD, "at mouth")

    def _backoff_pose(self) -> Pose:
        """Point backoff_mm back along the joint-space line toward pre_delivery."""
        cur = self.arm.current()
        pre = self.lib.pose("pre_delivery")
        d = K.tip_distance(self.cfg.kinematics, cur, pre)
        if d <= 1e-3:
            return pre
        s = min(1.0, self.cfg.gate.backoff_mm / d)
        return lerp_pose(cur, pre, s)

    def _enter_mouth_hold(self) -> None:
        self.arm.hold("mouth_hold")
        if self.bite:
            self.bite.hold_since = time.monotonic()

    def _in_mouth_hold(self, now: float) -> None:
        b = self.bite
        f, c = self.face, self.cfg.face
        if self.verdict.reason in ("no_face", "looking_away", "frame_stale", "face_too_brief"):
            if b:
                b.outcome = "taken" if b.seen_closed else "aborted"
            self._go(S.RETREAT, f"diner {self.verdict.reason}")
            return
        closed = f.present and f.mouth_open_ratio < c.mouth_open_ratio_threshold * c.hysteresis_fraction
        if b:
            if closed:
                b.mouth_closed_since = b.mouth_closed_since or now
                if now - b.mouth_closed_since >= c.mouth_closed_taken_s:
                    b.seen_closed = True
                    b.outcome = "taken"
                    self._go(S.RETREAT, "mouth closed: taken")
                    return
            else:
                b.mouth_closed_since = None
        if now - self.since >= self.profiles.current.hold_s:
            if b:
                b.outcome = "taken" if b.seen_closed else "timeout"
            self._go(S.RETREAT, "hold time")

    def _enter_retreat(self) -> None:
        cur = self.arm.current()
        items: list[tuple[list[Pose], str, str]] = []
        if self.lib.has("pre_delivery") and (K.in_face_zone(self.cfg.kinematics, cur) or
                                             (self.bite and self.bite.reached_face_side)):
            items.append(([self.lib.pose("pre_delivery")], "near_face", "away"))
            if self.lib.has("retreat"):
                # brief: reverse of present at normal speed; the arm guard still caps it inside the face zone
                items.append((self.lib.path("retreat"), "normal", "retreat"))
        rest = self.lib.pose("rest") if self.lib.has("rest") else self.cfg.arm.home_pose
        items.append(([rest], "normal", "rest"))
        try:
            self.motion = self.arm.run(plan(self.cfg, cur, items, 1.0, "retreat"))
        except Exception as e:
            self._fault("plan_error", time.monotonic(), {"error": str(e)})

    def _in_retreat(self, now: float) -> None:
        r = self._motion_result()
        if r is None:
            return
        if r != "ok":
            self._finish_bite("contact" if r == "contact" else "aborted")
            self._fault("retreat_failed", now, {"result": r})
            return
        self._finish_bite("aborted")
        self.mouth_target = None
        food = self.food
        if food is not None and food.bowl_empty and not self.last_said_last:
            self.last_said_last = self.speaker.say("last")
        if self.end_after_bite:
            self._end_session()
            self._go(S.IDLE, "that's enough")
        else:
            self._go(S.LISTENING, "bite done")

    def _enter_paused(self) -> None:
        self.arm.hold("paused")
        if self.session.active:
            self.session.pauses += 1
        self.speaker.say("paused")

    def _enter_stopped(self) -> None:
        self.arm.stop()
        self.speaker.say("stopped")
        self.stop_release_armed_at = None
        if self.bite:
            self._finish_bite("aborted")

    def _enter_fault(self) -> None:
        self.arm.hold("fault")
        try:
            self.motion = self.arm.park()
        except Exception as e:
            self.log.fault("park_failed", error=str(e))
        self.speaker.say("break")
        if self.bite:
            self._finish_bite("contact" if self.fault_reason == "contact" else "aborted")

    def _enter_idle(self) -> None:
        self.calibration_mode = False

    def _enter_listening(self) -> None:
        self.mouth_target = None

    # ================================================================== status for the UI
    def countdown(self, now: Optional[float] = None) -> Optional[float]:
        now = now or time.monotonic()
        if self.state == S.PRE_DELIVERY_HOLD:
            lim = self.cfg.fsm.pre_delivery_timeout_s
            if self.bite and self.bite.aborts:
                lim = min(lim, self.cfg.gate.retreat_after_s)
            return max(0.0, lim - (now - self.since))
        if self.state == S.MOUTH_HOLD:
            return max(0.0, self.profiles.current.hold_s - (now - self.since))
        return None

    def explain(self) -> str:
        """One plain sentence: what the arm is doing and why it is waiting (readable from 2 m)."""
        st, v = self.state, self.verdict
        reasons = {"no_face": "no one in view", "face_too_brief": "making sure you're there",
                   "looking_away": "waiting for you to face it", "mouth_closed": "waiting for an open mouth",
                   "mouth_outside_window": "mouth outside the delivery window", "frame_stale": "camera lagging",
                   "paused": "paused", "stop_latched": "stopped"}
        return {
            S.BOOT: "Starting up", S.HOMING: "Moving to home",
            S.IDLE: "Idle. Start a session to begin" + (" (CALIBRATION MODE: no diner)" if self.calibration_mode else ""),
            S.LISTENING: "Waiting for a bite request", S.SCOOPING: "Scooping", S.LIFTING: "Lifting the spoon",
            S.PRE_DELIVERY_HOLD: (f"Holding: {reasons.get(v.reason, 'ready')}" if not v.ok
                                  else "Ready: press the button to bring the spoon" if not self._go_fresh(time.monotonic())
                                  else "Permit: approaching"),
            S.DELIVERING: "Bringing the spoon to you", S.MOUTH_HOLD: "Holding still at your mouth",
            S.RETREAT: "Moving the spoon away", S.PAUSED: "Paused. Say resume or press RESUME",
            S.FAULT: f"Taking a break: {self.fault_reason}. Operator reset needed",
            S.STOPPED: ("STOPPED. Flip the STOP switch back on, then confirm in the app" if self.stop_switch_off
                        else "STOPPED. Press the STOP button, then confirm in the app"
                        if self.cfg.esp32.button_role == "stop" or self.cfg.esp32.stop_switch == "button"
                        else "STOPPED. Press the spacebar, then confirm in the app"),
        }[st]

    def status(self) -> dict:
        b = self.bite
        return {
            "state": self.state.value, "since_s": round(time.monotonic() - self.since, 1),
            "explain": self.explain(), "countdown": self.countdown(),
            "gate": {"ok": self.verdict.ok, "reason": self.verdict.reason},
            "bite": None if not b else {"source": b.source, "aborts": b.aborts, "holds": b.holds},
            "fault": self.fault_reason if self.state == S.FAULT else "",
            "stop_release_armed": bool(self.stop_release_armed_at),
            "stop_switch_off": self.stop_switch_off,
            "go_needed": self.state == S.PRE_DELIVERY_HOLD and not self._go_fresh(time.monotonic()),
            "calibration_mode": self.calibration_mode, "end_after_bite": self.end_after_bite,
            "offset_mm": round(self.offsets.offset_mm, 1), "depth_corr_mm": round(self.offsets.depth_mm, 1),
            "tick_ms": round(self.tick_ms, 2),
            "session": {"active": self.session.active, "id": self.session.id, "bites": self.session.bites,
                        "taken": self.session.taken, "pauses": self.session.pauses},
        }
