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
from ..arm.scoop_style import adapt_scoop, dip_index, shape_scoop, shift_scoop
from .aim_check import MAX_FOLLOW_MM
from ..arm.trajectory import Library, Pose, check_limits, lerp_pose, plan
from ..config import ARM_JOINTS, Config, ScoopStyle
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
    (S.PRE_DELIVERY_HOLD, S.SCOOPING, "face camera sees an empty spoon: re-scoop (at most plate.spoon_check.max_rescoops)"),
    (S.DELIVERING, S.PRE_DELIVERY_HOLD, "permit lost: back off backoff_mm, wait again"),
    (S.DELIVERING, S.RETREAT, "permit lost twice in one bite, or contact near the face"),
    (S.DELIVERING, S.MOUTH_HOLD, "spoon tip within mouth_arrive_mm of the (re-targeted) mouth"),
    (S.MOUTH_HOLD, S.RETREAT, "mouth closed mouth_closed_taken_s (taken), hold_s elapsed, or diner turned away"),
    (S.RETREAT, S.LISTENING, "retreat done, bite logged"),
    (S.RETREAT, S.IDLE, "retreat done after \"that's enough\""),
    (ANY, S.PAUSED, "voice \"pause\" / UI PAUSE (from an active state): hold position"),
    (S.PAUSED, S.LISTENING, "resume with the spoon at rest"),
    (S.PAUSED, S.RETREAT, "resume mid-bite: retreat first"),
    (ANY, S.STOPPED, "STOP switch off or unreadable / STOP button / spacebar / voice \"stop\" / UI STOP: hold position (torque off after arm.stop_torque_off_after_s; < 0 = keeps holding) (latched)"),
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
    bowl: Optional[str] = None
    texture: Optional[str] = None
    looks: int = 0                          # webcam aim checks made for this scoop
    attack: Optional[str] = None            # scoop strategy used (skim / dig / stab_and_saw / scrape / ladle)
    rescoops: int = 0
    spoon: Optional[dict] = None            # last spoon check answer
    spoon_pending_since: Optional[float] = None
    probe_touch_mm: Optional[float] = None  # where the spoon felt resistance, relative to the taught dip (+ = higher)
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
        self._probe_job: Optional[dict] = None
        self._perm_motion: Optional[str] = None   # permission(): the motion being checked and its closest approach
        self._perm_min_d = 0.0
        self._perm_from_inside = False    # probe scoop in progress (see _start_probe_scoop)
        self._home_retried = False             # one automatic homing retry per reset (see _handle_motion_failure)
        self._home_retry_at: Optional[float] = None
        self.plate = None                      # PlateWatcher (perception/plate.py), set by System
        self.last_bowl: Optional[str] = None
        self.bite_queue: list[str] = []
        self.aim = None                        # aim_check.AimCheck (System): the webcam checks the aim before a scoop
        self._look: Optional[dict] = None
        self.food_surface = None               # photo_map.FoodSurface (System): predict -> feel -> update        # planner: "a little of each" -> one bowl per GO press   # GO / plain "bite" repeats the last food asked for
        self.on_summary: Optional[Callable[[int, str], None]] = None
        self.session_last_id: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.listeners: list[Callable[[str, str, str], None]] = []
        arm.permission = self.permission
        arm.on_fault = lambda kind, d: self.post("fault", kind=kind, **d)
        speaker.muted_fn = lambda: not self.profiles.current.speech

    # ================================================================== public API (thread-safe)
    def post(self, kind: str, /, **kw) -> None:   # positional-only: a fault may carry its own kind=
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

    APPROACH_SLACK_MM = 15

    def permission(self, cur: Pose, nxt: Pose, name: str) -> Optional[str]:
        st = self.state
        if st in (S.STOPPED, S.PAUSED, S.BOOT):
            return f"state_{st.value}"
        kin = self.cfg.kinematics
        if not K.in_face_zone(kin, nxt):
            return None
        m = self.mouth_point()
        d_cur, d_nxt = K.dist(K.fk(kin, cur), m), K.dist(K.fk(kin, nxt), m)
        if st == S.DELIVERING:
            approaching = d_nxt < d_cur - 1e-3                 # strict: every mm closer needs the gate
        else:
            # Outside DELIVERING nothing may head for the face. But a move AWAY from it (homing or retreating from
            # the mouth) can start a millimetre closer, because the loaded arm sags below its setpoint. So: closer
            # than this motion has ever been, by more than APPROACH_SLACK_MM.
            # Only a motion that STARTED inside the face zone (leaving it) gets that slack; anything coming from
            # outside may not enter the zone toward the mouth at all.
            if name != self._perm_motion:
                self._perm_motion, self._perm_min_d = name, d_cur
                self._perm_from_inside = K.in_face_zone(kin, cur)
            if self._perm_from_inside:
                approaching = d_nxt < self._perm_min_d - self.APPROACH_SLACK_MM
                self._perm_min_d = min(self._perm_min_d, d_nxt + self.APPROACH_SLACK_MM)
            else:
                approaching = d_nxt < d_cur - 1e-3
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
                self._home_retried = False
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
            if st == S.IDLE:                   # forgot "Start session": GO starts it and asks for the bite
                self._event("session_start", {"source": src or "go"}, now)
                if self.state == S.LISTENING:
                    self._event("bite", {"source": src or "go"}, now)
            elif st == S.LISTENING:
                self._event("bite", {"source": src or "go"}, now)
            elif st == S.PRE_DELIVERY_HOLD:
                self.go_at = now
                self.log.event("go_pressed", detail_source=src)
        elif kind == "reset":
            if st == S.FAULT:
                self._home_retried = False
                self._go(S.HOMING, "operator reset")
            elif st == S.STOPPED:
                armed = self.stop_release_armed_at
                if self.stop_switch_off:
                    self.log.event("stop_release_refused", detail_why="STOP switch is off")
                elif armed and now - armed <= self.cfg.fsm.stop_release_window_s:
                    self.stop_release_armed_at = None
                    self._home_retried = False
                    self._go(S.HOMING, "stop released: button + UI confirm")
                else:
                    self.log.event("stop_release_refused", detail_why="press the STOP button or spacebar first")
        elif kind == "fault":
            self._fault(kw.get("kind", "fault"), now, {k: v for k, v in kw.items() if k != "kind"})
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
        elif kind == "select":                 # voice picked a food: remember it, say so, move nothing (GO bites)
            if st in (S.IDLE, S.LISTENING, S.RETREAT, S.PRE_DELIVERY_HOLD, S.MOUTH_HOLD):
                bowl = self._choose_bowl(kw.get("bowl"), announce=False)
                if bowl:
                    self.bite_queue = []
                    food = self.plate.food_of(bowl) or "that"
                    if not kw.get("quiet"):
                        self.speaker.say_text(f"{food.capitalize()}. Ready when you are"
                                              if st in (S.IDLE, S.LISTENING) else f"{food.capitalize()} next")
                    self.log.event("food_selected", detail_bowl=bowl, food=food)
        elif kind == "queue":                  # planner: several bites, one per GO press, in this order
            bowls = [b for b in (kw.get("bowls") or []) if self.plate and b in self.plate.bowls][:5]
            if bowls:
                self.bite_queue = list(bowls)
                self.last_bowl = bowls[-1]
                self.log.event("bite_queue", detail_bowls=bowls)
        elif kind == "bite":
            if st == S.LISTENING:
                asked = kw.get("bowl")
                if asked is None and self.bite_queue:
                    asked = self.bite_queue.pop(0)          # next one the planner lined up
                bowl = self._choose_bowl(asked, announce=bool(kw.get("bowl")))
                if bowl is not False:
                    self._start_bite(src or "ui", now, bowl)
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
        self.log.fault(kind, state=self.state.value, **{k: v for k, v in (detail or {}).items() if k not in ("kind", "state")})
        if self.session.active:
            self.session.faults += 1
        self._go(S.FAULT, kind)

    # ================================================================== bite bookkeeping
    def _choose_bowl(self, asked: Optional[str], announce: bool = True):
        """Which bowl to scoop from. None = the profile's default scoop (no plate set up). False = refuse (said why)."""
        p = self.plate
        if p is None or not p.enabled:
            return None
        bowl = asked or self.last_bowl
        if bowl is None and len(p.bowls) == 1:
            bowl = next(iter(p.bowls))          # one bowl on the table: a plain "bite" means that one
        if bowl is None:
            return None
        if bowl not in p.bowls:
            self.log.event("bowl_refused", detail_bowl=bowl, why="unknown bowl")
            return False
        food = p.food_of(bowl) or "that"
        if not self.lib.has(p.scoop_for(bowl)):
            self.speaker.say_text(f"I haven't learned to scoop the {food} yet")
            self.log.event("bowl_refused", detail_bowl=bowl, why=f"{p.scoop_for(bowl)} not taught")
            return False
        if p.is_empty(bowl):
            self.speaker.say_text(f"The {food} is finished" if food != "that" else "That bowl is empty")
            self.log.event("bowl_refused", detail_bowl=bowl, why="empty")
            return False
        shift = p.shift_for(bowl)
        if shift is not None:
            moved = math.hypot(*shift)
            why = ""
            if moved > self.cfg.plate.track.max_shift_mm:
                why = f"moved {moved:.0f} mm from its taught spot"
            else:
                sh = shift_scoop(self.cfg.kinematics, self.lib.items[p.scoop_for(bowl)].poses, *shift)
                if sh is None:
                    why = f"moved {moved:.0f} mm: out of reach"
                elif any(K.in_face_zone(self.cfg.kinematics, q) for q in sh):
                    why = f"moved {moved:.0f} mm: too close to the diner's face"
                else:
                    try:
                        plan(self.cfg, sh[0], [(sh, "normal", "check")])
                    except Exception:
                        why = f"moved {moved:.0f} mm: outside the arm's joint limits"
            if why:
                what = f"the {food}" if food != "that" else "the bowl"
                self.speaker.say_text(f"I can't reach {what} there. Please move it closer to its spot")
                self.log.event("bowl_refused", detail_bowl=bowl, why=why)
                return False
        if asked and announce:
            self.speaker.say_text(f"{food.capitalize()}, coming up" if food != "that" else "Coming up")
        self.last_bowl = bowl
        return bowl

    def _start_bite(self, source: str, now: float, bowl: Optional[str] = None) -> None:
        self.bite = Bite(source, bowl=bowl)
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
        if res == "contact" and self.state == S.HOMING and not self._home_retried:
            # The xArm controller sometimes ignores moves for a second or two (seen on the first move after start).
            # Homing is away from the face: wait, retry once; a real obstruction faults on the second try.
            self._home_retried = True
            c = self.arm.last_contact
            self.log.event("homing_retry", detail_joint=c.joint if c else "?")
            self._home_retry_at = now + 1.0
            self.motion = None
            return
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
        # A joint pushed past its servo's 0..1000 range (by hand, or the elbow's travel) can't drive itself back:
        # homing would just stall. Say which one so the operator can move it by hand with the power off.
        if self.arm.read_age_s() > 2.0:
            self._fault("no readings from the arm. Is its power switch on? Then restart the app",
                        time.monotonic(), {"read_age_s": round(min(self.arm.read_age_s(), 1e6), 1)})
            return
        units = (self.arm.status() or {}).get("units") or {}
        bad = [f"{j} (reads {u})" for j, u in units.items() if j in ARM_JOINTS and isinstance(u, (int, float))
               and not 0 <= u <= 1000]   # past the end: the servo cannot drive back
        if bad:
            self._fault("joint out of servo range: " + ", ".join(bad) + ". Arm power OFF, bend it back by hand",
                        time.monotonic(), {"units": units})
            return
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
        if self._home_retry_at is not None:
            if now >= self._home_retry_at:
                self._home_retry_at = None
                self._enter_homing()
            return
        r = self._motion_result()
        if r == "ok":
            self._home_retried = False
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
        self._look = None
        if self.bite:
            self.bite.looks = 0
        self._scoop_begin()

    # ------------------------------------------------------------------ look before scooping
    LOOK_TIMEOUT_S = 6.0
    CALIBRATE_TIMEOUT_S = 90.0
    LOOK_AGAIN_MM = 15          # after following the bowl this far, look once more from the new spot

    def _start_look(self) -> bool:
        """Pause above the bowl and ask the webcam whether the spoon is over it (brain/aim_check.py)."""
        b = self.bite
        if not (self.cfg.plate.look_before_scoop and self.aim is not None and b and b.bowl and b.looks < 2):
            return False
        if not self.aim.ready(b.bowl):
            if b.looks or not self.aim.can_calibrate(b.bowl):
                return False
            # first scoop from this bowl: let the camera learn its view of it and correct the aim (about 30 s, once)
            food = self.plate.food_of(b.bowl) or "bowl"
            self.speaker.say_text(f"Lining up with the {food}")
            self.aim.calibrate_async(b.bowl)
            self._look = {"stage": "calibrating", "bowl": b.bowl, "t": time.monotonic()}
            return True
        try:
            pose = self.aim.aim_pose(b.bowl)
            self.motion = self.arm.run(plan(self.cfg, self.arm.current(), [([pose], "normal", "aim")], 1.0, "aim"))
        except Exception as e:
            self.log.event("look_skipped", detail_error=str(e)[:160])
            return False
        self._look = {"stage": "moving", "bowl": b.bowl}
        return True

    def _in_look(self, now: float) -> None:
        lk = self._look
        if lk["stage"] == "calibrating":
            if self.aim.take_result() is None and now - lk["t"] < self.CALIBRATE_TIMEOUT_S:
                return
            self._look = None
            self.bite.looks = 2                         # the check just looked (twice): scoop now
            self.log.event("look", detail_bowl=lk["bowl"], first=True, confirmed=self.aim.ready(lk["bowl"]))
            self._scoop_begin(look=False)
            return
        if lk["stage"] == "moving":
            r = self._motion_result()
            if r is None:
                return
            if r != "ok":
                self._look = None
                self._handle_motion_failure(r, now)
                return
            lk.update(stage="looking", t=now)
            self.aim.look_async(lk["bowl"])
            return
        res = self.aim.take_result()
        if res is None and now - lk["t"] < self.LOOK_TIMEOUT_S:
            return
        self._look = None
        self.bite.looks += 1
        again = False
        if res and "offset" in res:
            f, l = res["offset"]
            off = math.hypot(f, l)
            moved = False
            if 4 <= off <= MAX_FOLLOW_MM:
                moved = self.aim.follow(lk["bowl"], f, l)
                again = moved and off >= self.LOOK_AGAIN_MM
                if moved and off >= self.LOOK_AGAIN_MM and self.bite.looks == 1:
                    food = self.plate.food_of(lk["bowl"]) or "bowl"
                    self.speaker.say_text(f"The {food} moved. Following it")
            elif off > MAX_FOLLOW_MM:
                self.speaker.say_text("I can't line up with that bowl. Please check it")
            self.log.event("look", detail_bowl=lk["bowl"], further_mm=round(f, 1), left_mm=round(l, 1),
                           off_mm=round(off, 1), followed=moved)
            if off > MAX_FOLLOW_MM:
                self.bite.outcome = "aim_lost"
                self._go(S.RETREAT, "bowl not where expected")
                return
        else:
            self.log.event("look", detail_bowl=lk["bowl"], seen=False, why=(res or {}).get("error", "no answer in time"))
        self._scoop_begin(look=again)

    def _scoop_begin(self, look: bool = True) -> None:
        self._probe_job = None
        prof = self.profiles.current
        scoop = prof.scoop_path if self.lib.has(prof.scoop_path) else "scoop"
        if self.bite and self.bite.bowl and self.plate is not None:
            scoop = self.plate.scoop_for(self.bite.bowl)   # checked taught in _choose_bowl
        missing = self._need(scoop, "lift", "pre_delivery")
        if missing:
            self._fault("missing_trajectory", time.monotonic(), {"name": missing})
            return
        if not (self.bite and self.bite.bowl and self.plate is not None):
            self._run_paths([(scoop, "normal")], name="scoop")
            return
        if look and self._start_look():
            return
        # How hard to dig depends on the food: same taught scoop, deeper / slower / with a saw for firm food.
        texture, why = self.plate.texture_of(self.bite.bowl)
        self.bite.texture = texture
        attack = self.plate.attack_of(self.bite.bowl)
        style = ScoopStyle(depth_mm=attack["depth_mm"], speed=attack["speed"], wiggle=attack["saw_strokes"],
                           contact_extra_deg=attack["contact_extra_deg"])
        self.bite.attack = attack["strategy"]
        self.log.event("attack_plan", detail_bowl=self.bite.bowl, **{k: attack[k] for k in attack})
        spoken = self.plate.bowls.get(self.bite.bowl, {})
        if self.cfg.plate.explain_attack and attack.get("why") and not spoken.get("attack_explained"):
            food = self.plate.food_of(self.bite.bowl) or "this"
            self.speaker.say_text(f"{food.capitalize()}: {attack['why'].rstrip('.')}")
            spoken["attack_explained"] = True
        base_poses = self.lib.items[scoop].poses
        shift = self.plate.shift_for(self.bite.bowl)
        if shift is not None and math.hypot(*shift) >= 5:
            moved = shift_scoop(self.cfg.kinematics, base_poses, *shift)
            if moved is not None:
                base_poses = moved
                self.log.event("scoop_shift", detail_bowl=self.bite.bowl, dx_mm=round(shift[0]), dy_mm=round(shift[1]))
            else:
                shift = None
        else:
            shift = None
        probe = self.cfg.plate.probe.enabled
        shaped_poses, shaped = shape_scoop(self.cfg.kinematics, base_poses, attack["entry_angle_deg"], attack["drag_mm"],
                                            self.cfg.arm.soft_limits_deg)
        try:
            for q in shaped_poses:
                check_limits(self.cfg, q)
            base_poses = shaped_poses
            self.log.event("scoop_shape", detail_strategy=attack["strategy"], **shaped)
        except Exception as e:                       # that angle isn't reachable here: keep the taught shape
            self.log.event("scoop_shape_skipped", detail_strategy=attack["strategy"], why=str(e)[:120])
        depth = 0.0 if probe else style.depth_mm           # probing finds the depth itself
        poses, info = adapt_scoop(self.cfg.kinematics, base_poses, depth, style.wiggle)
        self.log.event("scoop_style", detail_texture=texture, why=why, food=self.plate.food_of(self.bite.bowl),
                       speed=style.speed, contact_extra_deg=style.contact_extra_deg, probe=probe, **info)
        if probe and self._start_probe_scoop(poses, style, texture):
            return
        if probe:                                    # couldn't feel for the depth here: use the planned depth
            poses, info = adapt_scoop(self.cfg.kinematics, base_poses, style.depth_mm, style.wiggle)
            self.log.event("probe_unavailable", detail_depth_mm=style.depth_mm)
        try:
            p = plan(self.cfg, self.arm.current(), [(poses, "normal", scoop)], style.speed, f"scoop({texture})")
        except Exception as e:                       # a deeper pose outside the soft limits: use the taught one
            self.log.event("scoop_style_fallback", detail_error=str(e)[:200])
            self._run_paths([(scoop, "normal")], speed_factor=style.speed, name="scoop")
            return
        p.contact_extra_deg = style.contact_extra_deg
        self.motion = self.arm.run(p)

    # ------------------------------------------------------------------ probe scoop: feel for the bottom
    def _start_probe_scoop(self, poses: list, style, texture: str) -> bool:
        """Approach to just above the taught dip, descend slowly until the joints feel resistance, then run the
        rest of the scoop from that depth. Three plans in a row (motion_step 0, 1, 2), driven by _in_scooping."""
        from ..arm.scoop_style import lower_tip
        from ..arm.trajectory import Plan, Segment
        kin, pc = self.cfg.kinematics, self.cfg.plate.probe
        if len(poses) < 3:
            return False
        k = dip_index(kin, poses)
        dip = poses[k]

        def within(q):
            try:
                check_limits(self.cfg, q)
                return True
            except Exception:
                return False
        hover = lower_tip(kin, dip, -pc.above_mm)
        if hover is None or not within(hover):       # a steep spoon far out is at the edge of reach: hover less steep
            from ..arm.scoop_style import _rz, _spoon_angle, solve_rz
            r0, z0 = _rz(kin, dip)
            sgn = 1.0 if _spoon_angle(kin, dip) >= 0 else -1.0
            for t in (10, 20, 30, 40):
                q = solve_rz(kin, dip, r0, z0 + pc.above_mm, iters=25, tilt_deg=-sgn * t)
                if q is not None and within(q):
                    hover = q
                    break
        bottom, below = None, 0.0
        for frac in (1.0, 0.75, 0.5, 0.25):         # as deep as the joints allow at this spoon angle
            q = lower_tip(kin, dip, pc.below_mm * frac)
            if q is not None and within(q):
                bottom, below = q, pc.below_mm * frac
                break
        if hover is None or bottom is None or not within(hover):
            return False
        try:
            approach = plan(self.cfg, self.arm.current(), [(list(poses[:k]) + [hover], "normal", "scoop_approach")],
                            1.0, "scoop_approach")     # the slow speed is for inside the food, not the trip there
            plan(self.cfg, hover, [([bottom], "normal", "probe")])        # limits check only
        except Exception as e:
            self.log.event("probe_skipped", detail_error=str(e)[:160])
            return False
        T = (pc.above_mm + below) / max(1.0, pc.speed_mm_s)
        hover = {j: hover[j] for j in ARM_JOINTS}
        bottom = {j: bottom[j] for j in ARM_JOINTS}
        self._probe_job = {"poses": poses, "k": k, "dip_z": K.fk(kin, dip)[2], "style": style, "texture": texture,
                           "probe_plan": Plan([Segment([hover, bottom], "normal", T, "probe", linear=True)], "probe",
                                              probe_deg=pc.threshold_deg)}
        if self.food_surface is not None and self.bite and self.bite.bowl:
            amount = (self.plate.bowls.get(self.bite.bowl) or {}).get("amount") if self.plate else None
            soft = texture in ("soft", "liquid")
            pred = 8.0 if soft else self.food_surface.predict(self.bite.bowl, amount)   # soft: it'll meet the bottom
            self._probe_job["predicted_mm"] = pred
            self.log.event("surface_predict", detail_bowl=self.bite.bowl, predicted_mm=round(pred, 1),
                           what="bowl bottom" if soft else "food surface")
        self.motion_step = 0
        self.motion = self.arm.run(approach)
        return True

    def _probe_next(self, now: float) -> bool:
        """Advance a probe scoop. True = handled (still scooping); False = finished (caller moves on)."""
        job = getattr(self, "_probe_job", None)
        if not job:
            return False
        if self.motion_step == 0:                              # above the bowl: feel downward
            self.motion_step = 1
            self.motion = self.arm.run(job["probe_plan"])
            return True
        if self.motion_step == 1:                              # touched (or reached the limit): scoop from there
            from ..arm.scoop_style import lower_tip
            kin, pc = self.cfg.kinematics, self.cfg.plate.probe
            touched = self.motion.touched if self.motion else None
            rest = [dict(p) for p in job["poses"][job["k"]:]]
            if touched and self.food_surface is not None and self.bite and self.bite.bowl:
                tzz = K.fk(kin, touched)[2]
                if job["texture"] in ("soft", "liquid"):
                    self.food_surface.history.setdefault(self.bite.bowl, []).append(
                        {"predicted_mm": job.get("predicted_mm"), "felt_mm": round(tzz, 1),
                         "error_mm": round(tzz - (job.get("predicted_mm") or tzz), 1), "what": "bottom"})
                else:
                    rec = self.food_surface.felt(self.bite.bowl, tzz)
                    self.log.event("surface_update", detail_bowl=self.bite.bowl, **rec)
            soft_food = job["texture"] in ("soft", "liquid")
            if touched and soft_food and K.fk(kin, touched)[2] > job["dip_z"] - pc.backoff_mm:
                # Resistance above the taught dip: the bowl bottom can't be there (the taught scoop went lower), so
                # it is thick food or a false stall. Scoop at the taught depth rather than shallower.
                self.log.event("scoop_probe", detail_touch_mm=round(K.fk(kin, touched)[2] - job["dip_z"], 1),
                               note="touch above the taught dip ignored: scooping at the taught depth")
                touched = None
                job["ignored_high_touch"] = True
            if touched:
                tz = K.fk(kin, touched)[2]
                dig = job["style"].depth_mm if (self.bite and self.bite.attack in ("stab_and_saw", "dig", "scrape")) \
                    else 0.0                                   # skim / ladle stay near the surface
                target = tz + pc.backoff_mm - dig
                dz = job["dip_z"] - target                     # + = lower than taught
                band = [i for i, p in enumerate(rest[:-1]) if K.fk(kin, p)[2] <= job["dip_z"] + 15]
                for i in band:
                    q = lower_tip(kin, rest[i], dz)
                    if q is not None:
                        rest[i] = q
                if self.bite:
                    self.bite.probe_touch_mm = round(tz - job["dip_z"], 1)
                self.log.event("scoop_probe", detail_touch_mm=round(tz - job["dip_z"], 1), scoop_at_mm=round(-dz, 1),
                               texture=job["texture"])
            elif not job.get("ignored_high_touch"):
                self.log.event("scoop_probe", detail_touch_mm=None, note="no resistance within the probe range")
            try:
                # slow (the strategy's speed) while the spoon is in the food; normal speed once it is lifting out
                last_in = max([i for i, q in enumerate(rest[:-1]) if K.fk(kin, q)[2] <= job["dip_z"] + 15] or [0])
                p = plan(self.cfg, self.arm.current(), [(rest[:last_in + 1], "normal", "scoop_rest")],
                         job["style"].speed, f"scoop({job['texture']})")
                if last_in + 1 < len(rest):
                    out = plan(self.cfg, rest[last_in], [(rest[last_in + 1:], "normal", "scoop_out")], 1.0)
                    p.segments.extend(out.segments)
            except Exception as e:
                self.log.event("probe_rest_fallback", detail_error=str(e)[:160])
                p = plan(self.cfg, self.arm.current(), [(job["poses"][job["k"]:], "normal", "scoop_rest")],
                         job["style"].speed, "scoop")
            p.contact_extra_deg = job["style"].contact_extra_deg
            self.motion_step = 2
            self.motion = self.arm.run(p)
            return True
        self._probe_job = None
        return False

    def _in_scooping(self, now: float) -> None:
        if self._look:
            self._in_look(now)
            return
        r = self._motion_result()
        if r == "ok" and self._probe_next(now):
            return
        if r and r != "ok":
            self._probe_job = None
        if r == "ok":
            self._go(S.LIFTING, "scooped")
        elif r:
            self._handle_motion_failure(r, now)

    def _enter_lifting(self) -> None:
        lift = "lift"
        if self.bite and self.bite.bowl and self.lib.has(f"lift_{self.bite.bowl}"):
            lift = f"lift_{self.bite.bowl}"            # Quick setup builds one lift per bowl
        self._run_paths([(lift, "normal"), ("pre_delivery", "normal")], name="lift")

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
            if b.spoon is None and b.spoon_pending_since is None and self.plate is not None \
                    and self.plate.can_check_spoon:
                b.spoon_pending_since = time.monotonic()
                threading.Thread(target=self._check_spoon, args=(b,), name="spoon-check", daemon=True).start()
            if not b.ready_spoken:
                b.ready_spoken = self.speaker.say("ready") or True

    def _check_spoon(self, b) -> None:
        r = self.plate.check_spoon()
        b.spoon = r if r is not None else {"error": True}
        b.spoon_pending_since = None
        self.log.event("spoon_check", **{("detail_" + k if i == 0 else k): v for i, (k, v) in enumerate(b.spoon.items())})

    def _in_pre_delivery_hold(self, now: float) -> None:
        r = self._motion_result()
        if r and r != "ok":
            self._handle_motion_failure(r, now)
            return
        if self.motion and r is None:
            return  # still backing off
        b = self.bite
        sc = self.cfg.plate.spoon_check
        if b and b.spoon_pending_since is not None and now - b.spoon_pending_since < sc.wait_s:
            return                                   # looking at the spoon (a second or two)
        if b and b.spoon and b.spoon.get("spoon_visible") and b.spoon.get("food_on_spoon") is False:
            if b.rescoops < sc.max_rescoops:
                b.rescoops += 1
                b.spoon, b.ready_spoken = None, False
                self.speaker.say_text("Missed. Trying again")
                self._go(S.SCOOPING, f"spoon empty: re-scoop {b.rescoops}")
                return
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
            S.IDLE: "Press GO to begin" + (" (calibration mode: nobody in the chair)" if self.calibration_mode else ""),
            S.LISTENING: "Ask for a food, or press GO", S.SCOOPING: "Scooping", S.LIFTING: "Lifting the spoon",
            S.PRE_DELIVERY_HOLD: (f"Holding: {reasons.get(v.reason, 'ready')}" if not v.ok
                                  else "Ready: press the button to bring the spoon" if not self._go_fresh(time.monotonic())
                                  else "Coming to you"),
            S.DELIVERING: "Bringing the spoon to you" + (f" ({self.arm.servo_error_mm() / 10:.0f} cm to go)"
                                                         if self.arm.servo_error_mm() > 5 else ""), S.MOUTH_HOLD: "Holding still at your mouth",
            S.RETREAT: "Moving the spoon away", S.PAUSED: "Paused. Press Resume to continue",
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
            "bowl": self.bite.bowl if self.bite else None, "last_bowl": self.last_bowl,
            "texture": self.bite.texture if self.bite else None,
            "queue": list(self.bite_queue),
            "spoon_to_mouth_mm": round(self.arm.servo_error_mm()) if self.state == S.DELIVERING else None,
            "spoon": self.bite.spoon if self.bite else None,
            "probe_touch_mm": self.bite.probe_touch_mm if self.bite else None,
            "rescoops": self.bite.rescoops if self.bite else 0,
            "go_needed": self.state == S.PRE_DELIVERY_HOLD and not self._go_fresh(time.monotonic()),
            "calibration_mode": self.calibration_mode, "end_after_bite": self.end_after_bite,
            "offset_mm": round(self.offsets.offset_mm, 1), "depth_corr_mm": round(self.offsets.depth_mm, 1),
            "tick_ms": round(self.tick_ms, 2),
            "session": {"active": self.session.active, "id": self.session.id, "bites": self.session.bites,
                        "taken": self.session.taken, "pauses": self.session.pauses},
        }
