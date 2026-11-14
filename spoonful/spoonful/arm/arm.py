"""Arm: the single owner of hardware I/O. All motion is streamed from one thread at stream_hz.

Every setpoint passes, in order (R2):
  1. soft joint limits      (clamp for live servoing; taught plans were validated at load and raise instead)
  2. speed limit            (per-tick step cap; near_face cap whenever the setpoint is in the face zone)
  3. FSM permission         (permission callback: is motion allowed in this state at all?)
  4. gate permit            (same callback: inside the face zone, approaching the mouth needs a permit)
  5. watchdog liveness      (no FSM heartbeat within heartbeat_timeout_s -> hold)
and only then reaches the backend. The asyncio side never blocks on HID: it calls the non-blocking methods
below, which just post intent for the thread.
"""
from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import ARM_JOINTS, JOINTS, Config
from ..log import Log
from . import kinematics as K
from .backend import ArmBackend, BackendError
from .contact import Contact, ContactMonitor
from .trajectory import Plan, Pose, check_limits, joint_dist, plan_goto


class MotionDenied(RuntimeError):
    pass


@dataclass
class JointState:
    ts: float = 0.0
    deg: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    ok: bool = False


class Motion:
    def __init__(self, name: str):
        self.name = name
        self.done = threading.Event()
        self.error: Optional[BaseException] = None
        self.started = time.time()

    def finish(self, error: Optional[BaseException] = None) -> None:
        if not self.done.is_set():
            self.error = error
            self.done.set()

    @property
    def ok(self) -> bool:
        return self.done.is_set() and self.error is None

    def wait(self, timeout: Optional[float] = None) -> "Motion":
        self.done.wait(timeout)
        return self


PermissionFn = Callable[[Pose, Pose, str], Optional[str]]  # (current, next, motion_name) -> deny reason or None


class Arm:
    def __init__(self, cfg: Config, backend: ArmBackend, log: Optional[Log] = None):
        self.cfg = cfg
        self.backend = backend
        self.log = log
        self.ids = {j: cfg.arm.servo_map[j] for j in JOINTS}
        self.permission: Optional[PermissionFn] = None
        self.on_fault: Optional[Callable[[str, dict], None]] = None

        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._hist = {j: deque(maxlen=cfg.arm.read_median_window) for j in JOINTS}
        self.state = JointState()
        self.setpoint: Pose = {}
        self.torque_on = False
        self.mode = "idle"            # idle | traj | servo | hold
        self._plan: Optional[Plan] = None
        self._plan_t0 = 0.0
        self._motion: Optional[Motion] = None
        self._servo_target: Optional[Pose] = None
        self._servo_speed = "near_face"
        self._servo_v = 0.0
        self._servo_factor = 1.0
        self._last_heartbeat = time.monotonic()
        self._heartbeat_required = False
        self._last_read_ok = 0.0
        self._last_servo_log = 0.0
        self._last_read = 0.0
        self.contact = ContactMonitor(cfg)
        self.last_contact: Optional[Contact] = None
        self._hid_errors = 0
        self._torque_off_at: Optional[float] = None
        self.last_deny: str = ""
        self.voltage: Optional[float] = None
        self._last_voltage_read = 0.0

    # ------------------------------------------------------------------ conversions
    def to_deg(self, j: str, units: float) -> float:
        return (units - self.cfg.arm.center_units) / self.cfg.arm.units_per_deg

    def to_units(self, j: str, deg: float) -> int:
        return int(round(self.cfg.arm.center_units + deg * self.cfg.arm.units_per_deg))

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.backend.connect()
        for _ in range(self.cfg.arm.read_median_window):
            self._read()
        if not self.state.ok:
            raise BackendError("no position reads from the arm at startup")
        self.setpoint = {j: self.state.deg[j] for j in ARM_JOINTS}
        self.voltage = self.backend.voltage()
        self._thread = threading.Thread(target=self._run, name="arm-io", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.backend.close()

    def require_heartbeat(self, on: bool = True) -> None:
        self._heartbeat_required = on
        self._last_heartbeat = time.monotonic()

    def heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    # ------------------------------------------------------------------ public, non-blocking
    def read(self) -> JointState:
        with self._lock:
            return JointState(self.state.ts, dict(self.state.deg), dict(self.state.units), self.state.ok)

    def current(self) -> Pose:
        """Where motion planning should start: the setpoint when holding torque, the measured pose otherwise."""
        with self._lock:
            if self.torque_on and self.setpoint:
                return dict(self.setpoint)
            return {j: self.state.deg[j] for j in ARM_JOINTS}

    def torque(self, on: bool, joints: Optional[list[str]] = None) -> None:
        joints = joints or ARM_JOINTS
        with self._lock:
            self._cancel_motion(MotionDenied("torque change"))
            if on:
                self.setpoint = {j: self.state.deg[j] for j in ARM_JOINTS}
                self._send({j: self.setpoint[j] for j in joints}, 250)
                self.torque_on = True
                self.mode = "hold"
            else:
                self.backend.unload([self.ids[j] for j in joints])
                self.torque_on = False
                self.mode = "idle"
            self._torque_off_at = None
        if self.log:
            self.log.event("torque", detail_joints=joints, on=on)

    def run(self, plan: Plan) -> Motion:
        m = Motion(plan.name or "plan")
        with self._lock:
            if not self.torque_on:
                self.torque(True)
            self._cancel_motion(MotionDenied("superseded"))
            self._plan, self._plan_t0, self._motion = plan, time.monotonic(), m
            self.mode = "traj"
            self.contact.reset()
        return m

    def goto(self, pose: Pose, speed: str = "normal", speed_factor: float = 1.0, name: str = "goto") -> Motion:
        p = plan_goto(self.cfg, self.current(), pose, speed, speed_factor, name)
        return self.run(p)

    def servo_toward(self, target: Pose, speed: str = "near_face", speed_factor: float = 1.0) -> None:
        """Continuous re-targeting (DELIVERING). The thread walks the setpoint toward target at the speed cap."""
        t = {j: float(target.get(j, self.setpoint.get(j, 0.0))) for j in ARM_JOINTS}
        with self._lock:
            if not self.torque_on:
                self.torque(True)
            if self.mode == "traj":
                self._cancel_motion(MotionDenied("servo override"))
            if self.mode != "servo":
                self._servo_v = 0.0
                self.contact.reset()
            self._servo_target, self._servo_speed, self.mode = t, speed, "servo"
            self._servo_factor = max(0.1, min(1.0, speed_factor))

    def servo_error_mm(self) -> float:
        with self._lock:
            if self.mode != "servo" or not self._servo_target:
                return 0.0
            return K.tip_distance(self.cfg.kinematics, self.setpoint, self._servo_target)

    def hold(self, reason: str = "") -> None:
        with self._lock:
            self._cancel_motion(MotionDenied(f"hold {reason}".strip()))
            if self.torque_on:
                self.mode = "hold"
            self._servo_target = None

    def stop(self) -> None:
        """Hold the measured position immediately, torque off after stop_torque_off_after_s."""
        with self._lock:
            self._cancel_motion(MotionDenied("stop"))
            self._servo_target = None
            if self.torque_on and self.state.ok:
                self.setpoint = {j: self.state.deg[j] for j in ARM_JOINTS}
                try:
                    self._send(self.setpoint, 100)
                except BackendError:
                    pass
                self.mode = "hold"
            self._torque_off_at = time.monotonic() + self.cfg.arm.stop_torque_off_after_s
        if self.log:
            self.log.event("arm_stop")

    def park(self) -> Motion:
        return self.goto(self.cfg.arm.park_pose, "near_face", name="park")

    def gripper(self, deg: float, duration_ms: int = 800) -> None:
        check_limits(self.cfg, {"gripper": deg}, "gripper ")
        with self._lock:
            self.backend.move([(self.ids["gripper"], self.to_units("gripper", deg))], duration_ms)

    @property
    def busy(self) -> bool:
        return self.mode in ("traj", "servo")

    def status(self) -> dict:
        with self._lock:
            tip = K.fk(self.cfg.kinematics, self.setpoint or self.state.deg) if (self.setpoint or self.state.ok) else None
            return {
                "backend": self.backend.name, "mode": self.mode, "torque": self.torque_on,
                "deg": {j: round(v, 1) for j, v in self.state.deg.items()},
                "units": dict(self.state.units),
                "setpoint": {j: round(v, 1) for j, v in self.setpoint.items()},
                "error": {j: round(self.setpoint[j] - self.state.deg.get(j, 0), 1) for j in self.setpoint},
                "read_age_ms": round((time.monotonic() - self._last_read_ok) * 1000) if self._last_read_ok else None,
                "voltage": self.voltage, "tip_mm": [round(v) for v in tip] if tip else None,
                "in_face_zone": K.in_face_zone(self.cfg.kinematics, self.setpoint) if self.setpoint else False,
                "motion": self._motion.name if self._motion and not self._motion.done.is_set() else None,
                "last_deny": self.last_deny,
            }

    def read_age_s(self) -> float:
        return time.monotonic() - self._last_read_ok if self._last_read_ok else 1e9

    # ------------------------------------------------------------------ thread
    def _cancel_motion(self, err: BaseException) -> None:
        if self._motion and not self._motion.done.is_set():
            self._motion.finish(err)
        self._plan, self._motion = None, None
        if self.mode == "traj":
            self.mode = "hold" if self.torque_on else "idle"

    def _read(self) -> None:
        try:
            raw = self.backend.read_all(list(self.ids.values()))
        except BackendError:
            raw = {i: None for i in self.ids.values()}
        now = time.monotonic()
        ok = True
        for j, sid in self.ids.items():
            u = raw.get(sid)
            if u is None:
                ok = False
                continue
            self._hist[j].append(u)
        if all(self._hist[j] for j in JOINTS):
            units = {j: int(statistics.median(self._hist[j])) for j in JOINTS}
            with self._lock:
                self.state = JointState(time.time(), {j: self.to_deg(j, u) for j, u in units.items()}, units, True)
            if ok:
                self._last_read_ok = now

    def _send(self, pose: Pose, duration_ms: int) -> None:
        pairs = [(self.ids[j], self.to_units(j, v)) for j, v in pose.items()]
        self.backend.move(pairs, max(20, int(duration_ms)))

    def _guard(self, cur: Pose, nxt: Pose, dt: float, name: str) -> tuple[Pose, Optional[str]]:
        cfg = self.cfg
        # 1. soft limits (clamp)
        out = {}
        for j in ARM_JOINTS:
            lo, hi = cfg.arm.soft_limits_deg[j]
            out[j] = min(hi, max(lo, nxt[j]))
        # 2. speed cap per tick (15% headroom over the planner)
        near = K.in_face_zone(cfg.kinematics, out) or K.in_face_zone(cfg.kinematics, cur)
        vmax = cfg.arm.max_speed_deg_s.near_face if near else cfg.arm.max_speed_deg_s.default
        step = joint_dist(cur, out)
        cap = vmax * dt * 1.15
        if step > cap > 0:
            out = {j: cur[j] + (out[j] - cur[j]) * cap / step for j in ARM_JOINTS}
        # 3 + 4. FSM permission and gate permit (a slow park is always allowed: it only ever moves away)
        if self.permission is not None and not name.startswith("park"):
            reason = self.permission(cur, out, name)
            if reason:
                return cur, reason
        # 5. watchdog liveness
        if self._heartbeat_required and not name.startswith("park") and time.monotonic() - self._last_heartbeat > cfg.arm.heartbeat_timeout_s:
            return cur, "no_heartbeat"
        return out, None

    def _run(self) -> None:
        cfg = self.cfg
        dt = 1.0 / cfg.arm.stream_hz
        read_dt = 1.0 / cfg.arm.read_hz
        servo_log_dt = 1.0 / cfg.log.servo_log_hz
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now - self._last_read >= read_dt:
                self._last_read = now
                self._read()
            if now - self._last_voltage_read > 5.0:
                self._last_voltage_read = now
                try:
                    self.voltage = self.backend.voltage()
                except BackendError:
                    pass
            try:
                self._tick(now, dt)
                self._hid_errors = 0
            except BackendError as e:
                self._hid_errors += 1
                if self._hid_errors == 3:
                    with self._lock:
                        self._cancel_motion(e)
                        self.mode = "hold"
                    if self.log:
                        self.log.fault("hid", error=str(e))
                    if self.on_fault:
                        self.on_fault("hid", {"error": str(e)})
            if self.log and self.torque_on and self.mode in ("traj", "servo") and now - self._last_servo_log >= servo_log_dt:
                self._last_servo_log = now
                ts = time.time()
                for j in ARM_JOINTS:
                    self.log.servo(ts, j, round(self.setpoint.get(j, 0), 2), round(self.state.deg.get(j, 0), 2))
            next_t += dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()

    def _tick(self, now: float, dt: float) -> None:
        with self._lock:
            if self._torque_off_at and now >= self._torque_off_at:
                self._torque_off_at = None
                self.backend.unload([self.ids[j] for j in ARM_JOINTS])
                self.torque_on = False
                self.mode = "idle"
            if not self.torque_on:
                self.setpoint = {j: self.state.deg[j] for j in ARM_JOINTS} if self.state.ok else self.setpoint
                return
            cur = dict(self.setpoint)
            name = ""
            if self.mode == "traj" and self._plan:
                t = now - self._plan_t0
                desired = self._plan.sample(t)
                name = self._plan.name
                finished = t >= self._plan.duration
            elif self.mode == "servo" and self._servo_target:
                desired = self._servo_step(cur, dt)
                name = "servo"
                finished = False
            else:
                return
            nxt, deny = self._guard(cur, desired, dt, name)
            if deny:
                self.last_deny = deny
                if self.log:
                    self.log.event("motion_denied", detail_reason=deny, motion=name)
                self._cancel_motion(MotionDenied(deny))
                self._servo_target = None
                self.mode = "hold"
                return
            self.setpoint = nxt
            speed = joint_dist(cur, nxt) / dt
            self._send(nxt, dt * 1000 * 1.5)
            c = self.contact.check(now, self.setpoint, self.state.deg, speed)
            if c:
                if self.log:
                    self.log.fault("contact", joint=c.joint, err_deg=round(c.err_deg, 1), face_zone=c.in_face_zone)
                self.setpoint = {j: self.state.deg[j] for j in ARM_JOINTS}
                self._send(self.setpoint, 100)
                self._cancel_motion(c)
                self._servo_target = None
                self.mode = "hold"
                self.contact.reset()
                self.last_contact = c
                return
            settled = joint_dist(self.state.deg, nxt) <= self.cfg.arm.contact_error_deg
            if finished and joint_dist(nxt, self._plan.end) < 0.05 and settled:
                m = self._motion
                self._plan, self._motion = None, None
                self.mode = "hold"
                if m:
                    m.finish()

    def _servo_step(self, cur: Pose, dt: float) -> Pose:
        a = self.cfg.arm
        vmax = a.max_speed_deg_s.near_face if self._servo_speed == "near_face" else a.max_speed_deg_s.default
        vmax *= self._servo_factor
        tgt = self._servo_target
        d = joint_dist(cur, tgt)
        if d < 1e-3:
            self._servo_v = 0.0
            return dict(tgt)
        # accelerate up to vmax, decelerate to arrive at rest: v <= sqrt(2 a d)
        v = min(vmax, self._servo_v + a.max_accel_deg_s2 * dt, math.sqrt(2 * a.max_accel_deg_s2 * d))
        self._servo_v = v
        s = min(1.0, v * dt / d)
        return {j: cur[j] + (tgt[j] - cur[j]) * s for j in ARM_JOINTS}
