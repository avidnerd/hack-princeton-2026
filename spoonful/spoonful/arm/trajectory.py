"""Trajectories: file format, validation, min-jerk time scaling, blending, per-segment speed caps.

A *path* is an ordered list of joint poses (deg, ARM_JOINTS) with one speed class (`normal` | `near_face`).
A *Plan* is a sequence of timed segments starting at the arm's current pose; `sample(t)` gives the setpoint.
Each segment is time-scaled with a single min-jerk profile along its arc length, so it starts and ends at rest
and never exceeds the speed / accel caps:
  peak velocity = 1.875 L / T          ->  T >= 1.875 L / v_max
  peak accel    = 5.7735 L / T^2       ->  T >= sqrt(5.7735 L / a_max)
"""
from __future__ import annotations

import bisect
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterable, Mapping, Optional

from ..config import ARM_JOINTS, Config

Pose = dict[str, float]
SPEED_CLASSES = ("normal", "near_face")


class LimitViolation(ValueError):
    pass


def min_jerk(tau: float) -> float:
    tau = max(0.0, min(1.0, tau))
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


def joint_dist(a: Mapping[str, float], b: Mapping[str, float], joints: Iterable[str] = ARM_JOINTS) -> float:
    """Max-norm in degrees: what the slowest joint has to travel."""
    return max((abs(a.get(j, 0.0) - b.get(j, 0.0)) for j in joints), default=0.0)


def lerp_pose(a: Mapping[str, float], b: Mapping[str, float], s: float) -> Pose:
    return {j: a[j] + (b[j] - a[j]) * s for j in a}


def check_limits(cfg: Config, pose: Mapping[str, float], where: str = "") -> None:
    for j, v in pose.items():
        lo, hi = cfg.arm.soft_limits_deg[j]
        if not lo <= v <= hi:
            raise LimitViolation(f"{where}{j}={v:.1f} deg outside soft limits [{lo}, {hi}]")


def segment_duration(length_deg: float, v_max: float, a_max: float, speed_factor: float = 1.0) -> float:
    if length_deg <= 1e-6:
        return 0.0
    v = v_max * speed_factor
    a = a_max * speed_factor ** 2
    return max(1.875 * length_deg / v, math.sqrt(5.7735 * length_deg / a), 0.15)


@dataclass
class Segment:
    path: list[Pose]
    speed: str
    duration: float
    label: str = ""
    linear: bool = False                # constant speed (probe descents); default is minimum-jerk
    _cum: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._cum = [0.0]
        for a, b in zip(self.path, self.path[1:]):
            self._cum.append(self._cum[-1] + joint_dist(a, b))

    @property
    def length(self) -> float:
        return self._cum[-1]

    def at(self, t: float) -> Pose:
        if self.duration <= 0 or self.length <= 0:
            return dict(self.path[-1])
        tau = min(1.0, max(0.0, t / self.duration))
        d = (tau if self.linear else min_jerk(tau)) * self.length
        k = min(len(self.path) - 2, max(0, bisect.bisect_right(self._cum, d) - 1))
        span = self._cum[k + 1] - self._cum[k]
        s = 0.0 if span <= 0 else (d - self._cum[k]) / span
        return lerp_pose(self.path[k], self.path[k + 1], s)


@dataclass
class Plan:
    segments: list[Segment]
    name: str = ""
    contact_extra_deg: float = 0.0     # extra stall allowance (firm-food scoops only; capped in contact.py)
    probe_deg: Optional[float] = None  # probe: stop (successfully) when a joint lags this much beyond its starting sag

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def locate(self, t: float) -> tuple[Segment, float]:
        for seg in self.segments:
            if t <= seg.duration:
                return seg, t
            t -= seg.duration
        return self.segments[-1], self.segments[-1].duration

    def sample(self, t: float) -> Pose:
        seg, tl = self.locate(t)
        return seg.at(tl)

    def speed_at(self, t: float) -> str:
        return self.locate(t)[0].speed

    @property
    def end(self) -> Pose:
        return dict(self.segments[-1].path[-1])

    def describe(self, hz: float = 5.0) -> str:
        """Human-readable dry-run output."""
        lines = [f"PLAN {self.name or '(unnamed)'}: {len(self.segments)} segments, {self.duration:.2f} s"]
        for s in self.segments:
            lines.append(f"  - {s.label or 'segment'} [{s.speed}] {s.length:.1f} deg in {s.duration:.2f} s "
                         f"({len(s.path)} waypoints)")
        lines.append("  t(s)  " + "  ".join(f"{j[:8]:>8}" for j in ARM_JOINTS))
        t, dt = 0.0, 1.0 / hz
        while t < self.duration + dt / 2:
            p = self.sample(min(t, self.duration))
            lines.append(f"  {t:5.2f} " + "  ".join(f"{p[j]:8.1f}" for j in ARM_JOINTS))
            t += dt
        return "\n".join(lines)


def plan(cfg: Config, start: Mapping[str, float], paths: list[tuple[list[Pose], str, str]],
         speed_factor: float = 1.0, name: str = "") -> Plan:
    """paths: [(waypoints, speed_class, label)]. A blend segment joins `start` (and each path end) to the next
    path's first waypoint when they differ by > 0.5 deg. Every waypoint is checked against soft limits."""
    a = cfg.arm
    segs: list[Segment] = []
    cur: Pose = {j: float(start[j]) for j in ARM_JOINTS}
    for wps, speed, label in paths:
        if speed not in SPEED_CLASSES:
            raise ValueError(f"unknown speed class {speed!r}")
        wps = [{j: float(w[j]) for j in ARM_JOINTS} for w in wps]
        for i, w in enumerate(wps):
            check_limits(cfg, w, f"{label}[{i}] ")
        v = a.max_speed_deg_s.near_face if speed == "near_face" else a.max_speed_deg_s.default
        gap = joint_dist(cur, wps[0])
        if gap > 0.5:
            T = max(a.blend_s, segment_duration(gap, v, a.max_accel_deg_s2, speed_factor))
            segs.append(Segment([cur, wps[0]], speed, T, f"blend->{label}"))
        if len(wps) > 1:
            seg = Segment(wps, speed, 0.0, label)
            seg.duration = segment_duration(seg.length, v, a.max_accel_deg_s2, speed_factor)
            if seg.duration > 0:
                segs.append(seg)
        cur = wps[-1]
    if not segs:
        segs.append(Segment([cur, cur], "normal", 0.0, "noop"))
    return Plan(segs, name)


def plan_goto(cfg: Config, start: Mapping[str, float], target: Mapping[str, float], speed: str = "normal",
              speed_factor: float = 1.0, name: str = "goto") -> Plan:
    full = {j: float(target.get(j, start[j])) for j in ARM_JOINTS}
    return plan(cfg, start, [([full], speed, name)], speed_factor, name)


# ---------------------------------------------------------------- trajectory files
@dataclass
class TrajFile:
    """JSON on disk. kind=path has many waypoints; kind=pose has exactly one."""
    name: str
    kind: str  # "path" | "pose"
    waypoints: list[dict]  # [{t, deg:{joint:deg}, units:{joint:units}, speed}]
    meta: dict = field(default_factory=dict)
    raw: list[dict] = field(default_factory=list)

    @property
    def poses(self) -> list[Pose]:
        return [{j: float(w["deg"][j]) for j in ARM_JOINTS} for w in self.waypoints]

    @property
    def speed(self) -> str:
        return self.waypoints[0].get("speed", "normal") if self.waypoints else "normal"

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.name}.json"
        p.write_text(json.dumps({"name": self.name, "kind": self.kind, "meta": self.meta,
                                 "waypoints": self.waypoints, "raw": self.raw}, indent=1))
        return p

    @classmethod
    def load(cls, path: Path) -> "TrajFile":
        d = json.loads(Path(path).read_text())
        return cls(d["name"], d["kind"], d["waypoints"], d.get("meta", {}), d.get("raw", []))

    def validate(self, cfg: Config) -> None:
        if not self.waypoints:
            raise LimitViolation(f"{self.name}: no waypoints")
        for i, p in enumerate(self.poses):
            check_limits(cfg, p, f"{self.name}[{i}] ")


def make_waypoints(cfg: Config, poses: list[Pose], speed: str = "normal", times: Optional[list[float]] = None) -> list[dict]:
    upd, c = cfg.arm.units_per_deg, cfg.arm.center_units
    out = []
    for k, p in enumerate(poses):
        out.append({"t": times[k] if times else k * 0.1,
                    "deg": {j: round(p[j], 2) for j in ARM_JOINTS},
                    "units": {j: int(round(c + p[j] * upd)) for j in ARM_JOINTS},
                    "speed": speed})
    return out


class Library:
    """All trajectory files in one directory, validated at load (R2: taught waypoints outside limits are errors,
    never silently clamped)."""

    PATHS = ("scoop", "scoop_small", "scoop_full", "lift", "present", "retreat")
    POSES = ("rest", "pre_delivery", "mouth_nominal")

    def __init__(self, cfg: Config, directory: Path):
        self.cfg = cfg
        self.dir = directory
        self.items: dict[str, TrajFile] = {}
        self.errors: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self.items.clear()
        self.errors.clear()
        if not self.dir.exists():
            return
        for p in sorted(self.dir.glob("*.json")):
            try:
                tf = TrajFile.load(p)
                tf.validate(self.cfg)
                self.items[tf.name] = tf
            except Exception as e:
                self.errors[p.stem] = str(e)

    def has(self, name: str) -> bool:
        return name in self.items

    def path(self, name: str) -> list[Pose]:
        return self.items[name].poses

    def pose(self, name: str) -> Pose:
        tf = self.items[name]
        return tf.poses[-1]

    def missing(self, required: Iterable[str]) -> list[str]:
        return [n for n in required if n not in self.items]

    def summary(self) -> list[dict]:
        out = [{"name": n, "kind": t.kind, "waypoints": len(t.waypoints), "speed": t.speed,
                "recorded_at": t.meta.get("recorded_at"), "ok": True} for n, t in self.items.items()]
        out += [{"name": n, "ok": False, "error": e} for n, e in self.errors.items()]
        return out


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
