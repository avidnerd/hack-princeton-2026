"""The cup and the spoon as shapes, and where the spoon can go in.

The rest of the arm code treats the spoon as one point (the tip). A spoon whose tip is at the centre of a small cup
can still have its underside or neck on the near rim, so here the spoon is a row of sample points with widths (bowl,
then neck) and the cup is a wall ring up to the rim plus a floor. `clearance` is the signed gap between the two for
one pose, `check_path` walks a move the way the arm makes it (joint-linear), and `plan_entry` searches a fixed list
of entries (angle, entry point, depth, drag) for the cheapest one whose whole path keeps the required gap.

Pure geometry: no I/O, no randomness, nothing here moves the arm. None of it has been run on the real arm; it is
only as good as the measured cup and spoon numbers it is given. Only the spoon's bowl and neck are modelled, not
the gripper or the arm's links.

What the arm can hold decides two things the planner adapts instead of failing: the hover is at the asked height
over the rim or higher, as level as the joint limits allow there; and the pivot to the entry angle happens as high
as the arm can hold that angle, which for a long spoon held steeply is lower than the hover (never less than the
margin above the rim).

Angles: the tool angle is measured from straight up (90 = level, 120 = 30 deg below horizontal), as everywhere in
arm/. `EntryPlan.entry_angle_deg` is the spoon LINE's angle below horizontal (tool angle - 90 + axis_tilt_deg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Mapping, NamedTuple, Optional

import numpy as np

from ..config import CupCfg, KinCfg, SpoonGeomCfg
from .scoop_style import _rz, solve_rz
from .trajectory import joint_dist

FKJ = ("base", "shoulder", "elbow", "wrist_pitch")      # the joints that place the spoon (roll is ignored, as in fk)
PARTS = ("tip", "bowl", "neck")
LEVEL_DEG = 10.0            # travel orientation, deg below horizontal (quick_setup's "level": a little nose-down)
ANGLE_MIN, ANGLE_MAX = 25.0, 75.0
FLOOR_TIP_MM = 2.0          # the tip is never planned lower than this above the inside bottom
GRID_MM = 2.0               # spacing of the tip positions looked at inside the cup
OFFSET_STEP_MM = 6.0
DIP_STEP_MM = 5.0
DESCENT_CHECK_MM = 8.0      # the descent is checked in vertical steps this long (and as one direct move)
ORIENT_STEP_MM = 5.0        # the pivot height is lowered in these steps until the arm can reach it
TRAVEL_DEG = (LEVEL_DEG, 20.0, 30.0, 40.0)   # travel orientations tried, most level first
HOVER_RAISE_MM, HOVER_STEP_MM = 60.0, 10.0   # the hover may be this much higher than asked, tried in these steps
INSIDE_MM = 5.0             # a dip must put the tip at least this far below the rim
DIG_START = 0.3             # a dig starts this share of its length on the arm's side of the cup centre
DIG_LEVELS = (30.0, 20.0, 10.0)    # ...and levels the spoon by this much over the push (most first)
DIG_RISES = (0.0, 8.0, 16.0)        # ...rising this much by the end of it, if that is what keeps the neck clear
MAX_BUILDS = 30             # candidates taken as far as a full path check (bounds the time of one call)
# cost: mm of spare clearance (up to CLEAR_CAP) against deg / mm away from what was asked and deg of joint travel
W_CLEAR, CLEAR_CAP, W_ANGLE, W_DRAG, W_DEPTH, W_TRAVEL, W_WRIST = 0.3, 6.0, 0.5, 0.4, 0.6, 0.002, 0.002
W_CENTRE = 0.0              # per mm the middle of the spoon bowl is from the middle of the cup. 0: tried on the real cups, centring cost 5-15 mm of depth (the handle needs the far side to clear the near rim)


# ---------------------------------------------------------------- shapes
@dataclass(frozen=True)
class Cup:
    """One container in the table frame (mm): centre of the opening, inner radius, rim height."""
    x: float
    y: float
    radius: float
    rim: float
    wall: float = 2.0
    floor: float = 3.0
    uncertainty: float = 0.0
    confidence: float = 1.0
    stamp: float = 0.0
    source: str = ""
    rim_measured: bool = True           # False: `rim` is the assumed default, not a measurement

    @classmethod
    def from_cfg(cls, c: CupCfg, default_rim_mm: Optional[float]) -> "Cup":
        """An unmeasured rim takes the default (a high guess is the safe one). With no default either the rim is
        NaN and plan_entry refuses with "unmeasured"."""
        rim = c.rim_mm if c.rim_mm is not None else (default_rim_mm if default_rim_mm else float("nan"))
        return cls(x=c.x_mm, y=c.y_mm, radius=c.radius_mm, rim=float(rim), wall=c.wall_mm, floor=c.floor_mm,
                   uncertainty=c.uncertainty_mm, source=c.source, rim_measured=c.rim_mm is not None)

    def moved(self, dx: float, dy: float, **kw) -> "Cup":
        return replace(self, x=self.x + dx, y=self.y + dy, **kw)

    def safe_radius(self, half_width_mm: float, clearance_mm: float) -> float:
        """How far from the centre the middle of a part this wide may be (0 = it does not fit at all). The cup's
        own position uncertainty comes off as well."""
        return max(0.0, self.radius - half_width_mm - clearance_mm - self.uncertainty)


@dataclass(frozen=True)
class Spoon:
    """Lengths run back from the tip along the spoon's line (mm)."""
    bowl_length: float
    bowl_width: float
    bowl_depth: float
    neck_length: float
    neck_width: float
    axis_tilt_deg: float = 0

    @classmethod
    def from_cfg(cls, s: SpoonGeomCfg) -> "Spoon":
        return cls(s.bowl_length_mm, s.bowl_width_mm, s.bowl_depth_mm, s.neck_length_mm, s.neck_width_mm,
                   s.axis_tilt_deg)


class BodyPoint(NamedTuple):
    """A horizontal slice of the spoon: centre x, y, z and half its width. The width runs level, across the arm's
    plane (direction yaw + 90 deg)."""
    x: float
    y: float
    z: float
    half_width: float
    part: str
    yaw: float = 0.0


class Clearance(NamedTuple):
    mm: float                               # signed: negative = the spoon is in the cup's wall or floor
    point: tuple[float, float, float]
    part: str


class PathCheck(NamedTuple):
    """index, s, point, part: the first place the clearance is below the asked margin (segment index and the
    fraction along it); when ok, the place of the smallest clearance instead."""
    ok: bool
    min_mm: float
    index: int
    s: float
    point: tuple[float, float, float]
    part: str


@lru_cache(maxsize=8)
def _template(spoon: Spoon):
    """Sample points in the spoon's own frame: distance back along the line, drop below the line, half width,
    part. The bowl outline is an ellipse (nothing at the tip, widest mid-bowl); each bowl slice is there twice, on
    the line (the bowl's edge) and dropped by the underside depth at full width (a box around the real curve)."""
    L, W, D = spoon.bowl_length, spoon.bowl_width, max(0.0, spoon.bowl_depth)
    n = max(2, int(math.ceil(L / 2.5)))
    t = np.linspace(0.0, 1.0, n + 1)
    wb = W * np.sqrt(t * (1 - t))
    wb = np.where(t > 0.5, np.maximum(wb, spoon.neck_width / 2), wb)       # the bowl narrows into the neck
    db = 4 * D * t * (1 - t)
    nn = int(math.ceil(spoon.neck_length / 3.0)) if spoon.neck_length > 0 else 0
    sn = L + np.linspace(0.0, spoon.neck_length, nn + 1)[1:]
    s = np.concatenate([L * t, L * t[1:-1], sn])
    d = np.concatenate([0 * t, db[1:-1], 0 * sn])
    w = np.concatenate([wb, wb[1:-1], np.full(nn, spoon.neck_width / 2)])
    part = np.array([0] + [1] * n + [1] * (n - 1) + [2] * nn)
    return s, d, w, part


def _low(spoon: Spoon, line_deg: float) -> float:
    """Height of the spoon's lowest point relative to its tip (<= 0) with the line at line_deg from straight up."""
    s, d, _, _ = _template(spoon)
    p = math.radians(line_deg)
    return float(np.min(-s * math.cos(p) - d * math.sin(p)))


def _fk(kin: KinCfg, Q: np.ndarray):
    """kinematics.fk for many poses at once. Q: (N, 4) deg in FKJ order -> tip (N, 3), signed reach, tool angle
    (rad from straight up), yaw (rad)."""
    A = np.radians((Q - np.array([kin.zero_deg[j] for j in FKJ])) * np.array([kin.sign[j] for j in FKJ]))
    yaw, p1 = A[:, 0], A[:, 1]
    p2 = p1 + A[:, 2]
    p3 = p2 + A[:, 3]
    rho = kin.upper_arm_mm * np.sin(p1) + kin.forearm_mm * np.sin(p2) + kin.tool_mm * np.sin(p3) \
        - kin.tool_offset_mm * np.cos(p3)
    z = kin.base_height_mm + kin.upper_arm_mm * np.cos(p1) + kin.forearm_mm * np.cos(p2) \
        + kin.tool_mm * np.cos(p3) + kin.tool_offset_mm * np.sin(p3)
    return np.stack([rho * np.cos(yaw), rho * np.sin(yaw), z], axis=1), rho, p3, yaw


def _body(tip: np.ndarray, line: np.ndarray, yaw: np.ndarray, spoon: Spoon):
    """Sample points (N, M) for spoons with the tip at `tip`, the line at `line` rad from straight up, in the
    vertical plane at `yaw`. The underside is the side that faces down when the spoon points away from the arm."""
    s, d, _, _ = _template(spoon)
    sp, cp = np.sin(line)[:, None], np.cos(line)[:, None]
    drho = -s * sp + d * cp
    return (tip[:, 0:1] + drho * np.cos(yaw)[:, None], tip[:, 1:2] + drho * np.sin(yaw)[:, None],
            tip[:, 2:3] - s * cp - d * sp)


def _clear(tip: np.ndarray, line: np.ndarray, yaw: np.ndarray, cup: Cup, spoon: Spoon):
    """Signed clearance of every sample point, (N, M), and the points. Each point is a level segment across the
    arm's plane, so its nearest and farthest distances from the cup's axis are exact, not centre +- half width."""
    X, Y, Z = _body(tip, line, yaw, spoon)
    w = _template(spoon)[2]
    cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    dx, dy = X - cup.x, Y - cup.y
    along, across = dx * cy + dy * sy, np.abs(dy * cy - dx * sy)
    dmax = np.hypot(along, across + w)
    dmin = np.hypot(along, np.maximum(across - w, 0.0))
    inside = np.hypot(along, across) <= cup.radius + cup.wall / 2
    gap = np.where(inside, cup.radius - dmax, dmin - cup.radius - cup.wall)     # level gap to the wall
    up = Z - cup.rim
    c = np.where(up >= 0,
                 np.where(gap >= 0, np.hypot(gap, up), up),                     # above: to the rim ring
                 np.where(gap >= 0, gap, -np.minimum(-gap, -up)))               # below: beside the wall, or in it
    return np.where(inside, np.minimum(c, Z - cup.floor), c), (X, Y, Z)


def _q(poses: list[Mapping[str, float]]) -> np.ndarray:
    return np.array([[float(q[j]) for j in FKJ] for q in poses])


def _clear_q(kin: KinCfg, Q: np.ndarray, cup: Cup, spoon: Spoon):
    tip, _, p3, yaw = _fk(kin, Q)
    return _clear(tip, p3 + math.radians(spoon.axis_tilt_deg), yaw, cup, spoon)


def spoon_body(kin: KinCfg, q: Mapping[str, float], spoon: Spoon) -> list[BodyPoint]:
    """The space the spoon takes up at pose q, as sample slices from the tip back (bowl, then neck)."""
    tip, _, p3, yaw = _fk(kin, _q([q]))
    X, Y, Z = _body(tip, p3 + math.radians(spoon.axis_tilt_deg), yaw, spoon)
    _, _, w, part = _template(spoon)
    return [BodyPoint(float(X[0, i]), float(Y[0, i]), float(Z[0, i]), float(w[i]), PARTS[part[i]], float(yaw[0]))
            for i in range(len(w))]


def spoon_center(kin: KinCfg, q: Mapping[str, float], spoon: Spoon) -> tuple[float, float, float]:
    """Centre of the spoon's bowl (half a bowl length back from the tip, on the spoon's line)."""
    tip, _, p3, yaw = _fk(kin, _q([q]))
    p, h = float(p3[0]) + math.radians(spoon.axis_tilt_deg), spoon.bowl_length / 2
    r = -h * math.sin(p)
    return (float(tip[0, 0]) + r * math.cos(yaw[0]), float(tip[0, 1]) + r * math.sin(yaw[0]),
            float(tip[0, 2]) - h * math.cos(p))


def clearance(kin: KinCfg, q: Mapping[str, float], cup: Cup, spoon: Spoon) -> Clearance:
    """Smallest signed gap between the spoon at pose q and the cup (wall ring up to the rim, and the floor), the
    spoon point it is at and which part of the spoon that is. Negative: that part is inside the wall or floor."""
    c, (X, Y, Z) = _clear_q(kin, _q([q]), cup, spoon)
    i = int(np.argmin(c[0]))
    return Clearance(float(c[0, i]), (float(X[0, i]), float(Y[0, i]), float(Z[0, i])), PARTS[_template(spoon)[3][i]])


def _measured(cup: Cup, spoon: Spoon) -> bool:
    nums = (cup.x, cup.y, cup.radius, cup.rim, cup.wall, cup.floor, cup.uncertainty, spoon.bowl_length,
            spoon.bowl_width, spoon.bowl_depth, spoon.neck_length, spoon.neck_width, spoon.axis_tilt_deg)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in nums):
        return False
    return cup.radius > 0 and 0 <= cup.floor < cup.rim and cup.wall >= 0 and cup.uncertainty >= 0 \
        and spoon.bowl_length > 0 and spoon.bowl_width > 0 and spoon.bowl_depth >= 0 and spoon.neck_length >= 0 \
        and spoon.neck_width >= 0


def fits(cup: Cup, spoon: Spoon, clearance_mm: float) -> tuple[bool, str]:
    """Can the spoon's bowl pass through the opening at all, with the clearance and the cup's position
    uncertainty on both sides? (ok, why not)."""
    if not _measured(cup, spoon):
        return False, "unmeasured"
    need = spoon.bowl_width + 2 * (clearance_mm + cup.uncertainty)
    if need <= 2 * cup.radius:
        return True, ""
    return False, (f"opening {2 * cup.radius:.0f} mm; the spoon is {spoon.bowl_width:.0f} mm wide and needs "
                   f"{need:.0f} mm with {clearance_mm + cup.uncertainty:.0f} mm each side")


def check_path(kin: KinCfg, poses: list[Mapping[str, float]], cup: Cup, spoon: Spoon, clearance_mm: float,
               step_deg: float = 1.0) -> PathCheck:
    """The move through `poses`, sampled every step_deg along the straight joint-space line between each pair
    (that is how the arm goes from one to the next): is the spoon always at least clearance_mm from the cup?"""
    P = _q(poses)
    Qs, seg, frac = [P[:1]], [0], [0.0]
    for k in range(len(P) - 1):
        n = max(1, int(math.ceil(float(np.abs(P[k + 1] - P[k]).max()) / step_deg)))
        t = np.arange(1, n + 1) / n
        Qs.append(P[k] + (P[k + 1] - P[k]) * t[:, None])
        seg += [k] * n
        frac += [float(v) for v in t]
    c, (X, Y, Z) = _clear_q(kin, np.concatenate(Qs), cup, spoon)
    worst = c.min(axis=1)
    bad = np.nonzero(worst < clearance_mm)[0]
    i = int(bad[0]) if len(bad) else int(np.argmin(worst))
    j = int(np.argmin(c[i]))
    return PathCheck(not len(bad), float(worst.min()), seg[i], frac[i],
                     (float(X[i, j]), float(Y[i, j]), float(Z[i, j])), PARTS[_template(spoon)[3][j]])


# ---------------------------------------------------------------- the entry plan
@dataclass
class EntryPlan:
    cup: Cup
    entry_angle_deg: float                  # spoon line below horizontal, going in
    entry_xy: tuple[float, float]           # where the tip goes down
    dip_z: float                            # lowest planned tip height above the table
    drag_mm: float                          # stroke toward the arm at dip_z
    hover: dict                             # tip over the entry point, travel orientation (as level as the arm can
                                            # hold there), whole spoon at least hover_above over the rim
    oriented: dict                          # same x, y, entry orientation, whole spoon still above the rim (as high
                                            # as the arm can hold that angle: hover height or lower, never less
                                            # than the margin above the rim)
    dip: dict                               # straight down from `oriented`
    stroke: list[dict]                      # in-cup waypoints after the dip (the drag)
    out: dict                               # straight up from the end of the stroke, clear of the rim
    clear: dict                             # travel orientation at hover height or above (hand-off to the lift)
    min_clearance_mm: float                 # smallest gap over hover..clear (the approach is checked separately)
    cost: float
    rejected: list[dict]                    # candidates tried and why they were dropped
    approach: list[dict] = field(default_factory=list)     # poses to run before `hover` (may be empty)
    kin: Optional[KinCfg] = field(default=None, repr=False)

    def poses(self) -> list[dict]:
        return [self.hover, self.oriented, self.dip, *self.stroke, self.out, self.clear]

    def descent(self, step_mm: float) -> list[dict]:
        """`oriented` down to `dip` in vertical steps of at most step_mm, same reach and spoon angle. The first
        pose is `oriented` and the last is `dip`."""
        r, z_top = _rz(self.kin, self.oriented)
        z_dip = _rz(self.kin, self.dip)[1]
        n = max(1, int(math.ceil((z_top - z_dip) / max(0.5, step_mm) - 1e-6)))
        out, cur = [dict(self.oriented)], self.oriented
        for k in range(1, n):
            q = solve_rz(self.kin, cur, r, z_top + (z_dip - z_top) * k / n, iters=25)
            if q is not None:
                out.append(q)
                cur = q
        return out + [dict(self.dip)]


class NoSafeEntry(Exception):
    """reason: "unmeasured" | "too_narrow" | "unreachable" | "no_clear_path"."""

    def __init__(self, reason: str, rejected: Optional[list[dict]] = None, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.rejected = rejected or []
        self.detail = detail


def _ik(kin: KinCfg, ref: Mapping[str, float], rho: float, z: float, tool_deg: float) -> list[Optional[dict]]:
    """Closed form: ref's base and roll with the tip at reach rho, height z and the tool at tool_deg. Two arm
    shapes (elbow one way, elbow the other); None where the point is out of reach."""
    p = math.radians(tool_deg)
    a, b = kin.upper_arm_mm, kin.forearm_mm
    wr = rho - kin.tool_mm * math.sin(p) + kin.tool_offset_mm * math.cos(p)
    wz = z - kin.base_height_mm - kin.tool_mm * math.cos(p) - kin.tool_offset_mm * math.sin(p)
    c = (wr * wr + wz * wz - a * a - b * b) / (2 * a * b)
    if abs(c) > 1:
        return [None, None]
    out = []
    for e in (math.acos(c), -math.acos(c)):
        p1 = math.atan2(wr, wz) - math.atan2(b * math.sin(e), a + b * math.cos(e))
        q = {j: float(v) for j, v in ref.items()}
        for j, phi in (("shoulder", p1), ("elbow", e), ("wrist_pitch", p - p1 - e)):
            deg = (math.degrees(phi) + 180.0) % 360.0 - 180.0
            q[j] = kin.zero_deg[j] + deg / kin.sign[j]
        out.append(q)
    return out


def _within(limits: Mapping[str, tuple[float, float]], q: Mapping[str, float]) -> bool:
    return all(limits[j][0] <= q[j] <= limits[j][1] for j in q if j in limits)


def _base_deg(kin: KinCfg, limits: Mapping[str, tuple[float, float]], yaw: float) -> Optional[float]:
    b = kin.zero_deg["base"] + math.degrees(yaw) / kin.sign["base"]
    lo, hi = limits.get("base", (-1e9, 1e9))
    return next((v for v in (b, b + 360.0, b - 360.0) if lo <= v <= hi), None)


class _Search:
    """One plan_entry call for one arm shape (elbow branch): poses by closed-form IK, confirmed by solve_rz and
    by forward kinematics, remembered so candidates that share a pose don't solve it twice."""

    def __init__(self, kin, limits, ref, cup, spoon, branch, margin, hover_above, start):
        self.kin, self.limits, self.ref, self.cup, self.spoon = kin, limits, ref, cup, spoon
        self.branch, self.margin, self.hover_above, self.start = branch, margin, hover_above, start
        self.rc, self.yaw = math.hypot(cup.x, cup.y), math.atan2(cup.y, cup.x)
        self.memo: dict = {}
        self.approaches: dict = {}
        self.levels: dict = {}

    def pose(self, off: float, z: float, angle: float) -> Optional[dict]:
        """Tip `off` mm beyond the cup centre along the arm's line, at height z, spoon `angle` below horizontal."""
        key = (round(off, 3), round(z, 3), round(angle, 3))
        if key not in self.memo:
            rho = self.rc + off
            q = _ik(self.kin, self.ref, rho, z, 90.0 + angle - self.spoon.axis_tilt_deg)[self.branch]
            if q is not None and _within(self.limits, q):
                q = solve_rz(self.kin, q, rho, z, iters=8)
            else:
                q = None
            if q is not None:                      # the tip must really be there (not mirrored behind the base)
                tip = _fk(self.kin, _q([q]))[0][0]
                want = (rho * math.cos(self.yaw), rho * math.sin(self.yaw), z)
                if math.dist(tip, want) > 1.0 or not _within(self.limits, q):
                    q = None
            self.memo[key] = q
        return self.memo[key]

    def level(self, off: float) -> tuple[float, float, Optional[dict]]:
        """(angle, tip height, pose) for travelling over the cup: the whole spoon at least hover_above over the rim,
        as level as the arm can hold it there (level first, since a loaded spoon leaves in this pose; then higher,
        since height only costs time). Folded right over a near cup the arm often cannot hold the spoon level."""
        if off not in self.levels:
            self.levels[off] = (LEVEL_DEG, 0.0, None)
            for ang in TRAVEL_DEG:
                z0 = self.cup.rim + self.hover_above - _low(self.spoon, 90.0 + ang)
                q = next((q for q in (self.pose(off, z0 + k * HOVER_STEP_MM, ang)
                                      for k in range(int(HOVER_RAISE_MM / HOVER_STEP_MM) + 1)) if q is not None), None)
                if q is not None:
                    self.levels[off] = (ang, _rz(self.kin, q)[1], q)
                    break
        return self.levels[off]

    def orient(self, off: float, angle: float) -> tuple[float, Optional[dict]]:
        """The entry orientation over the entry point, as high as the arm can hold it: from the full hover height
        down to the whole spoon just the margin above the rim. (A long spoon held steeply puts the wrist high; this
        arm often cannot do that at hover height.)"""
        low = _low(self.spoon, 90.0 + angle)
        z, z_min = self.cup.rim + self.hover_above - low, self.cup.rim + self.margin + 1.0 - low
        while True:
            q = self.pose(off, max(z, z_min), angle)
            # servos sag: the command that puts the real spoon here is higher. It has to be reachable too.
            if q is not None and getattr(self, "headroom", 0.0) and                     self.pose(off, max(z, z_min) + self.headroom, angle) is None:
                q = None
            if q is not None or z <= z_min:
                return max(z, z_min), q
            z -= ORIENT_STEP_MM

    def check(self, poses: list[dict]) -> PathCheck:
        return check_path(self.kin, poses, self.cup, self.spoon, self.margin)

    def approach(self, hover: dict) -> Optional[list[dict]]:
        """Poses to run before `hover` so the move from `start` keeps the margin: nothing if the direct move is
        clear; else rise straight up first, then (if still not clear) take the hover's arm shape before turning."""
        if self.start is None:
            return []
        key = tuple(hover[j] for j in FKJ)
        if key not in self.approaches:
            self.approaches[key] = self._approach(hover)
        return self.approaches[key]

    def _approach(self, hover: dict) -> Optional[list[dict]]:
        start = {j: float(v) for j, v in self.start.items()}
        _, rho, p3, _ = _fk(self.kin, _q([start]))
        z0 = _rz(self.kin, start)[1]
        tool = math.degrees(float(p3[0]))
        z_up = self.cup.rim + self.hover_above - _low(self.spoon, tool + self.spoon.axis_tilt_deg)
        rise = None
        if z_up > z0 + 1.0:
            sols = [q for q in _ik(self.kin, start, float(rho[0]), z_up, tool) if q and _within(self.limits, q)]
            rise = min(sols, key=lambda q: joint_dist(q, start, FKJ)) if sols else None

        def swing(frm: dict) -> dict:
            return {**hover, "base": frm["base"]}

        options = [[]] + ([[rise], [rise, swing(rise)]] if rise else []) + [[swing(start)]]
        # A tall cup beside the resting spoon: none of those may clear it. Rise where the spoon is, at whatever
        # spoon angle the arm can hold up there (and a little nearer the base if need be), then turn, then reach.
        for back in (0.0, 20.0, 40.0):
            for ang in TRAVEL_DEG + (50.0, 60.0):
                z_a = self.cup.rim + self.hover_above - _low(self.spoon, 90.0 + ang)
                sols = [q for q in _ik(self.kin, start, float(rho[0]) - back, z_a, 90.0 + ang - self.spoon.axis_tilt_deg)
                        if q and _within(self.limits, q)]
                if sols:
                    up = min(sols, key=lambda q: joint_dist(q, start, FKJ))
                    options += [[up], [up, {**up, "base": hover["base"]}], [up, {**up, "base": hover["base"]}, swing(up)]]
        for opt in options:
            if all(_within(self.limits, q) for q in opt) and self.check([start, *opt, hover]).ok:
                return [dict(q) for q in opt]
        return None


def _candidates(cup: Cup, spoon: Spoon, margin: float, hover_above: float, want_angle: float, want_drag: float,
                want_depth: Optional[float]):
    """Every entry on the fixed list, checked as pure geometry (no arm yet): the spoon going straight down at the
    entry point and along the drag. Returns (survivors, rejected summaries). The cup is put on the x axis here;
    only distances from its centre matter."""
    here = replace(cup, x=1000.0, y=0.0)
    angles = sorted({min(ANGLE_MAX, max(ANGLE_MIN, want_angle + d)) for d in (0, 5, -5, 10, -10, 15, -15, 20, -20, 30, -30)},
                    reverse=True)                                              # steep first
    k_far = int(cup.safe_radius(0.0, margin - cup.uncertainty) // OFFSET_STEP_MM)
    offsets = [k * OFFSET_STEP_MM for k in range(k_far, -(k_far // 2) - 1, -1)]  # toward the far wall first
    drags = list(dict.fromkeys(math.floor(max(0.0, want_drag) * f / GRID_MM) * GRID_MM for f in (1, .75, .5, .25, 0)))
    z_want = cup.floor + FLOOR_TIP_MM if want_depth is None else max(cup.floor + FLOOR_TIP_MM, cup.rim - want_depth)
    rho = np.arange(offsets[-1] - drags[0], offsets[0] + GRID_MM / 2, GRID_MM) if offsets else np.zeros(0)
    out, rejected, order = [], [], 0
    for a in angles:
        line = math.radians(90.0 + a)
        z_low = max(z_want, cup.floor + margin - _low(spoon, 90.0 + a) + 0.05)    # lowest tip the floor allows
        dips = [z for z in (z_low + DIP_STEP_MM * k for k in range(4)) if z <= cup.rim - INSIDE_MM]
        z_top = cup.rim + hover_above - _low(spoon, 90.0 + a)
        n_ok, n_bad, best = 0, 0, -1e9
        if dips and offsets:
            zg = z_low + 2.5 * np.arange(int(math.ceil((z_top - z_low) / 2.5)) + 1)
            def table(r, z):                       # smallest clearance with the tip at each (r, z)
                tip = np.stack([1000.0 + r, 0 * r, z], axis=1)
                return _clear(tip, np.full(len(r), line), np.zeros(len(r)), here, spoon)[0].min(axis=1)
            R, Zc = np.meshgrid(np.array(offsets), zg, indexing="ij")
            down = table(R.ravel(), Zc.ravel()).reshape(R.shape)
            down = np.minimum.accumulate(down[:, ::-1], axis=1)[:, ::-1]        # from the hover down to each depth
            R, Zr = np.meshgrid(rho, np.array(dips), indexing="ij")
            side = table(R.ravel(), Zr.ravel()).reshape(R.shape)
        for oi, off in enumerate(offsets):
            for d in drags:
                for zi, z in enumerate(dips):
                    order += 1
                    j1 = int(round((off - rho[0]) / GRID_MM))
                    j0 = j1 - int(round(d / GRID_MM))
                    pre = min(float(down[oi, int(round((z - z_low) / 2.5))]), float(side[j0:j1 + 1, zi].min()))
                    if pre < margin:
                        n_bad, best = n_bad + 1, max(best, pre)
                        continue
                    n_ok += 1
                    # the middle of the stroke should be near the middle of the food, not against the far wall
                    mid = off - d / 2.0 - 0.5 * spoon.bowl_length * math.cos(math.radians(a))
                    dev = W_ANGLE * abs(a - want_angle) + W_DRAG * (max(0.0, want_drag) - d) + W_DEPTH * (z - z_want)                         + W_CENTRE * abs(mid)
                    out.append((dev - W_CLEAR * min(pre - margin, CLEAR_CAP), order, a, off, z, d, pre, dev))
        if not n_ok:
            why = "clearance" if n_bad else ("too_deep_for_floor" if offsets else "no_room_for_tip")
            rejected.append({"stage": "geometry", "angle_deg": a, "why": why, "count": n_bad, "kept": 0,
                             "best_mm": round(best, 1) if n_bad else None})
        elif n_bad:
            rejected.append({"stage": "geometry", "angle_deg": a, "why": "clearance", "count": n_bad, "kept": n_ok,
                             "best_mm": round(best, 1)})
    return sorted(out, key=lambda c: c[:2]), rejected


def plan_entry(kin: KinCfg, limits: Mapping[str, tuple[float, float]], seed: Mapping[str, float], cup: Cup,
               spoon: Spoon, *, clearance_mm: float, hover_above_rim_mm: float, want_angle_deg: float,
               want_drag_mm: float, want_depth_mm: Optional[float] = None,
               start: Optional[Mapping[str, float]] = None, dig_mm: float = 0.0,
               headroom_mm: float = 0.0) -> EntryPlan:
    """The cheapest entry into this cup whose whole path (hover, pivot, descent, drag, way out, and the approach
    from `start`) keeps clearance_mm + cup.uncertainty between spoon and cup and stays inside the soft limits.

    limits: joint -> (lo, hi). seed: a pose that gives the roll and picks the arm shape (elbow branch) to prefer.
    want_depth_mm: how far below the rim the tip should go (None = as deep as the floor margin allows).
    dig_mm > 0: a digging stroke instead of the drag (thick food): the tip goes in on the arm's side just under the
    rim and pushes forward and down along the spoon's own line, up to dig_mm across, ending deep on the far side
    (EntryPlan.dip_z is where that push starts; the last stroke pose is its deepest point). Checked like the rest.
    Raises NoSafeEntry; never returns a plan that failed a check. Same inputs give the same plan."""
    if not _measured(cup, spoon):
        raise NoSafeEntry("unmeasured", detail="cup or spoon dimensions missing")
    ok, why = fits(cup, spoon, clearance_mm)
    if not ok:
        raise NoSafeEntry("too_narrow", [{"stage": "fit", "why": why}], why)
    margin = clearance_mm + cup.uncertainty
    want = min(ANGLE_MAX, max(ANGLE_MIN, float(want_angle_deg)))
    dig = max(0.0, float(dig_mm))
    cands, rejected = _candidates(cup, spoon, margin, hover_above_rim_mm, want, 0.0 if dig else float(want_drag_mm),
                                  want_depth_mm)
    if not cands:
        raise NoSafeEntry("no_clear_path", rejected, "no entry keeps the clearance inside this cup")
    base = _base_deg(kin, limits, math.atan2(cup.y, cup.x))
    if base is None:
        raise NoSafeEntry("unreachable", rejected, "the base cannot turn that far")
    ref = {**{j: float(v) for j, v in seed.items()}, "base": base}
    # the arm shape nearest the seed first; the other one only if the first gives no plan at all
    probe = _ik(kin, ref, math.hypot(cup.x, cup.y), cup.rim + hover_above_rim_mm + 10.0, 90.0 + LEVEL_DEG)
    branches = sorted((0, 1), key=lambda b: (probe[b] is None or not _within(limits, probe[b]),
                                             joint_dist(probe[b], seed, FKJ[1:]) if probe[b] else 0.0, b))
    blocked = False
    for branch in branches:
        S = _Search(kin, limits, ref, cup, spoon, branch, margin, hover_above_rim_mm, start)
        S.dig, S.headroom = dig, max(0.0, float(headroom_mm))
        best, builds = None, 0
        for lb, order, a, off, z, d, pre, dev in cands:
            if (best is not None and lb >= best[0]) or builds >= MAX_BUILDS:
                break
            got = _build(S, a, off, z, d)
            if isinstance(got, str):
                blocked = blocked or got != "unreachable"
                builds += got != "unreachable"
                if len(rejected) < 60:
                    rejected.append({"stage": "arm", "angle_deg": a, "offset_mm": off, "dip_z": round(z, 1),
                                     "drag_mm": d, "why": got, "branch": branch})
                continue
            builds += 1
            plan, travel, wrist = got
            spare = min(plan.min_clearance_mm, pre) - margin
            cost = dev - W_CLEAR * min(spare, CLEAR_CAP) + W_TRAVEL * travel + W_WRIST * wrist
            if best is None or (cost, order) < best[:2]:
                best = (cost, order, plan)
        if best is not None:
            plan = best[2]
            plan.cost, plan.rejected = best[0], rejected
            return plan
    if blocked:
        raise NoSafeEntry("no_clear_path", rejected, "every reachable entry comes too close to the cup")
    raise NoSafeEntry("unreachable", rejected, "the arm cannot reach over this cup within its limits")


def _build(S: _Search, a: float, off: float, z: float, d: float):
    """One candidate as arm poses, fully checked. (EntryPlan, joint travel, wrist travel) or the reason it fails:
    "unreachable" (IK / soft limits), "clearance" (way in), "retract" (way out), "approach"."""
    cup, spoon = S.cup, S.spoon
    a_last, dig_stroke = a, None
    if getattr(S, "dig", 0.0) > 0:
        # Dig: (off, z) is where the tip ENDS, deep on the far side. It gets there along the spoon's own line: in
        # on the arm's side just under the rim, then forward and down. Sliding along its own line keeps the handle
        # at one height over the near rim, which is what lets a long tilted spoon dig in a tall tub at all.
        t, found, why = math.tan(math.radians(a)), None, "unreachable"
        for f in (1.0, 0.75, 0.5, 0.3):
            D = S.dig * f
            off_s, z_s = off - D, z + D * t
            if z_s > cup.rim - INSIDE_MM:
                continue
            d0 = S.pose(off_s, z_s, a)
            cand = [S.pose(off_s + D * k / 3, z_s - D * t * k / 3, a) for k in (1, 2, 3)]
            if d0 is None or None in cand:
                continue
            if not S.check([d0, *cand]).ok:
                why = "clearance"
                continue
            found = (off_s, z_s, D, cand)
            break
        if found is None:
            return why
        off, z, D, dig_stroke = found
        d = -D
    # After a dig the spoon is far out, where the arm sags most and cannot lift much higher: leave over the middle.
    clear_off = 0.0 if dig_stroke is not None and off - d > 0 else off - d
    hover, (a_clear, z_clear, clear) = S.level(off)[2], S.level(clear_off)
    if hover is None or clear is None:
        return "unreachable"
    z_or, oriented = S.orient(off, a)
    dip = S.pose(off, z, a) if oriented is not None else None
    if dig_stroke is not None:
        stroke = dig_stroke
    else:
        stroke = [S.pose(off - d, z, a)] if d > 0 and dip is not None else []
    if dip is None or None in stroke:
        return "unreachable"
    n = max(1, int(math.ceil((z_or - z) / DESCENT_CHECK_MM)))
    steps = [S.pose(off, z_or + (z - z_or) * k / n, a) for k in range(1, n)]
    if any(q is None for q in steps):
        return "unreachable"
    way_in = S.check([hover, oriented, *steps, dip, *stroke])
    direct = S.check([oriented, dip])                 # the same descent as one move (it bows a little)
    if not (way_in.ok and direct.ok):
        return "clearance"
    # way out: straight up, levelling as much as stays clear (level keeps the food on; steep is what fits)
    last, out, way_out, reach = (stroke or [dip])[-1], None, None, False
    for f in (1.0, 0.75, 0.5, 0.25, 0.0):
        ao = a_last + f * (a_clear - a_last)
        q = S.pose(off - d, min(z_clear, cup.rim + S.margin + 1.0 - _low(spoon, 90.0 + ao)), ao)
        if q is None:
            continue
        reach = True
        chk = S.check([last, q, clear])
        if chk.ok:
            out, way_out = q, chk
            break
    if out is None:
        return "retract" if reach else "unreachable"
    approach = S.approach(hover)
    if approach is None:
        return "approach"
    rho = S.rc + off
    plan = EntryPlan(cup=cup, entry_angle_deg=a, entry_xy=(rho * math.cos(S.yaw), rho * math.sin(S.yaw)),
                     dip_z=z, drag_mm=d, hover=dict(hover), oriented=dict(oriented), dip=dict(dip),
                     stroke=[dict(q) for q in stroke], out=dict(out), clear=dict(clear),
                     min_clearance_mm=min(way_in.min_mm, direct.min_mm, way_out.min_mm), cost=0.0, rejected=[],
                     approach=approach, kin=S.kin)
    seq = ([dict(S.start)] if S.start is not None else []) + approach + plan.poses()
    travel = sum(joint_dist(p, q, FKJ) for p, q in zip(seq, seq[1:]))
    wrist = sum(abs(p["wrist_pitch"] - q["wrist_pitch"]) for p, q in zip(seq, seq[1:]))
    return plan, travel, wrist
