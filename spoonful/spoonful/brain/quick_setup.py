"""Quick setup: adapt to a new table in about two minutes, from three touches.

At a new venue nothing is where it was: bowls, the diner's seat, the camera. Instead of recording every path by hand,
the operator (torque off) touches the spoon into each bowl ("Spoon is in bowl N") and to the diner's lips ("Spoon is
at my mouth"). From those poses this module builds every move the bite needs:

  scoop_bowlN   from above the bowl: down to the touched dip, drag 2.5 cm toward the arm, tilt the spoon near level,
                lift 6 cm. Spoon angle at the dip = the angle the operator used.
  pre_delivery  10 cm straight back from the lips toward the arm (8 cm if 10 is out of reach), same height.
  lift          from the end of the scoop up to the waiting height, then across to pre_delivery.
  face zone     a cylinder around the mouth (radius 6.5 cm, from 5 cm below the lips to 20 cm above), so it works
                wherever the diner sits. pre_delivery is outside it by construction.

Every generated waypoint is checked against the soft limits, every move (sampled along the planned motion) must stay
out of the face zone, and the mouth must be inside it. Nothing is written unless all of that passes.

Also, when the spoon is in a bowl the camera sees the spoon in that bowl: Gemini returns the box of the container the
spoon is in, which becomes that bowl's picture box (and Gemini names the food from it as usual).
"""
from __future__ import annotations

import math
from typing import Optional

from ..arm import kinematics as K
from ..arm.scoop_style import _spoon_angle, solve_rz
from ..arm.trajectory import TrajFile, check_limits, make_waypoints, plan
from ..config import ARM_JOINTS, Config, FaceCyl

ABOVE_MM = 60
DRAG_MM = 25
LIFT_MM = 60
LEVEL_DEG = 100            # spoon angle from straight up when "level" (a little nose-down so food stays on)
PRE_BACK_MM = (100, 80)
ZONE_R_MM, ZONE_BELOW_MM, ZONE_ABOVE_MM = 65, 50, 200


class SetupError(RuntimeError):
    pass


LIMITS: Optional[dict] = None          # soft limits, set by build_all (so _at only returns poses the arm can take)


def _ok(q: dict) -> bool:
    return LIMITS is None or all(LIMITS[j][0] <= q[j] <= LIMITS[j][1] for j in ARM_JOINTS if j in LIMITS)


def _at(kin, ref: dict, r: float, z: float, angle: Optional[float] = None) -> dict:
    """ref's base and roll, tip at (r, z), spoon at `angle` (default: ref's). If that exact angle is out of reach or
    out of the joint limits, the closest angle that works (searching up to 50 deg either way)."""
    tilt = 0.0 if angle is None else angle - _spoon_angle(kin, ref)
    for extra in (0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 40.0, -40.0, 50.0, -50.0):
        q = solve_rz(kin, ref, r, z, iters=30, tilt_deg=tilt + extra)
        if q is not None and _ok(q):
            return q
    raise SetupError(f"can't reach {r:.0f} mm out at {z:.0f} mm high from there within the joint limits")


def _rz(kin, q) -> tuple[float, float, float]:
    x, y, z = K.fk(kin, q)
    return math.hypot(x, y), z, math.atan2(y, x)


def build_scoop(kin, dip: dict) -> list[dict]:
    r, z, _ = _rz(kin, dip)
    a = _spoon_angle(kin, dip)
    above = _at(kin, dip, r, z + ABOVE_MM)
    drag = _at(kin, dip, r - DRAG_MM, z)
    tilt = _at(kin, dip, r - DRAG_MM - 5, z + 10, LEVEL_DEG if abs(a) >= 90 else -LEVEL_DEG)
    level = LEVEL_DEG if abs(a) >= 90 else -LEVEL_DEG
    up = _at(kin, tilt, r - DRAG_MM - 5, z + LIFT_MM, level)   # more freedom higher up: finish levelling here
    return [above, dict(dip), drag, tilt, up]


def build_pre(kin, mouth: dict) -> dict:
    r, z, _ = _rz(kin, mouth)
    for back in PRE_BACK_MM:
        try:
            return _at(kin, mouth, r - back, z)
        except SetupError:
            continue
    raise SetupError("can't find a waiting spot 8-10 cm in front of the mouth")


def build_lift(kin, scoop_end: dict, pre: dict) -> list[dict]:
    r, z, _ = _rz(kin, scoop_end)
    _, zp, _ = _rz(kin, pre)
    high = _at(kin, scoop_end, r, max(z, zp))
    return [dict(scoop_end), high, dict(pre)]


def face_cyl(kin, mouth: dict) -> FaceCyl:
    x, y, z = K.fk(kin, mouth)
    return FaceCyl(cx=round(x, 1), cy=round(y, 1), r=ZONE_R_MM, z=(round(z - ZONE_BELOW_MM, 1), round(z + ZONE_ABOVE_MM, 1)))


def _check_move(cfg: Config, kin, start: dict, poses: list[dict], name: str, may_end_inside: bool = False) -> None:
    for i, q in enumerate(poses):
        check_limits(cfg, q, f"{name}[{i}] ")
    p = plan(cfg, start, [(poses, "normal", name)])
    n = max(2, int(p.duration / 0.05))
    for k in range(n + 1):
        q = p.sample(p.duration * k / n)
        if K.in_face_zone(kin, q) and not (may_end_inside and k == n):
            x, y, z = K.fk(kin, q)
            raise SetupError(f"{name} would pass through the face zone at ({x:.0f}, {y:.0f}, {z:.0f}) mm")


def build_all(cfg: Config, dips: dict[str, dict], mouth: dict, rest: dict) -> dict:
    """{name: (kind, poses)} for every move, plus the face zone. Raises SetupError with a plain reason."""
    global LIMITS
    if not dips:
        raise SetupError("touch the spoon into at least one bowl first")
    LIMITS = cfg.arm.soft_limits_deg
    zone = face_cyl(cfg.kinematics, mouth)
    kin = cfg.kinematics.model_copy(update={"face_cyl": zone})
    if not K.in_face_zone(kin, mouth):
        raise SetupError("internal: the mouth is not inside its own zone")
    pre = build_pre(kin, mouth)
    if K.in_face_zone(kin, pre):
        raise SetupError("the waiting spot ended up in the face zone")
    out: dict = {"pre_delivery": ("pose", [pre]), "mouth_nominal": ("pose", [dict(mouth)])}
    first = None
    for bowl, dip in sorted(dips.items()):
        scoop = build_scoop(kin, dip)
        _check_move(cfg, kin, rest, scoop, f"scoop_{bowl}")
        lift = build_lift(kin, scoop[-1], pre)
        _check_move(cfg, kin, scoop[-1], lift, f"lift from {bowl}")
        out[f"scoop_{bowl}"] = ("path", scoop)
        out[f"lift_{bowl}"] = ("path", lift)
        first = first or (bowl, scoop, lift)
    out["scoop"] = ("path", first[1])
    out["lift"] = ("path", first[2])
    # the approach (pre -> mouth) and the retreat go through the face zone by design: they are gated by the face
    # check + GO and by the leaving-the-zone rule at run time. Here only their joint limits are checked.
    check_limits(cfg, mouth, "mouth "), check_limits(cfg, pre, "pre_delivery ")
    return {"moves": out, "face_cyl": zone, "kin": kin}


def save_all(cfg: Config, directory, built: dict) -> list[str]:
    saved = []
    for name, (kind, poses) in built["moves"].items():
        speed = "near_face" if name in ("pre_delivery", "mouth_nominal") else "normal"
        tf = TrajFile(name, kind, make_waypoints(cfg, poses, speed), {"source": "quick setup"})
        tf.save(directory)
        saved.append(name)
    return saved
