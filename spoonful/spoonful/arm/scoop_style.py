"""Scoop style per food texture: how hard to dig.

The LX-15D servos take only "go to this angle in this time"; there is no force or torque setting. What decides how
hard the spoon pushes into food is the motion itself, so a texture changes the taught scoop in three ways:

  depth_mm     the in-food part of the scoop is lowered by this much (spoon angle held constant), so the
               servos press into the food instead of skimming it. Capped at MAX_DEPTH_MM.
  speed        speed factor for the scoop (< 1 = slower: the servos keep up and push steadily through firm food).
  wiggle       small side-to-side saw strokes (base yaw, so the spoon stays at the same height) at the deepest
               point, to cut through firm food (jello, tofu, banana).
  contact_extra_deg
               extra stall-detection allowance while scooping, because pressing into firm food makes the servos
               lag a little. Capped at MAX_CONTACT_EXTRA_DEG: hitting the bowl or a hand still stops the arm.

The texture comes from vision (Gemini looks at the bowl) or, for a typed label, from the keyword table below.
Only the scoop changes: lift, delivery and everything near the face are untouched.
"""
from __future__ import annotations

import math
import re
from typing import Mapping, Optional

from ..config import KinCfg
from . import kinematics as K

TEXTURES = ("liquid", "soft", "thick", "firm")
MAX_DEPTH_MM = 15.0
MAX_CONTACT_EXTRA_DEG = 5.0
WIGGLE_DEG = 2.0            # base yaw: about 10 mm sideways at the bowl
IN_FOOD_BAND_MM = 15.0      # waypoints within this height of the scoop's lowest point count as "in the food"

KEYWORDS = {
    "liquid": ["soup", "broth", "milk", "juice", "stew", "chowder", "bisque", "gazpacho"],
    "soft": ["applesauce", "apple sauce", "yogurt", "yoghurt", "pudding", "puree", "purée", "custard", "mousse",
             "baby food", "smoothie", "sauce", "cream", "jam"],
    "thick": ["mashed", "oatmeal", "porridge", "rice", "hummus", "peanut butter", "refried", "polenta", "risotto",
              "cottage cheese", "mac and cheese", "macaroni", "grits", "quinoa", "couscous", "ice cream",
              "guacamole", "beans", "cereal", "oats"],
    "firm": ["jello", "jell-o", "jelly", "gelatin", "tofu", "banana", "melon", "pasta", "noodle", "cheese",
             "chicken", "pancake", "cake", "egg", "carrot", "broccoli", "potato", "apple", "peach", "pear",
             "mango", "meat", "fish", "bread", "waffle", "corn", "pea", "fruit"],
}


def texture_from_name(food: Optional[str]) -> Optional[str]:
    """Pure. A liquid word wins outright ('chicken soup' is soup); otherwise the longest keyword wins, so
    'mashed potatoes' is thick (not firm) and 'applesauce' soft (not firm)."""
    if not food:
        return None
    f = " " + re.sub(r"[^a-z ]", " ", food.lower()) + " "
    if any(re.search(r"(?<![a-z])" + re.escape(k), f) for k in KEYWORDS["liquid"]):
        return "liquid"
    best: tuple[int, Optional[str]] = (0, None)
    for tex, kws in KEYWORDS.items():
        for kw in kws:
            k = re.sub(r"[^a-z ]", " ", kw)
            if re.search(r"(?<![a-z])" + re.escape(k), f) and len(k) > best[0]:
                best = (len(k), tex)
    return best[1]


# ---------------------------------------------------------------- geometry
def _spoon_angle(kin: KinCfg, q: Mapping[str, float]) -> float:
    s, z0 = kin.sign, kin.zero_deg
    return sum(s[j] * (q[j] - z0[j]) for j in ("shoulder", "elbow", "wrist_pitch"))


def _rz(kin: KinCfg, q: Mapping[str, float]) -> tuple[float, float]:
    x, y, z = K.fk(kin, q)
    return math.hypot(x, y), z


def lower_tip(kin: KinCfg, q: Mapping[str, float], dz_mm: float, iters: int = 12) -> Optional[dict]:
    """Same pose with the spoon tip dz_mm lower, same reach and same spoon angle (shoulder, elbow and wrist pitch
    adjusted; base and roll unchanged). None if it does not converge (out of reach)."""
    r0, z0 = _rz(kin, q)
    return solve_rz(kin, q, r0, z0 - dz_mm, iters)


def solve_rz(kin: KinCfg, q: Mapping[str, float], r_mm: float, z_mm: float, iters: int = 12,
             tilt_deg: float = 0.0) -> Optional[dict]:
    """q with the spoon tip at reach r_mm and height z_mm, spoon angle unchanged + tilt_deg (base and roll
    unchanged)."""
    import numpy as np
    joints = ("shoulder", "elbow", "wrist_pitch")
    target = np.array([r_mm, z_mm, _spoon_angle(kin, q) + tilt_deg])
    cur = dict(q)

    def resid(p: dict) -> np.ndarray:
        r, z = _rz(kin, p)
        return np.array([r, z, _spoon_angle(kin, p)]) - target

    for _ in range(iters):
        e = resid(cur)
        if abs(e[0]) < 0.2 and abs(e[1]) < 0.2 and abs(e[2]) < 0.1:
            return cur
        J = np.zeros((3, 3))
        for i, j in enumerate(joints):
            p = dict(cur)
            p[j] += 0.1
            J[:, i] = (resid(p) - e) / 0.1
        try:
            step = np.linalg.solve(J, -e)
        except np.linalg.LinAlgError:
            return None
        step = np.clip(step, -5, 5)
        for i, j in enumerate(joints):
            cur[j] += float(step[i])
    e = resid(cur)
    return cur if abs(e[0]) < 0.5 and abs(e[1]) < 0.5 and abs(e[2]) < 0.2 else None


def scoop_anchor(kin: KinCfg, poses: list[Mapping[str, float]]) -> tuple[float, float]:
    """Table (x, y) of the scoop's lowest point: where the bowl was when the scoop was taught."""
    p = min(poses, key=lambda q: K.fk(kin, q)[2])
    x, y, _ = K.fk(kin, p)
    return x, y


def shift_scoop(kin: KinCfg, poses: list[Mapping[str, float]], dx: float, dy: float) -> Optional[list[dict]]:
    """Pure. The whole scoop moved dx, dy mm on the table: the base turns to the new direction and the arm reaches in
    or out, same height and spoon angle at every waypoint. None if some waypoint can't be reached."""
    s, z0 = kin.sign["base"], kin.zero_deg["base"]
    out = []
    for q in poses:
        x, y, z = K.fk(kin, q)
        xt, yt = x + dx, y + dy
        p = dict(q)
        p["base"] = z0 + math.degrees(math.atan2(yt, xt)) / s
        p = solve_rz(kin, p, math.hypot(xt, yt), z)
        if p is None:
            return None
        out.append(p)
    return out


def adapt_scoop(kin: KinCfg, poses: list[Mapping[str, float]], depth_mm: float = 0.0,
                wiggle: int = 0) -> tuple[list[dict], dict]:
    """Pure. The taught scoop, dug deeper by depth_mm over its in-food part, with `wiggle` saw strokes at the deepest
    point. The first and last waypoints never move (they join the rest of the bite). Returns (poses, info)."""
    out = [dict(p) for p in poses]
    info = {"depth_mm": 0.0, "lowered": 0, "wiggle": 0}
    depth_mm = max(0.0, min(MAX_DEPTH_MM, depth_mm))
    if len(out) < 3:
        return out, info
    zs = [K.fk(kin, p)[2] for p in out]
    inner = range(1, len(out) - 1)
    z_min = min(zs[i] for i in inner)
    deepest = min(inner, key=lambda i: zs[i])
    if depth_mm > 0:
        for i in inner:
            if zs[i] <= z_min + IN_FOOD_BAND_MM:
                low = lower_tip(kin, out[i], depth_mm)
                if low is not None:
                    out[i] = low
                    info["lowered"] += 1
        info["depth_mm"] = depth_mm if info["lowered"] else 0.0
    if wiggle > 0:
        base = out[deepest]
        saw = []
        for _ in range(int(wiggle)):
            for d in (WIGGLE_DEG, -WIGGLE_DEG):
                p = dict(base)
                p["base"] = base["base"] + d
                saw.append(p)
        out[deepest + 1:deepest + 1] = saw + [dict(base)]
        info["wiggle"] = int(wiggle)
    return out, info
