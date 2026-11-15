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


def dip_index(kin: KinCfg, poses: list[Mapping[str, float]]) -> int:
    """Index of the waypoint where the spoon first reaches the bottom of the scoop. The drag that follows is at the
    same height, so "the lowest point" alone would flip between the two."""
    idx = range(1, len(poses) - 1) if len(poses) >= 3 else range(len(poses))
    zs = {i: K.fk(kin, poses[i])[2] for i in idx}
    z_min = min(zs.values())
    return next(i for i in idx if zs[i] <= z_min + 1.0)


def scoop_anchor(kin: KinCfg, poses: list[Mapping[str, float]]) -> tuple[float, float]:
    """Table (x, y) of the scoop's lowest point: where the bowl was when the scoop was taught."""
    p = poses[dip_index(kin, poses)]
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
    deepest = dip_index(kin, out)
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


# ---------------------------------------------------------------- attack strategies (how to go into a food)
# A person doesn't scoop jelly like applesauce: they stab steeply and cut jelly, skim applesauce, dig into rice,
# scrape when little is left, keep a spoon of soup level. A vision model picks the strategy from a close-up of the
# bowl (perception/plate.py ATTACK_PROMPT); these are the defaults and the safe ranges every choice is clamped to.

STRATEGIES = {
    "skim":         {"entry_angle_deg": 25, "depth_mm": 5,  "saw_strokes": 0, "speed": 0.8, "drag_mm": 30, "contact_extra_deg": 0},
    "dig":          {"entry_angle_deg": 45, "depth_mm": 10, "saw_strokes": 0, "speed": 0.7, "drag_mm": 25, "contact_extra_deg": 2},
    "stab_and_saw": {"entry_angle_deg": 60, "depth_mm": 10, "saw_strokes": 2, "speed": 0.5, "drag_mm": 15, "contact_extra_deg": 4},
    "scrape":       {"entry_angle_deg": 50, "depth_mm": 12, "saw_strokes": 0, "speed": 0.6, "drag_mm": 35, "contact_extra_deg": 2},
    "ladle":        {"entry_angle_deg": 20, "depth_mm": 8,  "saw_strokes": 0, "speed": 0.5, "drag_mm": 10, "contact_extra_deg": 0},
}
RULE_WHY = {"skim": "it's soft, so I'll skim the top", "dig": "it's thick, so I'll dig in at an angle",
            "stab_and_saw": "it's firm, so I'll cut in steeply", "scrape": "there's little left, so I'll scrape the bottom",
            "ladle": "it's liquid, so I'll keep the spoon level"}
TEXTURE_STRATEGY = {"soft": "skim", "thick": "dig", "firm": "stab_and_saw", "liquid": "ladle"}
LIMITS = {"entry_angle_deg": (20, 70), "depth_mm": (0, MAX_DEPTH_MM), "saw_strokes": (0, 3), "speed": (0.4, 1.0),
          "drag_mm": (5, 40), "contact_extra_deg": (0, MAX_CONTACT_EXTRA_DEG)}


def default_strategy(texture: Optional[str], amount: Optional[str] = None) -> dict:
    name = "scrape" if amount == "low" and texture != "liquid" else TEXTURE_STRATEGY.get(texture or "soft", "skim")
    return {"strategy": name, **STRATEGIES[name], "why": RULE_WHY[name], "source": "rules"}


def verify_strategy(d: object, texture: Optional[str] = None, amount: Optional[str] = None) -> dict:
    """Pure. A model's attack plan -> a safe one: known strategy, every number clamped to LIMITS, missing numbers from
    the strategy's defaults. Anything unusable -> the rule-based default for the texture."""
    if not isinstance(d, dict) or d.get("strategy") not in STRATEGIES:
        return default_strategy(texture, amount)
    out = {"strategy": d["strategy"], "why": str(d.get("why") or "")[:120], "source": "vision"}
    base = STRATEGIES[d["strategy"]]
    for k, (lo, hi) in LIMITS.items():
        try:
            v = float(d.get(k, base[k]))
        except (TypeError, ValueError):
            v = base[k]
        out[k] = max(lo, min(hi, v))
    out["saw_strokes"] = int(round(out["saw_strokes"]))
    return out


def shape_scoop(kin: KinCfg, poses: list[Mapping[str, float]], entry_angle_deg: float,
                drag_mm: float = 25.0, limits: Optional[dict] = None) -> tuple[list[dict], dict]:
    """Pure. Re-angle the in-food part of a scoop so the spoon enters at entry_angle_deg below horizontal, and make the
    drag along the food drag_mm long (moving toward the arm). Points that can't be reached keep their taught shape."""
    out = [dict(p) for p in poses]
    info = {"entry_angle_deg": entry_angle_deg, "reangled": 0, "drag_mm": 0.0}
    if len(out) < 3:
        return out, info
    zs = [K.fk(kin, p)[2] for p in out]
    inner = range(1, len(out) - 1)
    z_min = min(zs[i] for i in inner)
    k = dip_index(kin, out)
    sign = 1.0 if _spoon_angle(kin, out[k]) >= 0 else -1.0
    target = sign * (90.0 + entry_angle_deg)            # angle from straight up: 90 = level, 180 = straight down
    band = [i for i in inner if zs[i] <= z_min + IN_FOOD_BAND_MM]
    # the points just above the bowl on the way in get the entry angle too, so the spoon arrives already tilted
    # (otherwise it swings ~90 deg of joint travel right over the food)
    xk, yk, _ = K.fk(kin, out[k])
    above = [i for i in range(0, k) if i not in band and math.hypot(K.fk(kin, out[i])[0] - xk, K.fk(kin, out[i])[1] - yk) < 40]
    for i in above + band:
        r, z = _rz(kin, out[i])
        want = target - _spoon_angle(kin, out[i])
        for frac in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):          # as close to the planned angle as the arm can get
            q = solve_rz(kin, out[i], r, z, iters=25, tilt_deg=want * frac)
            if q is not None and (limits is None or all(limits[j][0] <= q[j] <= limits[j][1] for j in limits if j in q)):
                out[i] = q
                info["reangled"] += 1
                break
    # drag: after the dip, the in-food points move this far toward the arm (radially), keeping their height
    after = [i for i in band if i > k]
    r_dip, z_dip = _rz(kin, out[k])
    if after:
        r_end, _ = _rz(kin, out[after[-1]])
        extra = drag_mm - max(0.0, r_dip - r_end)
        for i in after:
            r, z = _rz(kin, out[i])
            q = solve_rz(kin, out[i], r - extra, z, iters=25)
            if q is not None and (limits is None or all(limits[j][0] <= q[j] <= limits[j][1] for j in limits if j in q)):
                out[i] = q
        info["drag_mm"] = round(drag_mm, 1)
    else:
        q = solve_rz(kin, out[k], r_dip - drag_mm, z_dip, iters=25)
        if q is not None:
            out.insert(k + 1, q)
            info["drag_mm"] = round(drag_mm, 1)
    return out, info
