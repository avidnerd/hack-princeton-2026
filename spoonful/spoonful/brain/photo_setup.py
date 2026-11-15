"""Setup for a new table: the diner's mouth from one touch, the bowls from one overhead photo.

save_mouth   the spoon is at the diner's lips (moved there by hand): save the mouth pose, the waiting pose in front
             of it and the face zone around it.
find_bowls   one photo from above: find every bowl (photo_map), outline it, and work out whether the arm can scoop
             from it. Nothing is changed yet: the diner chooses which bowls to be served from.
use_bowls    the chosen bowls become the robot's bowls: a scoop and a lift for each, and vision picks how to scoop
             each food. Bowls that were not chosen are unknown to the robot (voice and planning can't pick them).
status       what the Setup screen shows for each step.
"""
from __future__ import annotations

import math
import time

from ..arm import kinematics as K
from ..arm.scoop_style import dip_index
from ..arm.trajectory import plan
from ..config import ARM_JOINTS, BowlCfg
from ..perception.photo_map import PHOTO_TIP_MM, PROMPT, analyze_robust, predicted_surface_mm
from . import quick_setup as QS

MAX_PHOTO_PX = 1600
PLACED_WITHIN_MM = 45      # how close to the photo pose the measured spoon tip must be to count as in place


NOT_PLACED = "The spoon wasn't in place for this photo. Press \"Pose for photo\", then retake it."


class SetupProblem(RuntimeError):
    """Something the operator can fix; the message says what."""


def save_mouth(sys) -> dict:
    mouth = {j: sys.arm.state.deg[j] for j in ARM_JOINTS}
    rest = sys.lib.pose("rest") if sys.lib.has("rest") else dict(sys.cfg.arm.home_pose)
    try:
        built = QS.build_all(sys.cfg, {}, mouth, rest, mouth_only=True)
    except Exception as e:
        raise SetupProblem(f"That position doesn't work: {e}") from e
    QS.save_all(sys.cfg, sys.lib.dir, built)
    sys.cfg.kinematics.face_cyl = built["face_cyl"]
    sys.cfg.save()
    sys.lib.reload()
    sys.setup_state["mouth_at"] = time.time()
    x, y, z = K.fk(sys.cfg.kinematics, mouth)
    sys.log.event("setup_mouth", detail_tip_mm=[round(x), round(y), round(z)])
    return {"tip_mm": [round(x), round(y), round(z)]}


def outline(img, box) -> list[list[int]]:
    """The bowl's outline inside its box, as a polygon in photo pixels. GrabCut separates the bowl from the table
    around it; if that gives something implausible, the box's ellipse is used."""
    import cv2
    import numpy as np
    x0, y0, x1, y1 = (int(v) for v in box)
    bw, bh = x1 - x0, y1 - y0
    ellipse = [[int(x0 + bw / 2 + bw / 2 * math.cos(a)), int(y0 + bh / 2 + bh / 2 * math.sin(a))]
               for a in (2 * math.pi * k / 24 for k in range(24))]
    if bw < 12 or bh < 12:
        return ellipse
    pad = int(0.25 * max(bw, bh))
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = img[cy0:cy1, cx0:cx1]
    k = min(1.0, 220 / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=k, fy=k) if k < 1 else crop
    rect = (int((x0 - cx0) * k), int((y0 - cy0) * k), max(2, int(bw * k)), max(2, int(bh * k)))
    mask = np.zeros(small.shape[:2], np.uint8)
    try:
        cv2.grabCut(small, mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return ellipse
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ellipse
    c = max(contours, key=cv2.contourArea)
    if not 0.35 * rect[2] * rect[3] <= cv2.contourArea(c) <= 1.05 * rect[2] * rect[3]:
        return ellipse
    c = cv2.approxPolyDP(cv2.convexHull(c), 0.012 * cv2.arcLength(c, True), True).reshape(-1, 2)
    return [[int(px / k + cx0), int(py / k + cy0)] for px, py in c]


def spoon_placed(tip) -> bool:
    """Is the spoon tip at the photo pose (within the arm's sag of it)? The photo uses it as its reference point."""
    return math.dist(tip, PHOTO_TIP_MM) <= PLACED_WITHIN_MM


def _public(c: dict) -> dict:
    return {k: v for k, v in c.items() if not k.startswith("_")}


def find_bowls(sys, img) -> list[dict]:
    """img: BGR overhead photo. Saves it for the Setup screen and returns the bowls found in it, each with an
    outline and whether the arm can scoop from it. Raises SetupProblem."""
    import cv2
    h, w = img.shape[:2]
    k = MAX_PHOTO_PX / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)))
    h, w = img.shape[:2]
    cv2.imwrite(str(sys.cfg.path("data/phone_photo.jpg")), img)
    sys.setup_state.update(candidates=[], photo_size=[w, h], photo_at=time.time())

    if sys.fsm.state.value != "IDLE":
        raise SetupProblem("End the session first, then retake the photo.")
    if sys.plate.image_generate is None:
        raise SetupProblem("Vision isn't available. Check the Gemini key.")
    if not (sys.lib.has("pre_delivery") and sys.lib.has("mouth_nominal")):
        raise SetupProblem("Set your mouth position first (step 1).")
    kin = sys.cfg.kinematics
    tip = K.fk(kin, sys.arm.state.deg)
    placed = spoon_placed(tip)       # without it the bowls can be outlined but not located: shown, not choosable
    try:
        m = analyze_robust(img, lambda data: sys.plate.image_generate(data, PROMPT), tip_mm=(tip[0], tip[1]))
    except Exception as e:
        if not placed:
            raise SetupProblem(NOT_PLACED) from e
        raise SetupProblem(f"Couldn't read the photo: {e}") from e

    QS.LIMITS = sys.cfg.arm.soft_limits_deg
    pre = sys.lib.pose("pre_delivery")
    ref = sys.lib.pose("rest") if sys.lib.has("rest") else dict(sys.cfg.arm.home_pose)
    found = []
    for i, bw in enumerate(m["bowls"]):
        box = [int(v) for v in bw["box_px"]]
        x, y = bw["mm"]
        c = {"id": i, "food": bw.get("food") or "food", "amount": bw.get("amount"), "texture": bw.get("texture"),
             "sure": bool(bw["sure"]), "votes": bw["votes"], "views": bw["views"],
             "uncertainty_mm": bw["uncertainty_mm"], "at_cm": [round(x / 10), round(y / 10)],
             "box": box, "poly": outline(img, box), "ok": True, "why": "", "used": False}
        surface = predicted_surface_mm(bw.get("amount"))
        try:
            scoop = QS.build_scoop_at(kin, ref, x, y, surface)
            if any(K.in_face_zone(kin, q) for q in scoop):
                raise QS.SetupError("too close to your face")
            lift = QS.build_lift(kin, scoop[-1], pre)
            plan(sys.cfg, scoop[0], [(scoop, "normal", "check")])
            plan(sys.cfg, scoop[-1], [(lift, "normal", "check")])
            crop = img[max(0, box[1] - 15):box[3] + 15, max(0, box[0] - 15):box[2] + 15]
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            c.update(_scoop=scoop, _lift=lift, _surface=surface, _mm=(x, y),
                     _crop=cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
        except Exception as e:
            c.update(ok=False, why="Too close to your face" if "face" in str(e) else "Out of reach")
        if not placed:
            c.update(ok=False, why="Retake with the spoon in place")
        found.append(c)
    sys.setup_state["candidates"] = found
    sys.log.event("photo_bowls", detail_found=[_public(c) | {"poly": len(c["poly"])} for c in found],
                  mm_per_px=round(m["mm_per_px"], 3))
    if not placed:
        raise SetupProblem(NOT_PLACED)
    if not any(c["ok"] for c in found):
        raise SetupProblem("No bowls the arm can reach in this photo. Move them closer and retake it."
                           if found else "No bowls found in this photo. Retake it from straight above.")
    return [_public(c) for c in found]


def use_bowls(sys, ids: list[int]) -> list[dict]:
    """Serve from exactly these bowls (ids from find_bowls)."""
    if sys.fsm.state.value != "IDLE":
        raise SetupProblem("End the session first.")
    cands = sys.setup_state.get("candidates") or []
    chosen = [c for c in cands if c["id"] in set(ids) and c["ok"]]
    if not chosen:
        raise SetupProblem("Choose at least one bowl.")
    chosen.sort(key=lambda c: math.atan2(c["_mm"][1], c["_mm"][0]), reverse=True)     # left to right, as the arm sees them
    moves, bowls = {}, []
    for n, c in enumerate(chosen, 1):
        c["name"] = f"bowl{n}"
        moves[f"scoop_bowl{n}"] = ("path", c["_scoop"])
        moves[f"lift_bowl{n}"] = ("path", c["_lift"])
        bowls.append(BowlCfg(name=c["name"], roi=(0, 0, 1280, 720), label=c["food"], watch=False))
    moves["scoop"], moves["lift"] = moves["scoop_bowl1"], moves["lift_bowl1"]
    QS.save_all(sys.cfg, sys.lib.dir, {"moves": moves})
    sys.cfg.plate.bowls = bowls
    sys.lib.reload()
    sys.plate.reset_bowls()
    sys.fsm.last_bowl, sys.fsm.bite_queue = None, []
    sys.aim_check.forget()                       # new bowls, maybe a moved laptop: the camera check starts over
    sys.setup_state["nudges"] = {}
    result = []
    for c in chosen:
        sys.food_surface.est[c["name"]] = c["_surface"]
        sys.plate.bowls[c["name"]].update(amount=c["amount"], vision_texture=c["texture"])
        attack = sys.plate.plan_attack(c["name"], c["_crop"])
        next(b for b in sys.cfg.plate.bowls if b.name == c["name"]).attack = attack
        c.update(strategy=attack["strategy"], strategy_why=attack["why"])
        result.append(_public(c))
    for c in cands:
        c["used"] = c in chosen
    sys.cfg.save()
    sys.setup_state.update(table_at=time.time(), bowls=result)
    sys.log.event("photo_map", detail_bowls=[{k: r[k] for k in ("name", "food", "strategy", "at_cm")} for r in result])
    return result


AIM_ABOVE_MM = 60          # how far above the dip the spoon hovers when showing where it will scoop
MAX_NUDGE_MM = 60          # total correction allowed from where the photo put the bowl


def _dip(sys, name: str) -> dict:
    scoop = f"scoop_{name}"
    if not sys.lib.has(scoop):
        raise SetupProblem("That bowl has no scoop yet. Map the table first.")
    poses = sys.lib.items[scoop].poses
    return poses[dip_index(sys.cfg.kinematics, poses)]


def aim(sys, name: str) -> None:
    """Hover the spoon tip above where it will dip into this bowl, so the aim can be checked by eye."""
    if sys.fsm.state.value != "IDLE":
        raise SetupProblem("End the session first.")
    kin = sys.cfg.kinematics
    dip = _dip(sys, name)
    x, y, z = K.fk(kin, dip)
    QS.LIMITS = sys.cfg.arm.soft_limits_deg
    try:
        above = QS._at(kin, dip, math.hypot(x, y), z + AIM_ABOVE_MM)
    except QS.SetupError as e:
        raise SetupProblem("Can't hover above that bowl.") from e
    sys.arm.goto(above, "normal", 0.5, name="aim")


def move_scoop(sys, name: str, further_mm: float, left_mm: float) -> None:
    """Move this bowl's scoop on the table: further from / closer to the arm's base, and to the arm's left / right.
    The lift is rebuilt to start where the scoop now ends. Raises SetupProblem if the spoon can't scoop there."""
    from ..arm.scoop_style import scoop_anchor, shift_scoop
    kin = sys.cfg.kinematics
    if not sys.lib.has(f"scoop_{name}"):
        raise SetupProblem("That bowl has no scoop yet. Map the table first.")
    scoop = sys.lib.items[f"scoop_{name}"].poses
    x, y = scoop_anchor(kin, scoop)
    r = math.hypot(x, y)
    dx = further_mm * x / r - left_mm * y / r
    dy = further_mm * y / r + left_mm * x / r
    QS.LIMITS = sys.cfg.arm.soft_limits_deg
    moved = shift_scoop(kin, scoop, dx, dy)
    try:
        if moved is not None:
            try:
                plan(sys.cfg, moved[0], [(moved, "normal", "check")])
            except Exception:
                moved = None
        if moved is None:        # the same shape doesn't fit there: build a fresh scoop at the new spot
            dip = _dip(sys, name)
            ref = sys.lib.pose("rest") if sys.lib.has("rest") else dict(sys.cfg.arm.home_pose)
            moved = QS.build_scoop_at(kin, ref, x + dx, y + dy, K.fk(kin, dip)[2] + QS.DIP_BELOW_SURFACE_MM)
        if any(K.in_face_zone(kin, q) for q in moved):
            raise QS.SetupError("too close to your face")
        lift = QS.build_lift(kin, moved[-1], sys.lib.pose("pre_delivery"))
        plan(sys.cfg, moved[0], [(moved, "normal", "check")])
        plan(sys.cfg, moved[-1], [(lift, "normal", "check")])
    except Exception as e:
        raise SetupProblem(f"The spoon can't scoop there ({e}).") from e
    moves = {f"scoop_{name}": ("path", moved), f"lift_{name}": ("path", lift)}
    if name == next(iter(sys.plate.bowls), None):
        moves["scoop"], moves["lift"] = moves[f"scoop_{name}"], moves[f"lift_{name}"]
    QS.save_all(sys.cfg, sys.lib.dir, {"moves": moves})
    sys.lib.reload()
    sys.log.event("scoop_moved", detail_bowl=name, further_mm=round(further_mm, 1), left_mm=round(left_mm, 1))


def nudge(sys, name: str, further_mm: float, left_mm: float) -> dict:
    """A correction made from the Setup screen: move the scoop, then hover over the new spot to show it."""
    if sys.fsm.state.value != "IDLE":
        raise SetupProblem("End the session first.")
    total = sys.setup_state.setdefault("nudges", {}).get(name, [0.0, 0.0])
    new_total = [total[0] + further_mm, total[1] + left_mm]
    if math.hypot(*new_total) > MAX_NUDGE_MM:
        raise SetupProblem("That's more than 6 cm from where the photo put the bowl. Retake the photo instead.")
    move_scoop(sys, name, further_mm, left_mm)
    sys.setup_state["nudges"][name] = new_total
    aim(sys, name)
    return {"further_mm": new_total[0], "left_mm": new_total[1]}


def status(sys) -> dict:
    st = sys.setup_state
    kin = sys.cfg.kinematics
    tip = K.fk(kin, sys.arm.state.deg) if sys.arm.state.ok else None
    posed = bool(tip) and sys.arm.torque_on and spoon_placed(tip)
    return {
        "mouth": {"set": sys.lib.has("mouth_nominal") and sys.lib.has("pre_delivery"), "at": st.get("mouth_at"),
                  "released": not sys.arm.torque_on},
        "table": {"at": st.get("table_at"), "bowls": st.get("bowls") or [], "posed": posed,
                  "candidates": [_public(c) for c in st.get("candidates") or []],
                  "photo_size": st.get("photo_size"), "photo_at": st.get("photo_at")},
        "looking": {"at": st.get("looking_at")},
        "aim": sys.aim_check.status(),
    }
