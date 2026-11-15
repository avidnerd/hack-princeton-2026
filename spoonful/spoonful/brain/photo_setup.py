"""Setup for a new table: the diner's mouth from one touch, the bowls from one overhead photo.

save_mouth   the spoon is at the diner's lips (moved there by hand): save the mouth pose, the waiting pose in front
             of it and the face zone around it.
find_bowls   one photo from above: find every bowl (photo_map), outline it, and work out whether the arm can scoop
             from it. Nothing is changed yet: the diner chooses which bowls to be served from.
use_bowls    the chosen bowls become the robot's bowls: a scoop and a lift for each, and vision picks how to scoop
             each food. Bowls that were not chosen are unknown to the robot (voice and planning can't pick them).
             Each also gets a cup (where the photo put it, how wide, how unsure): not yet confirmed.
confirm_cup  the spoon is hovering over the middle of a cup's opening and a person says so: where the arm measures
             its own tip becomes the cup's centre, and the table camera learns how that opening looks.
status       what the Setup screen shows for each step.
"""
from __future__ import annotations

import math
import time

from ..arm import kinematics as K
from ..arm.scoop_style import dip_index
from ..arm.trajectory import plan
from ..config import ARM_JOINTS, BowlCfg, CupCfg
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
        food = bw.get("food") or "food"
        food = sys.cfg.plate.food_names.get(food.strip().lower(), food)     # the operator's name for it wins
        c = {"id": i, "food": food, "amount": bw.get("amount"), "texture": bw.get("texture"),
             "sure": bool(bw["sure"]), "votes": bw["votes"], "views": bw["views"],
             "uncertainty_mm": bw["uncertainty_mm"], "diameter_mm": round(float(bw.get("diameter_mm") or 0), 1),
             "at_cm": [round(x / 10), round(y / 10)],
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
        bowls.append(BowlCfg(name=c["name"], roi=(0, 0, 1280, 720), label=c["food"], watch=False,
                             cup=_photo_cup(c, _measured_before(sys, c["food"]))))
    moves["scoop"], moves["lift"] = moves["scoop_bowl1"], moves["lift_bowl1"]
    QS.save_all(sys.cfg, sys.lib.dir, {"moves": moves})
    sys.cfg.plate.bowls = bowls
    sys.lib.reload()
    sys.plate.reset_bowls()
    sys.fsm.last_bowl, sys.fsm.bite_queue = None, []
    sys.aim_check.forget()                       # new bowls, maybe a moved laptop: the camera check starts over
    if getattr(sys, "rim", None) is not None:
        sys.rim.forget()                         # and no cup is where the camera learned it
    sys.setup_state.update(nudges={}, aimed={}, confirmed_aim={}, cups={})
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


def aim_pose(sys, name: str) -> dict:
    """Where "Show aim" holds the spoon. A bowl with a cup: tip over the cup's centre, clear of its rim, which is
    where the staged scoop goes in. A confirmed cup: the very pose it was confirmed in (the cup's centre is where
    the arm measured its tip there, sag included, so commanding that point again would stop short of it).
    Otherwise: above the saved scoop's dip."""
    kin = sys.cfg.kinematics
    cup = _cup(sys, name)
    held = (sys.setup_state.get("confirmed_aim") or {}).get(name)
    if cup is not None and cup.confirmed and held:
        return dict(held)
    dip = _dip(sys, name)
    x, y, z = K.fk(kin, dip)
    q, z = dict(dip), z + AIM_ABOVE_MM
    if cup is not None:
        rc = sys.cfg.plate.rim
        x, y = cup.x_mm, cup.y_mm
        rim = cup.rim_mm if cup.rim_mm is not None else rc.default_rim_mm
        z = max(z, rim + rc.hover_above_rim_mm + sys.cfg.spoon.bowl_depth_mm)
        q["base"] = kin.zero_deg["base"] + math.degrees(math.atan2(y, x)) / kin.sign["base"]
    QS.LIMITS = sys.cfg.arm.soft_limits_deg
    try:
        return QS._at(kin, q, math.hypot(x, y), z)
    except QS.SetupError as e:
        raise SetupProblem("Can't hover above that bowl.") from e


def aim(sys, name: str) -> None:
    """Hover the spoon tip above where it will go into this bowl, so the aim can be checked by eye."""
    if sys.fsm.state.value != "IDLE":
        raise SetupProblem("End the session first.")
    above = aim_pose(sys, name)
    sys.arm.goto(above, "normal", 0.5, name="aim")
    sys.setup_state.setdefault("aimed", {})[name] = dict(above)      # confirm_cup checks the arm is still here


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
    _cup_moved(sys, name, dx, dy)
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


# ---------------------------------------------------------------------------------------------- cups
CONFIRM_WITHIN_MM = 35     # measured tip vs the pose "Show aim" sent the arm to (the arm sags a centimetre or two)
CONFIRMED_UNCERTAINTY_MM = 3.0
UNCONFIRMED_MM = CupCfg.model_fields["uncertainty_mm"].default      # what a cup falls back to when it is moved
MAX_CUP_MM = 300           # a height or width beyond this is a typing slip
CAMERA_SETTLE_S = 0.7      # after the arm stops, before the picture the camera learns the cup from
ROI_PAD = 0.12             # the dish's box, grown by this much, is where the opening is looked for

NOT_HOVERING = "The spoon isn't hovering over this cup. Press Show aim, centre the spoon with the arrows, then confirm."
SEEN = "Position confirmed. Camera sees the opening."
WHY_UNSEEN = {              # RimNotFound / Observation reasons, in plain words
    "no_frame": "there is no picture from the table camera", "stale_frame": "the table camera's picture has stalled",
    "no_ellipse": "it can't make out the rim", "low_contrast": "it can't make out the rim",
    "ambiguous": "it can't tell which ring is the rim", "partly_hidden": "something is in the way",
    "too_flat": "the camera is too low to see into the cup", "no_reference": "confirm the cup again",
    "reference_stale": "it was confirmed a long time ago; confirm it again",
    "camera_moved": "the camera has moved; confirm the cup again",
    "size_changed": "the camera or the cup has moved; confirm it again",
    "moved_too_far": "the cup has moved too far; line it up again",
    "moved_unmeasurable": "the cup has moved; line it up again",
}


def _bowl(sys, name: str) -> BowlCfg:
    b = next((b for b in sys.cfg.plate.bowls if b.name == name), None)
    if b is None:
        raise SetupProblem("There is no such bowl. Map the table first.")
    return b


def _cup(sys, name: str):
    b = next((b for b in sys.cfg.plate.bowls if b.name == name), None)
    return b.cup if b is not None else None


def _measured_before(sys, food: str):
    """The cup this food was in before the table was mapped again, if its size was measured: a container keeps
    its height and width when it is moved, so a new photo must not replace a ruler with a guess."""
    for b in sys.cfg.plate.bowls:
        if b.label == food and b.cup is not None and b.cup.rim_mm is not None:
            return b.cup
    return None


def _photo_cup(c: dict, before=None) -> CupCfg:
    """The cup as the overhead photo saw it: its box's centre, the box's width less the walls, and the photo's own
    spread as how unsure the centre is. Not confirmed, and its height is not known."""
    wall = CupCfg.model_fields["wall_mm"].default
    x, y = c["_mm"]
    cup = CupCfg(x_mm=round(float(x), 1), y_mm=round(float(y), 1),
                 radius_mm=round(max(1.0, float(c.get("diameter_mm") or 0) / 2 - wall), 1),
                 uncertainty_mm=float(c.get("uncertainty_mm") or UNCONFIRMED_MM), source="photo")
    if before is not None:          # position from the new photo, size from the ruler
        cup.radius_mm, cup.rim_mm, cup.wall_mm, cup.floor_mm = (before.radius_mm, before.rim_mm, before.wall_mm,
                                                                before.floor_mm)
        cup.food_below_rim_mm = before.food_below_rim_mm
    return cup


def _cup_moved(sys, name: str, dx: float, dy: float) -> None:
    """The bowl's scoop was moved on the table: its cup moves with it, and nobody has seen the spoon over the new
    spot, so it is no longer confirmed and the camera's memory of it is dropped."""
    b = next((b for b in sys.cfg.plate.bowls if b.name == name), None)
    if b is None or b.cup is None:
        return
    c = b.cup
    b.cup = c.model_copy(update=dict(
        x_mm=round(c.x_mm + dx, 1), y_mm=round(c.y_mm + dy, 1), confirmed=False, source="photo",
        uncertainty_mm=max(c.uncertainty_mm, UNCONFIRMED_MM) if c.confirmed else c.uncertainty_mm))
    if getattr(sys, "rim", None) is not None:
        sys.rim.forget(name)
    for k in ("confirmed_aim", "aimed", "cups"):
        sys.setup_state.setdefault(k, {}).pop(name, None)
    sys.cfg.save()


def _dishes(sys, img) -> list[dict]:
    """Every dish the table camera sees, with its box in the pixels of img. (AimCheck._dishes asks the same
    question but keeps only a point on each dish; the rim search needs the region.) "x" is in the 1280-wide
    reference, like the spoon tip AimCheck.pick_dish compares it with."""
    import cv2
    from .aim_check import DISHES_PROMPT
    jpg = cv2.imencode(".jpg", cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    d = sys.plate.image_generate(jpg, DISHES_PROMPT)
    if isinstance(d, dict):
        d = d.get("dishes") or d.get("items") or [d]
    h, w = img.shape[:2]
    out = []
    for b in d if isinstance(d, list) else []:
        box = b.get("box_2d") if isinstance(b, dict) else None
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        y0, x0, y1, x1 = (float(v) for v in box)
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        out.append({"food": str(b.get("food") or ""), "x": (x0 + x1) / 2 * 1.28,
                    "box": (x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000)})
    return out


def _learn_view(sys, name: str) -> tuple[bool, str]:
    """Just after a confirm: get the arm out of the picture, then let the table camera learn how this cup's opening
    looks from where it stands. Vision is asked (setup time only) which dish is this one; the rim itself is then
    found by the rim observer. (True, "") or (False, why in plain words). The cup stays confirmed either way."""
    from ..perception.plate import match_food
    from ..perception.rim import RimNotFound
    from .aim_check import AimCheck
    rim = getattr(sys, "rim", None)
    if rim is None:
        return False, "the camera check isn't running"
    rim.forget(name)
    cup = _cup(sys, name)
    vision = sys.plate.image_generate is not None
    tip = None
    if vision and len(sys.cfg.plate.bowls) > 1:          # while the spoon still points at the cup: which dish is it
        try:
            tip = sys.aim_check._tip_px()
        except Exception:
            tip = None
    try:
        rest = sys.lib.pose("rest") if sys.lib.has("rest") else dict(sys.cfg.arm.home_pose)
        arrived = sys.arm.goto(dict(rest), "normal", 0.6, name="cup_confirmed").wait(15).ok
    except Exception:
        arrived = False
    if not arrived or sys.fsm.state.value != "IDLE":
        return False, "the arm didn't move out of the camera's way"
    time.sleep(CAMERA_SETTLE_S)
    if not vision:
        return False, "vision isn't available"
    img = sys.table_frame()
    if img is None:
        return False, WHY_UNSEEN["no_frame"]
    try:
        dishes = _dishes(sys, img)
    except Exception as e:
        sys.log.event("cup_view_failed", detail_bowl=name, error=repr(e)[:200])
        return False, "vision didn't answer"
    food = sys.plate.food_of(name) or ""
    if tip is None and len(dishes) > 1:                 # only the food's name can tell them apart
        dishes = [d for d in dishes if food and match_food(d["food"], {"it": food})]
        if len(dishes) != 1:
            return False, "it can't tell which cup is this one"
    dish = AimCheck.pick_dish(dishes, food, tip if tip is not None else (0.0, 0.0))
    if dish is None:
        return False, "it doesn't see a cup"
    h, w = img.shape[:2]
    x0, y0, x1, y1 = dish["box"]
    px, py = ROI_PAD * (x1 - x0), ROI_PAD * (y1 - y0)
    x0, y0, x1, y1 = max(0.0, x0 - px), max(0.0, y0 - py), min(float(w), x1 + px), min(float(h), y1 + py)
    try:
        rim.set_reference(name, img, (x0, y0, x1 - x0, y1 - y0), (cup.x_mm, cup.y_mm), cup.radius_mm)
    except RimNotFound as e:
        sys.log.event("cup_view_failed", detail_bowl=name, reason=e.reason)
        return False, WHY_UNSEEN.get(e.reason, "it can't make out the rim")
    obs = rim.observe(name)                             # the first look, so the screen shows what the camera says
    sys.log.event("cup_view", detail_bowl=name, status=obs.status, reason=obs.reason,
                  calibrated=rim.calib is not None)
    return True, ""


def confirm_cup(sys, name: str, rim_mm: float, width_mm: float) -> dict:
    """A person has looked: the spoon tip is over the middle of this cup's opening. rim_mm: the cup's height;
    width_mm: the inside width of its opening, both measured with a ruler.

    The arm must still be hovering where "Show aim" (or an arrow) left it. The cup's centre becomes the tip as the
    arm measures it from its own joints right now, sag included, not the point it was told to go to. Then the arm
    goes to rest and the table camera learns the opening (_learn_view)."""
    if sys.fsm.state.value != "IDLE":
        raise SetupProblem("End the session first.")
    try:
        rim_mm, width_mm = float(rim_mm), float(width_mm)
    except (TypeError, ValueError):
        rim_mm = width_mm = float("nan")
    if not (0 < rim_mm <= MAX_CUP_MM and 0 < width_mm <= MAX_CUP_MM):
        raise SetupProblem("Enter the cup's height and the width of its opening, in millimetres.")
    b = _bowl(sys, name)
    kin = sys.cfg.kinematics
    pose = (sys.setup_state.get("aimed") or {}).get(name)
    st = sys.arm.state
    if pose is None or sys.arm.busy or not sys.arm.torque_on or not st.ok:
        raise SetupProblem(NOT_HOVERING)
    tip = K.fk(kin, st.deg)
    if math.dist(tip, K.fk(kin, pose)) > CONFIRM_WITHIN_MM:
        raise SetupProblem(NOT_HOVERING)
    keep = {k: getattr(b.cup, k) for k in ("wall_mm", "floor_mm", "food_below_rim_mm")} if b.cup is not None else {}
    b.cup = CupCfg(x_mm=round(tip[0], 1), y_mm=round(tip[1], 1), radius_mm=width_mm / 2, rim_mm=rim_mm,
                   uncertainty_mm=CONFIRMED_UNCERTAINTY_MM, confirmed=True, source="confirmed", **keep)
    sys.setup_state.setdefault("confirmed_aim", {})[name] = dict(pose)
    blocked = getattr(sys.fsm, "rim_blocked", None)     # a person has lined it up again: the last touch is history
    if isinstance(blocked, dict):
        blocked.pop(name, None)
    sys.cfg.save()
    sys.log.event("cup_confirmed", detail_bowl=name, at_mm=[b.cup.x_mm, b.cup.y_mm], rim_mm=rim_mm,
                  width_mm=width_mm, tip_z_mm=round(tip[2], 1))
    seen, why = _learn_view(sys, name)
    sys.setup_state.setdefault("cups", {})[name] = {"why": why}
    paused = " Scoops from this cup are paused." if sys.cfg.plate.rim.require_view else ""
    message = SEEN if seen else f"Position confirmed, but the camera can't see the opening: {why}.{paused}"
    return {"confirmed": True, "at_mm": [b.cup.x_mm, b.cup.y_mm], "camera": seen, "message": message}


def cup_status(sys, b: BowlCfg) -> dict:
    """What the Bowls tab shows for one bowl's cup, with one plain line that says where it stands."""
    c, rim = b.cup, getattr(sys, "rim", None)
    obs = rim.last.get(b.name) if rim is not None else None
    st = {"has_cup": c is not None, "rim_mm": c.rim_mm if c else None,
          "width_mm": round(2 * c.radius_mm, 1) if c else None, "measured": bool(c and c.rim_mm is not None),
          "confirmed": bool(c and c.confirmed), "uncertainty_mm": c.uncertainty_mm if c else None,
          "source": c.source if c else "", "reference": bool(rim is not None and rim.has_reference(b.name)),
          "seen": obs.status if obs else None, "reason": obs.reason if obs else "",
          # brain/rim_scoop's note that the spoon touched this cup where it now stands
          "blocked": b.name in (getattr(sys.fsm, "rim_blocked", None) or ())}
    require_view = sys.cfg.plate.rim.require_view
    if not st["confirmed"]:
        ok, line = False, "Not confirmed yet: press Show aim, centre the spoon, then confirm."
    elif st["blocked"]:
        ok, line = False, "The spoon touched this cup. Centre the spoon over it and confirm again."
    elif not st["reference"] or st["seen"] == "lost":
        why = WHY_UNSEEN.get(st["reason"], "") if st["reference"] else \
            ((sys.setup_state.get("cups") or {}).get(b.name) or {}).get("why") or WHY_UNSEEN["no_reference"]
        ok = not require_view
        line = ("Camera can't see the opening: scoops from this cup are paused." if require_view
                else "Position confirmed. The camera can't see the opening.") + (f" ({why})" if why else "")
    elif st["seen"] == "moved":
        ok, line = True, "Position confirmed. The cup has moved a little; the spoon will follow it."
    else:
        ok, line = True, SEEN
    return {**st, "ok": ok, "line": line}


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
        "cups": {b.name: cup_status(sys, b) for b in sys.cfg.plate.bowls},
        "spoon": sys.cfg.spoon.model_dump(),
    }
