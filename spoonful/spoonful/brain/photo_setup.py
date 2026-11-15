"""Setup for a new table: the diner's mouth from one touch, the bowls from one overhead photo.

save_mouth   the spoon is at the diner's lips (moved there by hand): save the mouth pose, the waiting pose in front
             of it and the face zone around it.
map_table    one photo from above: find the bowls (photo_map), build a scoop and a lift for each one the arm can
             reach, and let vision choose how to scoop each food.
status       what the Setup screen shows for each step.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from ..arm import kinematics as K
from ..arm.trajectory import plan
from ..config import ARM_JOINTS, BowlCfg
from ..perception.photo_map import PHOTO_TIP_MM, PROMPT, analyze_robust, predicted_surface_mm
from . import quick_setup as QS

MAX_PHOTO_PX = 1600
MIN_TIP_REACH_MM = 80      # the spoon tip must be this far from the base axis for the photo's scale to be reliable
MAX_TIP_HEIGHT_MM = 90     # and this close to the table, so its position in the photo isn't shifted by perspective


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


def _annotate(img, m: dict, used: set[str]):
    import cv2
    out = img.copy()
    for p, col in ((m["base_px"], (133, 114, 11)), (m["tip_px"], (133, 114, 11))):
        cv2.drawMarker(out, (int(p[0]), int(p[1])), col, cv2.MARKER_CROSS, 36, 3)
    for bw in m["bowls"]:
        col = (76, 138, 31) if bw["name"] in used else (0, 106, 176)
        bx = [int(v) for v in bw["box_px"]]
        cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), col, 3)
        label = (bw.get("food") or "food") + ("" if bw["name"] in used else " ?")
        cv2.putText(out, label, (bx[0], max(24, bx[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return out


def map_table(sys, img) -> dict:
    """img: BGR overhead photo. Returns {"bowls": [...], "skipped": [...]} or raises SetupProblem."""
    import cv2
    if sys.fsm.state.value != "IDLE":
        raise SetupProblem("End the session first, then retake the photo.")
    if sys.plate.image_generate is None:
        raise SetupProblem("Vision isn't available. Check the Gemini key.")
    if not (sys.lib.has("pre_delivery") and sys.lib.has("mouth_nominal")):
        raise SetupProblem("Set your mouth position first (step 1).")
    kin = sys.cfg.kinematics
    tip = K.fk(kin, sys.arm.state.deg)
    if math.hypot(tip[0], tip[1]) < MIN_TIP_REACH_MM or tip[2] > MAX_TIP_HEIGHT_MM:
        raise SetupProblem("Press \"Pose for photo\" first, then retake the photo.")

    h, w = img.shape[:2]
    k = MAX_PHOTO_PX / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)))
    try:
        m = analyze_robust(img, lambda data: sys.plate.image_generate(data, PROMPT), tip_mm=(tip[0], tip[1]))
    except Exception as e:
        raise SetupProblem(f"Couldn't read the photo: {e}") from e

    QS.LIMITS = sys.cfg.arm.soft_limits_deg
    pre = sys.lib.pose("pre_delivery")
    ref = sys.lib.pose("rest") if sys.lib.has("rest") else dict(sys.cfg.arm.home_pose)
    moves, bowls, found, skipped = {}, [], [], []
    for bw in m["bowls"]:
        food = bw.get("food") or "food"
        if not bw["name"]:
            skipped.append({"food": food, "why": f"seen in only {bw['votes']} of {bw['views']} views"})
            continue
        x, y = bw["mm"]
        surface = predicted_surface_mm(bw.get("amount"))
        try:
            scoop = QS.build_scoop_at(kin, ref, x, y, surface)
            if any(K.in_face_zone(kin, q) for q in scoop):
                raise QS.SetupError("too close to your face")
            lift = QS.build_lift(kin, scoop[-1], pre)
            plan(sys.cfg, scoop[0], [(scoop, "normal", "check")])
            plan(sys.cfg, scoop[-1], [(lift, "normal", "check")])
        except Exception as e:
            skipped.append({"food": food, "why": "out of reach" if "reach" in str(e) or "limits" in str(e) else str(e)})
            bw["name"] = None
            continue
        moves[f"scoop_{bw['name']}"] = ("path", scoop)
        moves[f"lift_{bw['name']}"] = ("path", lift)
        bx = [int(v) for v in bw["box_px"]]
        crop = img[max(0, bx[1] - 15):bx[3] + 15, max(0, bx[0] - 15):bx[2] + 15]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        bw["crop_jpg"] = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        bowls.append(BowlCfg(name=bw["name"], roi=(0, 0, 1280, 720), label=bw.get("food") or "", watch=False))
        sys.food_surface.est[bw["name"]] = surface
        found.append(bw)
    if not bowls:
        why = "; ".join(f"{s['food']}: {s['why']}" for s in skipped) or "no bowls found"
        raise SetupProblem(f"No bowls the arm can reach ({why}). Move them closer and retake the photo.")

    first = bowls[0].name
    moves["scoop"], moves["lift"] = moves[f"scoop_{first}"], moves[f"lift_{first}"]
    QS.save_all(sys.cfg, sys.lib.dir, {"moves": moves})
    sys.cfg.plate.bowls = bowls
    sys.lib.reload()
    sys.plate.reset_bowls()
    result = []
    for bw in found:
        sys.plate.bowls[bw["name"]].update(amount=bw.get("amount"), vision_texture=bw.get("texture"))
        attack = sys.plate.plan_attack(bw["name"], bw["crop_jpg"])
        next(c for c in sys.cfg.plate.bowls if c.name == bw["name"]).attack = attack
        result.append({"name": bw["name"], "food": bw.get("food"), "amount": bw.get("amount"),
                       "strategy": attack["strategy"], "why": attack["why"], "votes": bw["votes"], "views": bw["views"],
                       "uncertainty_mm": bw["uncertainty_mm"],
                       "at_cm": [round(bw["mm"][0] / 10), round(bw["mm"][1] / 10)]})
    sys.cfg.save()
    cv2.imwrite(str(sys.cfg.path("data/photo_map.jpg")), _annotate(img, m, {b.name for b in bowls}))
    sys.setup_state.update(table_at=time.time(), bowls=result, skipped=skipped)
    sys.log.event("photo_map", detail_bowls=result, skipped=skipped, mm_per_px=round(m["mm_per_px"], 3))
    return {"bowls": result, "skipped": skipped}


def status(sys) -> dict:
    st = sys.setup_state
    kin = sys.cfg.kinematics
    tip = K.fk(kin, sys.arm.state.deg) if sys.arm.state.ok else None
    posed = bool(tip) and math.dist(tip, PHOTO_TIP_MM) < 25
    return {
        "mouth": {"set": sys.lib.has("mouth_nominal") and sys.lib.has("pre_delivery"), "at": st.get("mouth_at"),
                  "released": not sys.arm.torque_on},
        "table": {"at": st.get("table_at"), "bowls": st.get("bowls") or [], "skipped": st.get("skipped") or [],
                  "posed": posed},
        "looking": {"at": st.get("looking_at")},
    }
