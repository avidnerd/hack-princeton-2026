"""Camera aim check: before trusting a scoop, look and see whether the spoon is really over the bowl.

The table photo places a bowl to within a centimetre or two; a small cup needs better. So for each bowl:

  1. Hover the spoon above where it will dip.
  2. Learn how the picture responds to the arm (an image Jacobian, measured, not assumed): move the tip 20 mm
     further, 20 mm left and 20 mm up, and see how far its pixel moves each time. No camera calibration is needed.
  3. Find the bowl in the same picture: the centre of its base, where it rests on the table.
  4. The spoon tip is at a known height. Subtracting that height's worth of pixel shift gives the spot on the table
     directly under the tip. Its distance from the bowl's base, turned back into millimetres with the Jacobian, is
     how far the scoop is off.
  5. Move the scoop by that much (photo_setup.nudge) and look again to confirm.

A side-on webcam sees left-right well and depth less well, so corrections are capped, a correction that doesn't
reduce the error is undone, and nothing is changed when the camera can't see the spoon or the bowl clearly.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

from ..arm import kinematics as K
from . import photo_setup
from . import quick_setup as QS

PROBE_MM = 20
MAX_FIX_MM = 35            # larger than this: the camera and the photo disagree badly; ask for a new photo
GOOD_MM = 7                # within this the aim is confirmed
MIN_SHIFT_PX = 4           # a 20 mm move must show up at least this much in the picture
SETTLE_S = 0.7

CUP_PROMPT = (
    "In this picture a small robot arm holds a spoon near a cup or bowl of {food}. Point to that container. Answer "
    "only JSON {{\"base\": [y, x], \"opening\": [y, x]}}: \"base\" is the centre of its bottom, where it rests on the "
    "table; \"opening\" is the centre of its open top. Coordinates normalised to 0-1000. Use null for a point you "
    "cannot see clearly."
)


class AimProblem(RuntimeError):
    """The check could not be done; the message says why. The scoop is left as it was."""


def solve_offset(tip_px, base_px, tip_height_mm: float, j_further, j_left, j_up) -> tuple[float, float]:
    """Pure. (further_mm, left_mm) to move the tip so it is directly above the bowl's base.
    j_*: pixels the tip moves per mm of motion in that direction (2-vectors)."""
    under_tip = np.asarray(tip_px, float) - np.asarray(j_up, float) * tip_height_mm     # the table point below the tip
    J = np.column_stack([j_further, j_left]).astype(float)
    if abs(np.linalg.det(J)) < 1e-3 or np.linalg.cond(J) > 25:
        raise AimProblem("From this angle the webcam can't tell which way the spoon needs to move.")
    d = np.linalg.solve(J, np.asarray(base_px, float) - under_tip)
    return float(d[0]), float(d[1])


class AimCheck:
    def __init__(self, system):
        self.s = system
        self.running: Optional[str] = None
        self.results: dict[str, dict] = {}
        self._abort = threading.Event()

    def status(self) -> dict:
        return {"running": self.running, "results": self.results}

    def start(self, bowl: str) -> None:
        if self.running:
            raise photo_setup.SetupProblem("Already checking a bowl.")
        if self.s.fsm.state.value != "IDLE":
            raise photo_setup.SetupProblem("End the session first.")
        if self.s.plate._point is None or self.s.plate.image_generate is None:
            raise photo_setup.SetupProblem("Vision isn't available. Check the Gemini key.")
        self._abort.clear()
        self.running = bowl
        self.results[bowl] = {"state": "checking", "message": "Looking at the spoon"}
        threading.Thread(target=self._run, args=(bowl,), name="aim-check", daemon=True).start()

    # ------------------------------------------------------------------ pieces
    def _goto(self, pose: dict) -> None:
        if self._abort.is_set() or self.s.fsm.state.value != "IDLE":
            raise AimProblem("Stopped.")
        m = self.s.arm.goto(pose, "normal", 0.5, name="aim_check")
        if not m.wait(15).ok:
            raise AimProblem("The arm couldn't make a test move.")
        time.sleep(SETTLE_S)

    def _tip_px(self) -> np.ndarray:
        for _ in range(2):
            p = self.s.plate.point_spoon_tip()
            if p is not None and 20 <= p[0] <= 1260 and 20 <= p[1] <= 700:
                return np.array(p, float)
        raise AimProblem("The webcam can't see the spoon tip here.")

    def _bowl_px(self, food: str) -> np.ndarray:
        import cv2
        img = self.s.plate.frame_fn()
        if img is None:
            raise AimProblem("No webcam picture.")
        jpg = cv2.imencode(".jpg", cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        d = self.s.plate.image_generate(jpg, CUP_PROMPT.format(food=food or "food"))
        if isinstance(d, list) and d:
            d = d[0]
        b = d.get("base") if isinstance(d, dict) else None
        if not (isinstance(b, (list, tuple)) and len(b) == 2):
            raise AimProblem(f"The webcam can't see the {food or 'bowl'} clearly.")
        return np.array([float(b[1]) * 1.28, float(b[0]) * 0.72])

    def _measure(self, bowl: str, food: str) -> tuple[float, float]:
        """Hover, probe, look: how far (further_mm, left_mm) the spoon is from being over the bowl."""
        s, kin = self.s, self.s.cfg.kinematics
        dip = photo_setup._dip(s, bowl)
        x, y, z = K.fk(kin, dip)
        r, yaw = math.hypot(x, y), math.atan2(y, x)
        QS.LIMITS = s.cfg.arm.soft_limits_deg

        def pose(dr=0.0, dl=0.0, dz=0.0) -> dict:
            px, py = x + dr * x / r - dl * y / r, y + dr * y / r + dl * x / r
            q = dict(dip)
            q["base"] = kin.zero_deg["base"] + math.degrees(math.atan2(py, px)) / kin.sign["base"]
            return QS._at(kin, q, math.hypot(px, py), z + photo_setup.AIM_ABOVE_MM + dz)
        try:
            poses = [pose(), pose(dr=PROBE_MM), pose(dl=PROBE_MM), pose(dz=PROBE_MM)]
        except QS.SetupError as e:
            raise AimProblem("The arm can't make its test moves around that bowl.") from e
        seen, tips = [], []
        for q in poses:
            self._goto(q)
            seen.append(self._tip_px())
            tips.append(np.array(K.fk(kin, s.arm.state.deg)))          # where the arm really is (sag included)
        # pixels per mm, from where the tip really went
        cols = []
        for i in (1, 2, 3):
            if np.linalg.norm(seen[i] - seen[0]) < MIN_SHIFT_PX and i != 3:
                raise AimProblem("The spoon barely moves in the webcam picture. Turn the laptop to see it from the side.")
            cols.append(seen[i] - seen[0])
        moved = np.array([tips[i] - tips[0] for i in (1, 2, 3)])        # rows: actual (dx, dy, dz) of each probe
        out_dir = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left_dir = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        basis = np.column_stack([moved @ out_dir, moved @ left_dir, moved[:, 2]])   # each probe in (further, left, up)
        try:
            Jt = np.linalg.solve(basis, np.array(cols))                 # rows: px per mm of further / left / up
        except np.linalg.LinAlgError as e:
            raise AimProblem("The test moves were too small to measure.") from e
        self._goto(poses[0])
        tip = self._tip_px()
        height = float(K.fk(kin, s.arm.state.deg)[2])
        return solve_offset(tip, self._bowl_px(food), height, Jt[0], Jt[1], Jt[2])

    def _run(self, bowl: str) -> None:
        s = self.s
        food = s.plate.food_of(bowl) or "bowl"
        res = self.results[bowl]
        try:
            f, l = self._measure(bowl, food)
            off = math.hypot(f, l)
            s.log.event("aim_check", detail_bowl=bowl, further_mm=round(f, 1), left_mm=round(l, 1))
            if off <= GOOD_MM:
                res.update(state="ok", message=f"Lined up (within {off:.0f} mm)")
            elif off > MAX_FIX_MM:
                res.update(state="problem", message=f"The webcam sees the spoon {off / 10:.0f} cm from the {food}. "
                                                    "Retake the table photo.")
            else:
                res.update(message=f"Off by {off:.0f} mm. Correcting")
                photo_setup.nudge(s, bowl, f, l)
                time.sleep(SETTLE_S)
                f2, l2 = self._measure(bowl, food)
                off2 = math.hypot(f2, l2)
                s.log.event("aim_check", detail_bowl=bowl, after_fix=True, further_mm=round(f2, 1), left_mm=round(l2, 1))
                if off2 < off - 2:
                    res.update(state="ok" if off2 <= 2 * GOOD_MM else "problem",
                               message=f"Corrected by {off:.0f} mm; now within {off2:.0f} mm")
                else:                                       # the correction didn't help: the camera was wrong, undo
                    photo_setup.nudge(s, bowl, -f, -l)
                    res.update(state="problem", message="Couldn't confirm the aim from this camera angle. "
                                                        "Use Check aim and the nudge buttons.")
        except (AimProblem, photo_setup.SetupProblem) as e:
            res.update(state="problem", message=str(e))
        except Exception as e:
            s.log.event("aim_check_failed", detail_error=repr(e)[:200])
            res.update(state="problem", message="The check failed. Use Check aim and the nudge buttons.")
        finally:
            self.running = None
            try:
                if s.fsm.state.value == "IDLE":
                    photo_setup.aim(s, bowl)                # finish hovering over the bowl, for the eye to confirm
            except Exception:
                pass
