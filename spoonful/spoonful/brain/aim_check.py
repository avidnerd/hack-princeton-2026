"""Camera aim check: look and see whether the spoon is really over the bowl, at setup and before every scoop.

The table photo places a bowl to within a centimetre or two, and a bowl can be moved after the photo. So the webcam
has the last word on where the spoon dips.

Once per bowl (calibrate, about 30 s):
  Hover the spoon above the bowl and move the tip 20 mm further, 20 mm left and 20 mm up, noting how far its pixel
  moves each time. That is the image Jacobian, measured rather than assumed: no camera calibration is needed. It is
  kept only while the app runs, because it is wrong as soon as the laptop is moved.

Every look (about 2 s, one webcam frame):
  Find the spoon tip, and a box around every dish. (Boxes come back tight and repeatable; asking for a single
  point on a dish does not.) The dish is the one whose food matches, or failing that the one the spoon is hovering
  over; the centre of its base is taken from the bottom of its box. The tip is at a known height;
  subtracting that height's worth of pixel shift gives the table point directly under the tip. Its distance from
  the bowl's base, turned back into millimetres with the Jacobian, is how far the scoop is off.

The check runs by itself on the first scoop from each bowl (or from the Setup screen): the scoop is corrected and
checked again. Before every later scoop the FSM pauses above the bowl, looks, and moves the scoop if the bowl isn't
where it was (brain/fsm.py). A side-on webcam sees left-right well and depth less
well, so corrections are capped, a setup correction that doesn't reduce the error is undone, and nothing changes
when the camera can't see the spoon or the bowl clearly.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from ..arm import kinematics as K
from . import photo_setup
from . import quick_setup as QS

PROBE_MM = 20
MAX_FIX_MM = 35            # setup: more than this means the camera and the photo disagree badly
MAX_FOLLOW_MM = 60         # before a scoop: follow a bowl that moved up to this far
GOOD_MM = 7                # within this the aim is confirmed
MIN_SHIFT_PX = 4           # a 20 mm move must show up at least this much in the picture
SETTLE_S = 0.7

DISHES_PROMPT = (
    "Find every cup, bowl or tray of food on the table in this picture. For each one return {\"food\": 1-3 lowercase "
    "words, \"box_2d\": [ymin, xmin, ymax, xmax]} with a tight box around the whole container, coordinates normalised "
    "to 0-1000. Do not include the robot or the spoon. Answer only a JSON list."
)
BASE_UP = 0.08             # the centre of a dish's base sits about this fraction of its box height above the box bottom


class AimProblem(RuntimeError):
    """The check could not be done; the message says why. The scoop is left as it was."""


WEAK_AXIS = 0.2            # a table direction the camera sees this much less strongly than the other is not trusted


def solve_offset(tip_px, base_px, tip_height_mm: float, j_further, j_left, j_up) -> tuple[float, float]:
    """Pure. (further_mm, left_mm) to move the tip so it is directly above the bowl's base.
    j_*: pixels the tip moves per mm of motion in that direction (2-vectors).

    A camera near table height barely sees motion toward or away from itself: a few pixels of error there would
    turn into centimetres. When one table direction is that weak, only the direction the camera sees well is
    corrected, and the other is left where the table photo put it."""
    under_tip = np.asarray(tip_px, float) - np.asarray(j_up, float) * tip_height_mm     # the table point below the tip
    J = np.column_stack([j_further, j_left]).astype(float)
    err = np.asarray(base_px, float) - under_tip
    U, S, Vt = np.linalg.svd(J)
    if S[0] < 0.05:
        raise AimProblem("The spoon barely moves in the camera's picture.")
    if S[1] < WEAK_AXIS * S[0]:
        d = Vt[0] * (U[:, 0] @ err) / S[0]
    else:
        d = np.linalg.solve(J, err)
    return float(d[0]), float(d[1])


class AimCheck:
    def __init__(self, system):
        self.s = system
        self.running: Optional[str] = None
        self.results: dict[str, dict] = {}
        self.jacobians: dict[str, np.ndarray] = {}      # bowl -> 3x2: px per mm of further / left / up
        self._abort = threading.Event()
        self._pending: Optional[dict] = None            # result of the look started by the FSM
        self._states = {"IDLE"}                         # FSM states in which the check may move the arm
        self.failed: set[str] = set()                   # bowls this camera view could not be confirmed on

    # ------------------------------------------------------------------ what was learned about the camera
    def _jacobian(self, bowl: str) -> Optional[np.ndarray]:
        return self.jacobians.get(bowl)

    def forget(self) -> None:
        """The table was mapped again (the laptop or the bowls may have moved)."""
        self.jacobians = {}
        self.failed = set()
        if not self.running:
            self.results = {}

    def status(self) -> dict:
        return {"running": self.running, "results": self.results, "calibrated": sorted(self.jacobians)}

    def can_calibrate(self, bowl: str) -> bool:
        """Could the camera be checked on this bowl now? (It hasn't been yet, and there is a picture and vision.)"""
        return (bowl not in self.jacobians and bowl not in self.failed and not self.running
                and self.s.plate._point is not None and self.s.plate.image_generate is not None
                and self.s.plate.frame_fn() is not None)

    def calibrate_async(self, bowl: str) -> None:
        """The first scoop from a bowl: do the whole setup check now, with the arm already on its way there."""
        self._pending = pending = {"bowl": bowl, "done": False}
        self._abort.clear()
        self.running = bowl
        self._states = {"SCOOPING"}

        def run():
            try:
                self._check(bowl)
            except (AimProblem, photo_setup.SetupProblem) as e:
                self.jacobians.pop(bowl, None)
                self.results.setdefault(bowl, {}).update(state="problem", message=str(e))
            except Exception as e:
                self.jacobians.pop(bowl, None)
                self.s.log.event("aim_check_failed", detail_error=repr(e)[:200])
                self.results.setdefault(bowl, {}).update(state="problem", message="The camera check failed.")
            finally:
                if bowl not in self.jacobians:
                    self.failed.add(bowl)
                self.running = None
                self._states = {"IDLE"}
                pending["done"] = True
        threading.Thread(target=run, name="aim-first", daemon=True).start()

    def ready(self, bowl: str) -> bool:
        return (self._jacobian(bowl) is not None and self.s.plate._point is not None
                and self.s.plate.image_generate is not None)

    # ------------------------------------------------------------------ looking
    def aim_pose(self, bowl: str, dr: float = 0.0, dl: float = 0.0, dz: float = 0.0) -> dict:
        kin = self.s.cfg.kinematics
        dip = photo_setup._dip(self.s, bowl)
        x, y, z = K.fk(kin, dip)
        r = math.hypot(x, y)
        px, py = x + dr * x / r - dl * y / r, y + dr * y / r + dl * x / r
        q = dict(dip)
        q["base"] = kin.zero_deg["base"] + math.degrees(math.atan2(py, px)) / kin.sign["base"]
        QS.LIMITS = self.s.cfg.arm.soft_limits_deg
        return QS._at(kin, q, math.hypot(px, py), z + photo_setup.AIM_ABOVE_MM + dz)

    def _tip_px(self) -> np.ndarray:
        if self.s.plate.frame_fn() is None:
            raise AimProblem("No picture of the table. Is the phone still sending?")
        for _ in range(2):
            p = self.s.plate.point_spoon_tip()
            if p is not None and 20 <= p[0] <= 1260 and 20 <= p[1] <= 700:
                return np.array(p, float)
        raise AimProblem("The webcam can't see the spoon tip here.")

    def _dishes(self) -> list[dict]:
        """Every dish the camera sees: [{"food", "base": px of the centre of its base, "x": px centre}]."""
        import cv2
        img = self.s.plate.frame_fn()
        if img is None:
            raise AimProblem("No camera picture of the table.")
        jpg = cv2.imencode(".jpg", cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        d = self.s.plate.image_generate(jpg, DISHES_PROMPT)
        if isinstance(d, dict):
            d = d.get("dishes") or d.get("items") or [d]
        out = []
        for b in d if isinstance(d, list) else []:
            box = b.get("box_2d") if isinstance(b, dict) else None
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            y0, x0, y1, x1 = (float(v) for v in box)
            x = (x0 + x1) / 2 * 1.28
            out.append({"food": str(b.get("food") or ""), "x": x,
                        "base": np.array([x, (y1 - BASE_UP * (y1 - y0)) * 0.72])})
        return out

    @staticmethod
    def pick_dish(dishes: list[dict], food: str, tip_px) -> Optional[dict]:
        """Pure. The dish for this food: the only one whose name matches, else the one the spoon is hovering over."""
        from ..perception.plate import match_food
        named = [d for d in dishes if food and match_food(d["food"], {"it": food})]
        if len(named) == 1:
            return named[0]
        pool = named or dishes
        return min(pool, key=lambda d: abs(d["x"] - tip_px[0])) if pool else None

    def look(self, bowl: str) -> tuple[float, float]:
        """With the spoon hovering over the bowl: (further_mm, left_mm) from the spoon to the bowl, from one glance."""
        J = self._jacobian(bowl)
        if J is None:
            raise AimProblem("The camera hasn't been checked for this bowl yet.")
        food = self.s.plate.food_of(bowl) or "bowl"
        with ThreadPoolExecutor(max_workers=2) as ex:                    # the two questions at the same time
            tip_f, dishes_f = ex.submit(self._tip_px), ex.submit(self._dishes)
            tip, dishes = tip_f.result(), dishes_f.result()
        dish = self.pick_dish(dishes, food, tip)
        if dish is None:
            raise AimProblem(f"The camera can't see the {food}.")
        base = dish["base"]
        height = float(K.fk(self.s.cfg.kinematics, self.s.arm.state.deg)[2])
        return solve_offset(tip, base, height, J[0], J[1], J[2])

    # ------------------------------------------------------------------ before a scoop (called by the FSM)
    def look_async(self, bowl: str) -> None:
        self._pending = pending = {"bowl": bowl, "done": False}

        def run():
            try:
                pending["offset"] = self.look(bowl)
            except Exception as e:
                pending["error"] = str(e)
            pending["done"] = True
        threading.Thread(target=run, name="aim-look", daemon=True).start()

    def take_result(self) -> Optional[dict]:
        p = self._pending
        if p is None or not p["done"]:
            return None
        self._pending = None
        return p

    def follow(self, bowl: str, further_mm: float, left_mm: float) -> bool:
        """Move the scoop to where the camera sees the bowl. False if the spoon can't scoop there."""
        try:
            photo_setup.move_scoop(self.s, bowl, further_mm, left_mm)
            return True
        except photo_setup.SetupProblem:
            return False

    # ------------------------------------------------------------------ at setup
    def start(self, bowls: list[str]) -> None:
        if self.running:
            raise photo_setup.SetupProblem("Already checking a bowl.")
        if self.s.fsm.state.value != "IDLE":
            raise photo_setup.SetupProblem("End the session first.")
        if self.s.plate._point is None or self.s.plate.image_generate is None:
            raise photo_setup.SetupProblem("Vision isn't available. Check the Gemini key.")
        if not bowls:
            raise photo_setup.SetupProblem("Choose the bowls first.")
        self._abort.clear()
        self.running = bowls[0]
        for b in bowls:
            self.results[b] = {"state": "checking", "message": "Waiting"}
        threading.Thread(target=self._run_all, args=(list(bowls),), name="aim-check", daemon=True).start()

    def _goto(self, pose: dict) -> None:
        if self._abort.is_set() or self.s.fsm.state.value not in self._states:
            raise AimProblem("Stopped.")
        m = self.s.arm.goto(pose, "normal", 0.5, name="aim_check")
        if not m.wait(15).ok:
            raise AimProblem("The arm couldn't make a test move.")
        time.sleep(SETTLE_S)

    def calibrate(self, bowl: str) -> None:
        """Three small test moves above the bowl: how arm motion shows up in the webcam picture. Ends hovering."""
        s, kin = self.s, self.s.cfg.kinematics
        try:
            poses = [self.aim_pose(bowl), self.aim_pose(bowl, dr=PROBE_MM), self.aim_pose(bowl, dl=PROBE_MM),
                     self.aim_pose(bowl, dz=PROBE_MM)]
        except QS.SetupError as e:
            raise AimProblem("The arm can't make its test moves around that bowl.") from e
        seen, tips = [], []
        for q in poses:
            self._goto(q)
            seen.append(self._tip_px())
            tips.append(np.array(K.fk(kin, s.arm.state.deg)))          # where the arm really is (sag included)
        if min(np.linalg.norm(seen[i] - seen[0]) for i in (1, 2)) < MIN_SHIFT_PX:
            raise AimProblem("The spoon barely moves in the webcam picture. Turn the laptop to see it from the side.")
        x, y, _ = tips[0]
        yaw = math.atan2(y, x)
        out_dir = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left_dir = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        moved = np.array([tips[i] - tips[0] for i in (1, 2, 3)])        # what each test move really was, in mm
        basis = np.column_stack([moved @ out_dir, moved @ left_dir, moved[:, 2]])
        try:
            self.jacobians[bowl] = np.linalg.solve(basis, np.array([seen[i] - seen[0] for i in (1, 2, 3)]))
        except np.linalg.LinAlgError as e:
            raise AimProblem("The test moves were too small to measure.") from e
        self._goto(poses[0])

    def _check(self, bowl: str) -> None:
        s = self.s
        res = self.results.setdefault(bowl, {})
        res.update(state="checking", message="Looking at the spoon")
        self.calibrate(bowl)
        f, l = self.look(bowl)
        off = math.hypot(f, l)
        s.log.event("aim_check", detail_bowl=bowl, further_mm=round(f, 1), left_mm=round(l, 1))
        if off <= GOOD_MM:
            res.update(state="ok", message=f"Lined up (within {off:.0f} mm)")
            return
        if off > MAX_FIX_MM:
            del self.jacobians[bowl]                    # not believable: this camera view may not steer this bowl
            res.update(state="problem", message=f"The camera puts the spoon {off / 10:.0f} cm from the bowl, which "
                                                "can't be right. Scooping where the photo put it.")
            return
        res.update(message=f"Off by {off:.0f} mm. Correcting")
        photo_setup.move_scoop(s, bowl, f, l)
        self._goto(self.aim_pose(bowl))
        f2, l2 = self.look(bowl)
        off2 = math.hypot(f2, l2)
        s.log.event("aim_check", detail_bowl=bowl, after_fix=True, further_mm=round(f2, 1), left_mm=round(l2, 1))
        if off2 < off - 2:
            res.update(state="ok", message=f"Corrected by {off:.0f} mm; now within {off2:.0f} mm")
        else:                                           # the correction didn't help: the camera was wrong, undo it
            photo_setup.move_scoop(s, bowl, -f, -l)
            del self.jacobians[bowl]
            res.update(state="problem", message="Couldn't confirm the aim from this camera angle. "
                                                "Use Show aim and the arrows.")

    def _run_all(self, bowls: list[str]) -> None:
        s = self.s
        try:
            for bowl in bowls:
                self.running = bowl
                try:
                    self._check(bowl)
                except (AimProblem, photo_setup.SetupProblem) as e:
                    self.jacobians.pop(bowl, None)
                    self.results.setdefault(bowl, {}).update(state="problem", message=str(e))
                except Exception as e:
                    self.jacobians.pop(bowl, None)
                    s.log.event("aim_check_failed", detail_error=repr(e)[:200])
                    self.results.setdefault(bowl, {}).update(state="problem",
                                                             message="The check failed. Use Show aim and the arrows.")
                if bowl not in self.jacobians:
                    self.failed.add(bowl)
                if self._abort.is_set() or s.fsm.state.value != "IDLE":
                    break
        finally:
            self.running = None
            try:
                if s.fsm.state.value == "IDLE":
                    rest = s.lib.pose("rest") if s.lib.has("rest") else dict(s.cfg.arm.home_pose)
                    s.arm.goto(dict(rest), "normal", 0.6, name="aim_done")
            except Exception:
                pass
