"""Self-calibration of the plate camera against the table: the arm calibrates the camera by itself.

The arm carries the spoon tip to a set of spots over the table (lifting between them), holds still, and vision points
to the spoon tip in the camera picture. Forward kinematics says exactly where the tip is, so every spot is a
(pixel, table mm) pair; a homography over them maps any pixel in that region to the table. Afterwards a bowl seen
anywhere in that region (and within reach) can be scooped: the taught scoop is moved there.

Spots are at CAL_Z_MM above the table: about the height of a cup's middle, which is what the bowl detector's box
centre sees. Runs only in IDLE, with the cup removed (the spoon passes low over its spot). Any STOP, state change or
motion error aborts and parks the arm at rest/home.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

from ..arm import kinematics as K
from ..arm.scoop_style import scoop_anchor, solve_rz
from ..arm.trajectory import check_limits

CAL_Z_MM = 50                       # spoon tip this high over the table (never pressed into it)
TRAVEL_Z_MM = 120
MAX_SPOTS = 12                      # stop after this many clearly seen spots
MAX_VISITS = 22
STEP_MM = 70                        # before 4 spots are seen, only step this far from a seen one
FRAME_MARGIN_PX = 90                # only visit spots predicted at least this far inside the picture
EDGE_PX = 40                        # a tip "seen" this close to the border is probably cut off: ignored
SETTLE_S = 0.8


TILTS_DEG = (0, 15, -15, 30, -30, 45, -45, 60)


def pose_at(kin, ref: dict, x: float, y: float, z: float, ok=lambda q: True) -> Optional[dict]:
    """Joint pose with the spoon tip at (x, y, z). The spoon's tilt doesn't matter for calibration, so try the ref
    tilt first, then tilted further; the first one that solves and passes ok(q) wins. None if none does."""
    p = dict(ref)
    p["base"] = kin.zero_deg["base"] + math.degrees(math.atan2(y, x)) / kin.sign["base"]
    for t in TILTS_DEG:
        q = solve_rz(kin, p, math.hypot(x, y), z, iters=25, tilt_deg=t)
        if q is not None and ok(q):
            return q
    return None


class TableAutoCal:
    def __init__(self, system):
        self.s = system
        self.running = False
        self.msg = ""
        self.done_points = 0
        self.total = 0
        self._abort = threading.Event()

    def status(self) -> dict:
        return {"running": self.running, "msg": self.msg, "points": self.done_points, "total": self.total}

    def start(self) -> None:
        if self.running:
            raise RuntimeError("already calibrating")
        if self.s.fsm.state.value != "IDLE":
            raise RuntimeError("only in IDLE (end the session first)")
        if not self.s.cfg.kinematics.spoon_measured:
            raise RuntimeError("measure the spoon first (Calibrate > Spoon length): without it the robot doesn't know "
                               "how low the spoon tip really is")
        self._abort.clear()
        self.running = True
        threading.Thread(target=self._run, name="table-autocal", daemon=True).start()

    def abort(self) -> None:
        self._abort.set()

    # ------------------------------------------------------------------ the routine
    def _candidates(self) -> tuple[list[dict], tuple[float, float]]:
        """Every reachable spot on a fan grid around the arm, plus the cup's taught spot (the seed: the camera has
        seen the cup there). Each: {"mm": (x, y), "high": pose, "low": pose}."""
        cfg, s = self.s.cfg, self.s
        kin = cfg.kinematics
        scoop = s.plate.scoop_for(next(iter(s.plate.bowls))) if s.plate.bowls else "scoop"
        if not s.lib.has(scoop):
            scoop = "scoop"
        if not s.lib.has(scoop):
            raise RuntimeError("teach a scoop first (its spot is where the calibration starts)")
        poses = s.lib.items[scoop].poses
        ref = min(poses, key=lambda q: K.fk(kin, q)[2])          # the dip: spoon angle to start from
        seed = scoop_anchor(kin, poses)

        def ok(q):
            try:
                check_limits(cfg, q)
            except Exception:
                return False
            return not K.in_face_zone(kin, q)
        pts = [seed] + [(r * math.cos(math.radians(yaw)), r * math.sin(math.radians(yaw)))
                        for yaw in range(-90, 125, 15) for r in range(150, 331, 30)]
        out = []
        for x, y in pts:
            low, high = pose_at(kin, ref, x, y, CAL_Z_MM, ok), pose_at(kin, ref, x, y, TRAVEL_Z_MM, ok)
            if low is not None and high is not None:
                out.append({"mm": (x, y), "high": high, "low": low})
        return out, seed

    def _plan_spots(self) -> list[tuple[dict, dict]]:
        """(for tests/inspection) all reachable candidate spots."""
        return [(c["high"], c["low"]) for c in self._candidates()[0]]

    @staticmethod
    def _next(cands: list[dict], seen: list[tuple], misses: list[tuple], seed) -> Optional[dict]:
        """Pick the next spot so the spoon stays in the picture.

        < 4 good spots: step out from what was seen (nearest to the seen spots, never near a miss).
        >= 4: predict each spot's pixel from the seen ones; keep only those well inside the frame; take the one
        farthest (in the picture) from everything already seen, to spread out."""
        import numpy as np
        if not cands:
            return None
        if not seen:
            return min(cands, key=lambda c: math.dist(c["mm"], seed))
        good_mm = [m for m, _ in seen]

        def near_miss(c):
            return any(math.dist(c["mm"], m) < 45 for m in misses)
        pool = [c for c in cands if not near_miss(c)]
        if len(seen) < 4:
            pool = [c for c in pool if min(math.dist(c["mm"], m) for m in good_mm) <= STEP_MM]
            return min(pool, key=lambda c: min(math.dist(c["mm"], m) for m in good_mm)) if pool else None
        import cv2
        H, _ = cv2.findHomography(np.array(good_mm, np.float64), np.array([p for _, p in seen], np.float64), 0)
        if H is None:
            return None
        best, best_d = None, -1.0
        for c in pool:
            v = H @ np.array([c["mm"][0], c["mm"][1], 1.0])
            if v[2] <= 0:
                continue
            u, w = v[0] / v[2], v[1] / v[2]
            if not (FRAME_MARGIN_PX <= u <= 1280 - FRAME_MARGIN_PX and FRAME_MARGIN_PX <= w <= 720 - FRAME_MARGIN_PX):
                continue                                           # would leave (or hug the edge of) the picture
            d = min(math.dist((u, w), p) for _, p in seen)
            if d > best_d:
                best, best_d = c, d
        return best if best_d >= 25 else None                       # nothing new left to see

    def _snapshot(self, n: int, px, tip, note: str = "") -> None:
        """data/autocal/spot_N.jpg: what the camera saw, with vision's answer marked (for checking a bad fit)."""
        try:
            import cv2
            img = getattr(self.s.plate, "last_point_frame", None)
            if img is None:
                return
            img = img.copy()
            if px:
                cv2.drawMarker(img, (int(px[0]), int(px[1])), (255, 0, 255), cv2.MARKER_CROSS, 40, 3)
            cv2.putText(img, f"spot {n}: tip at ({tip[0]:.0f}, {tip[1]:.0f}) mm  vision: {px and [round(v) for v in px]} {note}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            d = self.s.cfg.path(self.s.cfg.plate.track.calib_file).parent / "autocal"   # next to the calibration
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"spot_{n:02d}.jpg"), img)
        except Exception:
            pass

    def _move(self, pose: dict, name: str) -> None:
        if self._abort.is_set() or self.s.fsm.state.value != "IDLE":
            raise RuntimeError("stopped")
        m = self.s.arm.goto(pose, "normal", 0.5, name=name)
        if not m.wait(20).ok:
            raise RuntimeError(f"move failed: {m.error!r}")

    def _run(self) -> None:
        s = self.s
        try:
            cands, seed = self._candidates()
            if len(cands) < 4:
                raise RuntimeError(f"only {len(cands)} reachable spots")
            s.plate.table.clear()
            s.speaker.say_text("Calibrating the table. Keep hands clear")
            seen: list[tuple] = []            # (mm, px) with the tip clearly in the picture
            misses: list[tuple] = []
            self.total, self.done_points = MAX_SPOTS, 0
            n = 0
            while n < MAX_VISITS and len(seen) < MAX_SPOTS:
                c = self._next(cands, seen, misses, seed)
                if c is None:
                    break
                cands.remove(c)
                n += 1
                self.msg = f"spot {n} ({len(seen)} seen so far)"
                self._move(c["high"], "autocal_up")
                self._move(c["low"], "autocal_spot")
                time.sleep(SETTLE_S)
                tip = K.fk(s.cfg.kinematics, s.arm.state.deg)          # measured, sag included
                px = s.plate.point_spoon_tip()
                edge = px is not None and not (EDGE_PX <= px[0] <= 1280 - EDGE_PX and EDGE_PX <= px[1] <= 720 - EDGE_PX)
                high = px is not None and px[1] < s.cfg.plate.track.table_min_v_px
                self._snapshot(n, px, tip, "at the edge: ignored" if edge else ("above the table: ignored" if high else ""))
                edge = edge or high
                if px is not None and not edge:
                    seen.append(((tip[0], tip[1]), px))
                    s.plate.table.add(px, (tip[0], tip[1]))
                    self.done_points = len(seen)
                else:
                    misses.append((tip[0], tip[1]))
                s.log.event("table_autocal_spot", detail_spot=n, px=[round(v) for v in px] if px else None,
                            mm=[round(tip[0]), round(tip[1])], used=bool(px is not None and not edge))
                self._move(c["high"], "autocal_up")
            t = s.plate.table
            st = t.status()
            self.msg = (f"done: {st['inliers']} of {st['points']} spots agree, fit error {t.rms_mm} mm "
                        f"({n} visited)" if t.ready
                        else f"only {len(seen)} spots clearly seen ({n} visited): see the data/autocal pictures")
            s.speaker.say_text("Table calibrated" if t.ready else "Calibration failed")
        except Exception as e:
            self.msg = f"stopped: {e}"
            s.speaker.say_text("Calibration stopped")
        finally:
            try:
                if s.fsm.state.value == "IDLE":
                    home = s.lib.pose("rest") if s.lib.has("rest") else s.cfg.arm.home_pose
                    s.arm.goto(dict(home), "normal", 0.5, name="autocal_done").wait(20)
            except Exception:
                pass
            s.log.event("table_autocal", detail_msg=self.msg)
            self.running = False
