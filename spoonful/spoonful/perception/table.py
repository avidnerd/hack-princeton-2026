"""Plate camera picture -> table position (mm, arm base frame), so the arm can follow a bowl that was moved.

Calibration uses the arm itself as the ruler: with torque off, touch the spoon tip to a spot on the table, then click
that spot in the plate camera picture. The tip's table position comes from forward kinematics, the click gives the
pixel. Four or more spread-out spots fit a homography (the table is a plane), saved to data/table_calib.json.

Pixels are in 1280x720 reference coordinates, like the bowl boxes in config.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

MIN_POINTS = 4
MIN_AGREE = 8          # a homography has 8 unknowns: with 5-6 points it can 'fit' wrong answers (2026-11-14)
RANSAC_MM = 20


class TableCalib:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.points: list[dict] = []          # {"px": [u, v], "mm": [x, y]}
        self.H: Optional[np.ndarray] = None   # pixel -> mm
        self.rms_mm: Optional[float] = None
        self.inliers = 0
        self.load()

    # ------------------------------------------------------------------ persistence
    def load(self) -> None:
        try:
            d = json.loads(self.path.read_text())
            self.points = d.get("points", [])
        except (OSError, ValueError):
            self.points = []
        self.fit()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"points": self.points, "rms_mm": self.rms_mm}, indent=1))

    # ------------------------------------------------------------------ calibration
    def add(self, px: tuple[float, float], mm: tuple[float, float]) -> None:
        self.points.append({"px": [round(px[0], 1), round(px[1], 1)], "mm": [round(mm[0], 1), round(mm[1], 1)]})
        self.fit()
        self.save()

    def clear(self) -> None:
        self.points = []
        self.fit()
        self.save()

    def fit(self) -> None:
        self.H, self.rms_mm, self.inliers = None, None, 0
        if len(self.points) < MIN_POINTS:
            return
        import cv2
        src = np.array([p["px"] for p in self.points], np.float64)
        dst = np.array([p["mm"] for p in self.points], np.float64)
        # RANSAC: a vision answer that pointed at the wrong thing must not drag the whole fit (2026-11-14: 10 of 15)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_MM)
        if H is None:
            return
        inl = [bool(m) for m in mask.ravel()]
        for p, ok in zip(self.points, inl):
            p["inlier"] = ok
        self.inliers = sum(inl)
        if self.inliers < MIN_AGREE:
            return                       # too few to tell a real fit from one that bends to fit wrong answers
        err = [math.dist(self._apply(H, p["px"]), p["mm"]) for p, ok in zip(self.points, inl) if ok]
        self.H = H
        self.rms_mm = round(float(np.sqrt(np.mean(np.square(err)))), 1)

    @staticmethod
    def _apply(H: np.ndarray, px) -> tuple[float, float]:
        v = H @ np.array([px[0], px[1], 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    # ------------------------------------------------------------------ use
    @property
    def ready(self) -> bool:
        return self.H is not None

    def to_table(self, px: tuple[float, float]) -> Optional[tuple[float, float]]:
        return self._apply(self.H, px) if self.H is not None else None

    def status(self) -> dict:
        return {"points": len(self.points), "needed": MIN_POINTS, "ready": self.ready, "rms_mm": self.rms_mm,
                "inliers": self.inliers,
                "list": self.points}
