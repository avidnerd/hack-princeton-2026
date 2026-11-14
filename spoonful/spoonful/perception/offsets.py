"""Adaptive delivery: where the mouth is (image position + depth) -> joint-space correction added to mouth_nominal.

Calibration (operator drawer or `python -m spoonful calibrate`): put the spoon tip at nominal and ~30 mm
left / right / up / down, and click the tip on the camera image at each. We fit the linear map
    dq = M^T . dpx      (least squares over the offset samples, M is 2 x n_joints)
and px_per_mm from the mean pixel distance of the 30 mm offsets. Optional depth samples: 'away' (tip moved
30 mm straight away from the camera, so it barely moves in the image) and 'toward'. They give the joint
change per mm of depth, J. The camera geometry follows: Z_nominal = focal_px / px_per_mm (camera to tip at
nominal), with focal_px from a tape-measured Z_nominal if given, else from the camera's field of view. An error
in focal_px scales the depth correction by the same factor; it does not bias it.

Runtime, all in 1280-wide image coords:
  z      = face.depth_mm - eyes_behind_lips_mm          (DepthEstimator in face.py)
  lat_mm = (mouth_px - nominal_px) * z / focal_px        clamped to max_correction_mm
  dz_mm  = z - Z_nominal                                  deadband, clamped to max_depth_correction_mm
  dq     = M^T (lat_mm * px_per_mm) + J dz_mm             rate-limited to rate_limit_mm_s in mm
Without depth (no iris visible, old map), lat_mm falls back to the face-width ratio and dz_mm = 0.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from .face import REF_W, FaceState

LABELS = ("nominal", "left", "right", "up", "down", "away", "toward")
DEPTH_LABELS = ("away", "toward")
CAL_OFFSET_MM = 30.0


@dataclass
class OffsetMap:
    joints: list[str]
    nominal_px: tuple[float, float]
    nominal_pose: dict[str, float]
    M: list[list[float]]                # 2 x len(joints): row 0 = d(joint)/d(px_x), row 1 = d(joint)/d(px_y)
    px_per_mm: float
    face_width_px_nominal: float = 220.0
    samples: list[dict] = field(default_factory=list)
    created: str = ""
    residual_deg: float = 0.0
    focal_px: float = 0.0                           # 1280-wide coords
    nominal_depth_mm: float = 0.0                   # camera to spoon tip at mouth_nominal
    depth_joint_per_mm: Optional[list[float]] = None   # J: joint deg per mm of depth (+ = away from camera)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_depth(self) -> bool:
        return bool(self.depth_joint_per_mm) and self.focal_px > 0 and self.nominal_depth_mm > 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: Path) -> Optional["OffsetMap"]:
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        d["nominal_px"] = tuple(d["nominal_px"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def joint_delta(self, dpx: tuple[float, float]) -> dict[str, float]:
        v = np.asarray(dpx) @ np.asarray(self.M)
        return {j: float(v[k]) for k, j in enumerate(self.joints)}


def fit(samples: list[dict], joints: list[str], face_width_px: float = 220.0, hfov_focal_px: float = 914.0,
        measured_depth_mm: Optional[float] = None) -> OffsetMap:
    """samples: [{label, px: [u, v] (1280-wide), pose: {joint: deg}}]: 'nominal' plus >= 2 sideways/up/down
    samples, optionally 'away' / 'toward'. measured_depth_mm: tape-measured camera lens to spoon tip at nominal."""
    by = {s["label"]: s for s in samples}
    if "nominal" not in by:
        raise ValueError("calibration needs a 'nominal' sample")
    nom = by["nominal"]
    others = [s for s in samples if s["label"] not in ("nominal",) + DEPTH_LABELS]
    if len(others) < 2:
        raise ValueError("calibration needs at least two offset samples besides nominal")
    dpx = np.array([[s["px"][0] - nom["px"][0], s["px"][1] - nom["px"][1]] for s in others])
    dq = np.array([[s["pose"][j] - nom["pose"][j] for j in joints] for s in others])
    if np.linalg.matrix_rank(dpx) < 2:
        raise ValueError("offset samples are collinear in the image; move the tip both sideways and up/down")
    M, res, *_ = np.linalg.lstsq(dpx, dq, rcond=None)
    pred = dpx @ M
    residual = float(np.sqrt(np.mean((pred - dq) ** 2)))
    px_per_mm = float(np.mean(np.linalg.norm(dpx, axis=1))) / CAL_OFFSET_MM
    if measured_depth_mm and measured_depth_mm > 0:
        z_nom, f = float(measured_depth_mm), px_per_mm * float(measured_depth_mm)
    else:
        f = float(hfov_focal_px)
        z_nom = f / px_per_mm
    warnings: list[str] = []
    cols = []
    for lab, sign in (("away", 1.0), ("toward", -1.0)):
        if lab not in by:
            continue
        s = by[lab]
        moved_px = float(np.hypot(s["px"][0] - nom["px"][0], s["px"][1] - nom["px"][1]))
        if moved_px > 0.5 * CAL_OFFSET_MM * px_per_mm:
            warnings.append(f"'{lab}' tip moved {moved_px:.0f} px in the image: move straight along the camera's "
                            "line of sight, then redo it")
        cols.append([sign * (s["pose"][j] - nom["pose"][j]) / CAL_OFFSET_MM for j in joints])
    J = np.mean(cols, axis=0).tolist() if cols else None
    if J is not None and max(abs(v) for v in J) < 1e-3:
        warnings.append("depth samples did not move the arm; depth correction is off")
        J = None
    return OffsetMap(joints, (float(nom["px"][0]), float(nom["px"][1])), {j: float(nom["pose"][j]) for j in nom["pose"]},
                     M.tolist(), px_per_mm, face_width_px, samples, time.strftime("%Y-%m-%dT%H:%M:%S"), residual,
                     f, z_nom, J, warnings)


class OffsetTracker:
    def __init__(self, cfg: Config, omap: Optional[OffsetMap]):
        self.cfg = cfg
        self.map = omap
        self.mm = np.zeros(3)               # current (rate-limited) correction, mm: image x, image y, depth
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        self.mm[:] = 0
        self._last_t = None

    def _depth_ok(self, face: FaceState) -> bool:
        return bool(self.map and self.map.focal_px > 0 and face.depth_mm > 0)

    def target_mm(self, face: FaceState) -> np.ndarray:
        out = np.zeros(3)
        m = self.map
        if not m or not face.present or not face.mouth_center_px:
            return out
        k = REF_W / face.frame_w if face.frame_w else 1.0
        dpx = np.asarray(face.mouth_center_px) * k - np.asarray(m.nominal_px)
        o = self.cfg.offsets
        if self._depth_ok(face):
            z = face.depth_mm - o.eyes_behind_lips_mm
            lat = dpx * z / m.focal_px
        else:
            fw = face.face_width_px * k
            lat = dpx * (m.face_width_px_nominal / fw if fw > 1 else 1.0) / max(1e-6, m.px_per_mm)
        n = float(np.linalg.norm(lat))
        lim = o.max_correction_mm
        out[:2] = lat * (lim / n) if n > lim else lat
        if o.depth and m.has_depth and self._depth_ok(face):
            dz = face.depth_mm - o.eyes_behind_lips_mm - m.nominal_depth_mm
            db = o.depth_deadband_mm
            dz = 0.0 if abs(dz) <= db else dz - math.copysign(db, dz)
            out[2] = max(-o.max_depth_correction_mm, min(o.max_depth_correction_mm, dz))
        return out

    def update(self, face: FaceState, now: float) -> dict[str, float]:
        """Advance the rate-limited correction toward the face and return joint deltas."""
        tgt = self.target_mm(face)
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        step = tgt - self.mm
        n = float(np.linalg.norm(step))
        cap = self.cfg.offsets.rate_limit_mm_s * dt
        self.mm = self.mm + (step * (cap / n) if n > cap > 0 else (step if cap > 0 else 0))
        return self.joint_delta()

    def joint_delta(self) -> dict[str, float]:
        if not self.map:
            return {}
        d = self.map.joint_delta(tuple(self.mm[:2] * self.map.px_per_mm))
        if self.map.depth_joint_per_mm and self.cfg.offsets.depth:
            for j, v in zip(self.map.joints, self.map.depth_joint_per_mm):
                d[j] += v * float(self.mm[2])
        return d

    @property
    def offset_mm(self) -> float:
        """Sideways/up-down correction, mm."""
        return float(math.hypot(self.mm[0], self.mm[1]))

    @property
    def depth_mm(self) -> float:
        """Depth correction, mm (+ = spoon goes further from the camera, the diner leaned back)."""
        return float(self.mm[2])
