"""MediaPipe Face Landmarker (tasks API, VIDEO mode, one face) -> FaceState, plus the on-screen overlay.

Verified against mediapipe 1.2.3: vision.FaceLandmarker.create_from_options, detect_for_video(mp.Image, ts_ms),
result.face_landmarks[0][i].x/.y (normalised), result.facial_transformation_matrixes[0] (4x4).
Yaw/pitch signs are not trusted anywhere: the gate uses |yaw| and |pitch - pitch_offset|.
The face_landmarker.task model outputs 478 landmarks: 468..472 left iris (centre + 4 ring), 473..477 right iris.

Depth (camera to eyes, mm): see DepthEstimator.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from ..config import Config
from .camera import Camera, Frame

UPPER_INNER_LIP, LOWER_INNER_LIP, FOREHEAD, CHIN = 13, 14, 10, 152
CHEEK_L, CHEEK_R = 234, 454
IRIS_RINGS = ((469, 470, 471, 472), (474, 475, 476, 477))   # each: right, top, left, bottom edge
REF_W = 1280.0                                             # calibration and focal_px live in 1280-wide coords
SIM_FACE_WIDTH_MM = 140.0
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
        149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@dataclass
class FaceState:
    present: bool = False
    ts: float = 0.0                     # capture time (monotonic)
    latency_ms: float = 0.0             # capture -> landmarks
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    mouth_open_ratio: float = 0.0
    mouth_center_px: Optional[tuple[float, float]] = None
    face_width_px: float = 0.0
    frame_w: int = 0
    frame_h: int = 0
    landmarks: Optional[list[tuple[float, float]]] = field(default=None, repr=False)  # overlay subset, px
    simulated: bool = False
    iris_px: float = 0.0                # mean iris diameter, px (0 = not visible)
    depth_mm: float = 0.0               # camera to eyes, mm (0 = unknown)
    depth_src: str = ""                 # "iris" (learning this face) | "face" (face width, learned via iris)

    def scalars(self) -> dict:
        d = asdict(self)
        d.pop("landmarks")
        return d


def yaw_pitch_from_matrix(m: np.ndarray) -> tuple[float, float]:
    r = np.asarray(m)[:3, :3]
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))
    return yaw, pitch


class Ema:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.v: dict[str, float] = {}

    def __call__(self, key: str, x: float) -> float:
        prev = self.v.get(key)
        self.v[key] = x if prev is None else prev + self.alpha * (x - prev)
        return self.v[key]

    def reset(self) -> None:
        self.v.clear()


def focal_px(cfg: Config) -> float:
    """Focal length in 1280-wide pixel coords: measured by calibration, else from the camera's field of view."""
    if cfg.offsets.focal_px > 0:
        return cfg.offsets.focal_px
    return (REF_W / 2) / math.tan(math.radians(cfg.camera.hfov_deg) / 2)


def iris_diameter_px(px) -> float:
    """Mean of both eyes; per eye the larger of the horizontal and vertical diameter (yaw foreshortens one,
    pitch the other)."""
    ds = [max(math.dist(px(a), px(c)), math.dist(px(b), px(d))) for a, b, c, d in IRIS_RINGS]
    return sum(ds) / len(ds)


class DepthEstimator:
    """Camera-to-eyes distance in mm, from one 2D camera.

    The iris is close to the same size in every adult (11.7 +- 0.5 mm), so it gives absolute distance, but it is
    only ~25 px across and jitters by a pixel or two (about 5 %). Face width is ~300 px and smooth, but varies
    about +-8 % between people. So the iris is used to learn *this* person's face width in mm (median over frames
    where they face the camera), and from then on depth comes from face width: absolute like the iris, smooth
    like the face. The learned width is dropped when the face has been gone for depth_person_reset_s."""

    MIN_SAMPLES = 20

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.samples: deque[float] = deque(maxlen=150)
        self.face_width_mm: Optional[float] = None
        self._last_seen = -1e9

    def reset(self) -> None:
        self.samples.clear()
        self.face_width_mm = None

    def update(self, iris_px: float, face_w_px: float, yaw_deg: float, frame_w: int,
               now: float) -> tuple[float, str]:
        o = self.cfg.offsets
        if now - self._last_seen > o.depth_person_reset_s:
            self.reset()
        self._last_seen = now
        f = focal_px(self.cfg) * (frame_w / REF_W if frame_w else 1.0)
        fw = face_w_px / math.cos(math.radians(min(abs(yaw_deg), 40.0)))   # undo yaw foreshortening
        if iris_px > 1 and fw > 1:
            z_iris = f * o.iris_mm / iris_px
            if abs(yaw_deg) < 12:
                self.samples.append(fw * z_iris / f)
                if len(self.samples) >= self.MIN_SAMPLES:
                    self.face_width_mm = float(np.median(self.samples))
        if self.face_width_mm and fw > 1:
            return f * self.face_width_mm / fw, "face"
        if iris_px > 1:
            return f * o.iris_mm / iris_px, "iris"
        return 0.0, ""


class FaceTracker:
    """Runs in its own thread: newest camera frame -> FaceState. `sim` lets the operator drawer fake a face
    (dev without a person in front of the laptop); it is off unless the app was started with --sim-face."""

    def __init__(self, cfg: Config, camera: Camera):
        self.cfg = cfg
        self.camera = camera
        self.state = FaceState()
        self.fps = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ema = Ema(cfg.face.ema_alpha)
        self.depth = DepthEstimator(cfg)
        self.sim_enabled = False
        # dz: mm the simulated diner sits further from the camera than the calibrated nominal (+ = leaned back)
        self.sim = {"present": True, "mouth_open": False, "yaw": 0.0, "pitch": 0.0, "dx": 0.0, "dy": 0.0, "dz": 0.0}
        self.sim_nominal_depth_mm = 450.0   # System sets this from the offset map
        self.landmarker = None
        self.error: Optional[str] = None
        self._last_ts_ms = 0

    def start(self) -> None:
        try:
            from mediapipe.tasks.python import BaseOptions, vision
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.cfg.path(self.cfg.face.model_path))),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                output_facial_transformation_matrixes=True)
            self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        except Exception as e:
            self.error = f"face model failed to load: {e!r}"
        self._thread = threading.Thread(target=self._loop, name="face", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        # mediapipe 1.2.3: FaceLandmarker.__del__ -> close() deadlocks at interpreter exit (its dispatcher is
        # already gone). Close explicitly while it is still alive, then drop the reference.
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def latest(self) -> FaceState:
        with self._lock:
            return self.state

    def _loop(self) -> None:
        seq, n, t_fps, low_since = 0, 0, time.monotonic(), None
        while not self._stop.is_set():
            fr = self.camera.wait_newer(seq, 0.5)
            if fr is None or fr.seq == seq:
                continue
            seq = fr.seq
            st = self._simulate(fr) if self.sim_enabled else self.process(fr)
            with self._lock:
                self.state = st
            n += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                self.fps = n / (now - t_fps)
                n, t_fps = 0, now
                c = self.cfg.camera
                if self.fps < c.min_fps_before_fallback and self.camera.width > c.fallback_width:
                    low_since = low_since or now
                    if now - low_since > 3:
                        self.camera.set_resolution(c.fallback_width, c.fallback_height)
                else:
                    low_since = None

    def process(self, fr: Frame) -> FaceState:
        h, w = fr.image.shape[:2]
        if self.landmarker is None:
            return FaceState(False, fr.ts, 0.0, frame_w=w, frame_h=h)
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(fr.image, cv2.COLOR_BGR2RGB)
        ts_ms = max(self._last_ts_ms + 1, int(fr.ts * 1000))   # VIDEO mode needs strictly increasing timestamps
        self._last_ts_ms = ts_ms
        res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        latency = (time.monotonic() - fr.ts) * 1000
        if not res.face_landmarks:
            self._ema.reset()
            return FaceState(False, fr.ts, latency, frame_w=w, frame_h=h)
        lm = res.face_landmarks[0]
        px = lambda i: (lm[i].x * w, lm[i].y * h)  # noqa: E731
        up, lo, top, chin = px(UPPER_INNER_LIP), px(LOWER_INNER_LIP), px(FOREHEAD), px(CHIN)
        face_h = max(1.0, math.dist(top, chin))
        ratio = math.dist(up, lo) / face_h
        mouth = ((up[0] + lo[0]) / 2, (up[1] + lo[1]) / 2)
        face_w = math.dist(px(CHEEK_L), px(CHEEK_R))
        iris = iris_diameter_px(px) if len(lm) >= 478 else 0.0
        yaw = pitch = 0.0
        if res.facial_transformation_matrixes:
            yaw, pitch = yaw_pitch_from_matrix(res.facial_transformation_matrixes[0])
        e = self._ema
        e.alpha = self.cfg.face.ema_alpha
        st = FaceState(
            True, fr.ts, latency, e("yaw", yaw), e("pitch", pitch), e("ratio", ratio),
            (e("mx", mouth[0]), e("my", mouth[1])), e("fw", face_w), w, h,
            [px(i) for i in LIPS + OVAL], iris_px=e("iris", iris) if iris else 0.0)
        return self._with_depth(st)

    def _with_depth(self, st: FaceState) -> FaceState:
        st.depth_mm, st.depth_src = self.depth.update(st.iris_px, st.face_width_px, st.yaw_deg, st.frame_w, st.ts)
        return st

    def _simulate(self, fr: Frame) -> FaceState:
        h, w = fr.image.shape[:2]
        s = self.sim
        if not s["present"]:
            return FaceState(False, fr.ts, 1.0, frame_w=w, frame_h=h, simulated=True)
        wx, wy = self.cfg.face.delivery_window_px.x, self.cfg.face.delivery_window_px.y
        k = w / REF_W
        cx, cy = ((wx[0] + wx[1]) / 2 + s["dx"]) * k, ((wy[0] + wy[1]) / 2 + s["dy"]) * k
        # an exact pinhole projection of a 140 mm face with a 11.7 mm iris at the simulated distance
        z = self.sim_nominal_depth_mm + self.cfg.offsets.eyes_behind_lips_mm + float(s.get("dz", 0.0))
        f = focal_px(self.cfg) * k
        yaw = float(s["yaw"])
        fw = f * SIM_FACE_WIDTH_MM / z * math.cos(math.radians(min(abs(yaw), 40.0)))
        st = FaceState(True, fr.ts, 1.0, yaw, s["pitch"], 0.15 if s["mouth_open"] else 0.01, (cx, cy), fw, w, h,
                       None, True, iris_px=f * self.cfg.offsets.iris_mm / z)
        return self._with_depth(st)


# --------------------------------------------------------------------------- overlay
STATE_COLOURS = {True: (90, 200, 90), False: (60, 60, 230)}


def draw_overlay(img: np.ndarray, cfg: Config, face: FaceState, verdict_ok: Optional[bool], reason: str,
                 fsm_state: str, countdown: Optional[float] = None) -> np.ndarray:
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    sx, sy = w / 1280.0, h / 720.0          # delivery window is stored in 1280x720 coords
    win = cfg.face.delivery_window_px
    col = STATE_COLOURS.get(bool(verdict_ok), (200, 200, 200))
    cv2.rectangle(out, (int(win.x[0] * sx), int(win.y[0] * sy)), (int(win.x[1] * sx), int(win.y[1] * sy)), col, 2)
    if face.landmarks:
        for x, y in face.landmarks:
            cv2.circle(out, (int(x), int(y)), 2, (230, 230, 230), -1)
    if face.mouth_center_px:
        mx, my = face.mouth_center_px
        cv2.drawMarker(out, (int(mx), int(my)), col, cv2.MARKER_CROSS, 28, 3)
        if face.depth_mm > 0:
            cv2.putText(out, f"{face.depth_mm / 10:.0f} cm", (int(mx) + 20, int(my) + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (230, 230, 230), 2, cv2.LINE_AA)
    banner = 64
    cv2.rectangle(out, (0, h - banner), (w, h), (20, 20, 20), -1)
    label = "OK TO APPROACH" if verdict_ok else (reason or "").replace("_", " ").upper()
    cv2.putText(out, label, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3, cv2.LINE_AA)
    cv2.putText(out, fsm_state, (w - 20 - 22 * len(fsm_state), h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (230, 230, 230), 2, cv2.LINE_AA)
    if face.simulated:
        cv2.putText(out, "SIMULATED FACE", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    return out
