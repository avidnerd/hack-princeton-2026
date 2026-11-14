"""Threaded camera capture. Consumers get the newest frame with its capture timestamp; nobody waits on I/O.

index < 0 gives a synthetic grey frame (headless tests, or no camera): the face pipeline then reports no face.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import CameraCfg


@dataclass
class Frame:
    image: np.ndarray       # BGR
    ts: float               # time.monotonic() at capture
    seq: int


def open_capture(index: int, api: str = "auto"):
    """cv2.VideoCapture with an explicit backend. Windows' default (Media Foundation) can take 10+ s to open and
    often ignores the 720p request, so `auto` uses DirectShow there."""
    import cv2
    if api == "auto":
        api = "dshow" if sys.platform == "win32" else "any"
    flag = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
            "avfoundation": cv2.CAP_AVFOUNDATION, "v4l2": cv2.CAP_V4L2}[api]
    return cv2.VideoCapture(index, flag)


class Camera:
    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self.cap = None
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fps = 0.0
        self.width, self.height = cfg.width, cfg.height
        self.error: Optional[str] = None
        self._want_res: Optional[tuple[int, int]] = None

    def start(self) -> None:
        if self.cfg.index >= 0:
            self.cap = open_capture(self.cfg.index, self.cfg.api)
            self._apply_res(self.cfg.width, self.cfg.height)
            if not self.cap.isOpened():
                self.error = f"camera index {self.cfg.index} did not open"
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _apply_res(self, w: int, h: int) -> None:
        import cv2
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

    def set_resolution(self, w: int, h: int) -> None:
        self._want_res = (w, h)

    def _loop(self) -> None:
        seq, t_fps, n_fps = 0, time.monotonic(), 0
        while not self._stop.is_set():
            if self.cap is None:
                img = np.full((self.cfg.height, self.cfg.width, 3), 40, np.uint8)
                time.sleep(1.0 / self.cfg.fps)
                ok = True
            else:
                if self._want_res:
                    self._apply_res(*self._want_res)
                    self._want_res = None
                ok, img = self.cap.read()
                if not ok:
                    self.error = "frame read failed"
                    time.sleep(0.05)
                    continue
                if self.cfg.mirror:
                    img = img[:, ::-1].copy()
            ts = time.monotonic()
            seq += 1
            n_fps += 1
            if ts - t_fps >= 1.0:
                self.fps = n_fps / (ts - t_fps)
                t_fps, n_fps = ts, 0
            self.height, self.width = img.shape[:2]
            with self._cond:
                self._latest = Frame(img, ts, seq)
                self.error = None
                self._cond.notify_all()

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def wait_newer(self, seq: int, timeout: float = 0.5) -> Optional[Frame]:
        with self._cond:
            self._cond.wait_for(lambda: self._latest is not None and self._latest.seq > seq, timeout)
            return self._latest

    def age_s(self) -> float:
        f = self.latest()
        return time.monotonic() - f.ts if f else 1e9

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()
