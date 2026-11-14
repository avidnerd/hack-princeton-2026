"""Kinesthetic teaching: torque off (gripper stays on), sample at read_hz, median filter, downsample, save.

Paths: scoop, scoop_small, scoop_full, lift, present, retreat. Poses: rest, pre_delivery, mouth_nominal.
`present` / `retreat` / `pre_delivery` / `mouth_nominal` are near-face, so they are stored with speed=near_face.
"""
from __future__ import annotations

import statistics
import threading
import time
from pathlib import Path
from typing import Optional

from ..config import ARM_JOINTS, Config
from .arm import Arm
from .trajectory import Library, TrajFile, joint_dist, make_waypoints, now_iso

NEAR_FACE = {"present", "pre_delivery", "mouth_nominal"}


def median_filter(samples: list[dict], window: int) -> list[dict]:
    if window <= 1 or len(samples) < window:
        return [dict(s) for s in samples]
    h = window // 2
    out = []
    for i in range(len(samples)):
        lo, hi = max(0, i - h), min(len(samples), i + h + 1)
        out.append({j: statistics.median(s[j] for s in samples[lo:hi]) for j in ARM_JOINTS})
    return out


def downsample(poses: list[dict], times: list[float], step_deg: float) -> tuple[list[dict], list[float]]:
    if not poses:
        return [], []
    keep_p, keep_t = [poses[0]], [times[0]]
    for p, t in zip(poses[1:], times[1:]):
        if joint_dist(p, keep_p[-1]) >= step_deg:
            keep_p.append(p)
            keep_t.append(t)
    if keep_p[-1] is not poses[-1]:
        keep_p.append(poses[-1])
        keep_t.append(times[-1])
    return keep_p, keep_t


class Recorder:
    def __init__(self, cfg: Config, arm: Arm, directory: Path, profile: str = "demo"):
        self.cfg, self.arm, self.dir, self.profile = cfg, arm, directory, profile
        self.name: Optional[str] = None
        self.kind = "path"
        self.samples: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.meta: dict = {}

    @property
    def recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, name: str, kind: str = "path", meta: Optional[dict] = None) -> None:
        if self.recording:
            raise RuntimeError("already recording")
        self.name, self.kind, self.meta = name, kind, dict(meta or {})
        self.samples = []
        self._stop.clear()
        self.arm.torque(False, ARM_JOINTS)
        self._thread = threading.Thread(target=self._loop, name="teach", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        dt = 1.0 / self.cfg.arm.read_hz
        t0 = time.monotonic()
        last_ts = 0.0
        while not self._stop.is_set():
            st = self.arm.read()
            if st.ok and st.ts != last_ts:
                last_ts = st.ts
                self.samples.append((time.monotonic() - t0, {j: st.deg[j] for j in ARM_JOINTS}))
            time.sleep(dt)

    def live(self) -> dict:
        st = self.arm.read()
        return {"recording": self.recording, "name": self.name, "kind": self.kind,
                "samples": len(self.samples), "deg": {j: round(v, 1) for j, v in st.deg.items()}}

    def stop(self, hold: bool = True) -> TrajFile:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if hold:
            self.arm.torque(True, ARM_JOINTS)  # hold wherever the hand left it
        if not self.samples:
            raise RuntimeError("no samples recorded (is the arm reading positions?)")
        times = [t for t, _ in self.samples]
        raw = [p for _, p in self.samples]
        speed = "near_face" if self.name in NEAR_FACE else "normal"
        if self.kind == "pose":
            tail = [p for t, p in self.samples if t >= times[-1] - 0.5] or raw[-1:]
            pose = {j: statistics.median(p[j] for p in tail) for j in ARM_JOINTS}
            wps = make_waypoints(self.cfg, [pose], speed, [0.0])
        else:
            filt = median_filter(raw, self.cfg.arm.teach_median_window)
            poses, ts = downsample(filt, times, self.cfg.arm.teach_downsample_deg)
            wps = make_waypoints(self.cfg, poses, speed, ts)
        meta = {"profile": self.profile, "spoon_id": "silicone_159", "bowl_id": "default",
                "recorded_at": now_iso(), "duration": round(times[-1], 2), "backend": self.arm.backend.name,
                **self.meta}
        tf = TrajFile(self.name, self.kind, wps, meta,
                      [{"t": round(t, 3), "deg": {j: round(p[j], 2) for j in ARM_JOINTS}} for t, p in self.samples])
        tf.validate(self.cfg)   # raises LimitViolation: re-teach inside the soft limits
        tf.save(self.dir)
        return tf


def save_current_pose(cfg: Config, arm: Arm, directory: Path, name: str) -> TrajFile:
    st = arm.read()
    pose = {j: st.deg[j] for j in ARM_JOINTS}
    speed = "near_face" if name in NEAR_FACE else "normal"
    tf = TrajFile(name, "pose", make_waypoints(cfg, [pose], speed, [0.0]),
                  {"recorded_at": now_iso(), "backend": arm.backend.name})
    tf.validate(cfg)
    tf.save(directory)
    return tf


def library_for(cfg: Config, backend_name: str) -> Library:
    d = cfg.arm.sim_trajectories_dir if backend_name == "sim" else cfg.arm.trajectories_dir
    return Library(cfg, cfg.path(d))
