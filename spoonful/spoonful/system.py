"""Builds and wires every component. Used by the CLI and the web app."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .arm.arm import Arm
from .arm.backend import SimBackend, make_backend
from .arm.teach import Recorder, library_for
from .config import ARM_JOINTS, JOINTS, Config
from .log import Log
from .perception.camera import Camera
from .perception.face import FaceTracker
from .perception.food import FoodWatcher
from .perception.gate import Gate
from .perception.face import REF_W, focal_px
from .perception.offsets import OffsetMap, OffsetTracker, fit
from .brain.fsm import FSM
from .brain.pace import Pace
from .brain.profiles import ProfileStore
from .brain.session import Session
from .brain.speech import Speaker
from .brain.voice import VoiceListener
from .brain.watchdog import Watchdog
from .io.esp32 import Esp32


@dataclass
class Options:
    backend: Optional[str] = None
    sim_face: bool = False
    camera: bool = True
    voice: Optional[bool] = None
    esp32: Optional[bool] = None
    watchdog: bool = True


class System:
    def __init__(self, cfg: Config, opts: Options = Options()):
        self.cfg, self.opts = cfg, opts
        self.backend_kind = opts.backend or cfg.arm.backend
        self.log = Log(cfg.path(cfg.log.db))
        ids = [cfg.arm.servo_map[j] for j in JOINTS]
        start = None
        if self.backend_kind == "sim":
            start = {cfg.arm.servo_map[j]: int(cfg.arm.center_units + cfg.arm.park_pose.get(j, 0) * cfg.arm.units_per_deg)
                     for j in JOINTS}
        self.backend = make_backend(self.backend_kind, ids, start)
        self.arm = Arm(cfg, self.backend, self.log)
        self.lib = library_for(cfg, self.backend.name)
        if not opts.camera:
            cfg.camera.index = -1
        self.camera = Camera(cfg.camera)
        self.face = FaceTracker(cfg, self.camera)
        self.face.sim_enabled = opts.sim_face
        self.gate = Gate(cfg.face)
        self.offset_map = OffsetMap.load(cfg.path(cfg.offsets.joint_map))
        self.offsets = OffsetTracker(cfg, self.offset_map)
        self._use_map_geometry(self.offset_map)
        self.speaker = Speaker(cfg, self.log)
        self.pace = Pace(cfg)
        self.profiles = ProfileStore(cfg.path(cfg.profiles.dir), cfg.profiles.default)
        self.session = Session(self.log)
        self.fsm = FSM(cfg, self.arm, self.lib, self.gate, self.offsets, self.speaker, self.pace, self.profiles,
                       self.session, self.log, self.face.latest, self.camera.age_s)
        self.food = FoodWatcher(cfg)
        self.fsm.food = self.food
        self.fsm.on_summary = self._on_summary
        self.watchdog = Watchdog(self.fsm, self.arm, self.camera, self.log)
        if opts.esp32 is not None:
            cfg.esp32.enabled = opts.esp32
        btn_event = "go" if cfg.esp32.button_role == "go" else "button"
        self.esp32 = Esp32(cfg.esp32, lambda: self.fsm.post(btn_event, source="button"),
                           lambda env: self.fsm.post("piezo", envelope=env), self.on_stop_switch)
        self._switch_prev: Optional[bool] = None
        if opts.voice is not None:
            cfg.voice.enabled = opts.voice
        self.voice = VoiceListener(cfg, lambda cmd, text, conf: self.fsm.post(cmd, source="voice"), self.log)
        self.recorder = Recorder(cfg, self.arm, self.lib.dir, self.profiles.current.name)
        self.calib_samples: list[dict] = []
        self.summary_outbox: list[dict] = []
        self.started_at = time.time()

    def on_stop_switch(self, closed: Optional[bool]) -> None:
        """ESP32 STOP input. closed=None means the switch can't be read (device lost after it was connected).

        rocker: closed (I) = run, open (O) or unreadable = STOP, held while off. Called every heartbeat as well as
                on change, so the level is re-asserted. Flipping back on arms the release (like a STOP press);
                the UI confirm then releases.
        button: closed = pressed = STOP; a press while STOPPED arms the release.
        """
        mode = self.cfg.esp32.stop_switch
        prev, self._switch_prev = self._switch_prev, closed
        if mode == "rocker":
            # through the FSM queue (not set here) so a UI confirm queued while the switch was off is judged
            # against the switch state at the time, in order
            self.fsm.post("stop_switch", off=closed is not True, lost=closed is None)
        elif mode == "button":
            if closed is True and prev is not True:
                self.fsm.post("stop", source="button")

    @property
    def is_sim(self) -> bool:
        return isinstance(self.backend, SimBackend)

    def start(self) -> None:
        self.log.event("boot", detail_backend=self.backend_kind, versions=versions())
        self.arm.start()
        self.camera.start()
        self.face.start()
        self.speaker.prewarm()
        self.fsm.start()
        self.esp32.start()
        self.voice.start()
        if self.opts.watchdog:
            self.watchdog.start()
        if self.lib.errors:
            self.log.fault("trajectory_invalid", errors=self.lib.errors)
        self.fsm.post("boot")

    def stop(self) -> None:
        try:
            self.arm.hold("shutdown")
        except Exception:
            pass
        for c in (self.watchdog, self.voice, self.esp32):
            c.stop()
        self.fsm.stop_thread()
        self.face.stop()
        self.camera.stop()
        self.arm.close()
        self.log.close()

    def _on_summary(self, sid: int, text: str) -> None:
        item = {"session_id": sid, "text": text, "sent": False, "detail": "", "ts": time.time()}
        self.summary_outbox.append(item)
        if self.cfg.photon.enabled:
            import threading
            threading.Thread(target=self.send_summary, args=(item,), daemon=True).start()

    def send_summary(self, item: dict) -> dict:
        from .io.photon_client import send_text
        ok, detail = send_text(self.cfg.photon, item["text"])
        item["sent"], item["detail"] = ok, detail
        if ok and item.get("session_id"):
            self.log.mark_summary_sent(item["session_id"])
        self.log.event("summary_send", detail_ok=ok, why=detail)
        return item

    # --------------------------------------------------------------- calibration (offset map)
    def _use_map_geometry(self, m: Optional[OffsetMap]) -> None:
        """The map's measured focal length drives the depth estimator; the sim face sits at the map's nominal."""
        if m and m.focal_px > 0:
            self.cfg.offsets.focal_px = m.focal_px
        if m and m.nominal_depth_mm > 0:
            self.face.sim_nominal_depth_mm = m.nominal_depth_mm

    def calib_add(self, label: str, px: tuple[float, float]) -> dict:
        """px in the current camera frame's coords; stored in 1280-wide coords like everything else."""
        st = self.arm.read()
        k = REF_W / self.camera.width if self.camera.width else 1.0
        s = {"label": label, "px": [float(px[0]) * k, float(px[1]) * k], "pose": {j: st.deg[j] for j in ARM_JOINTS}}
        self.calib_samples = [x for x in self.calib_samples if x["label"] != label] + [s]
        self.log.event("calib_sample", detail_label=label, px=s["px"])
        return s

    def calib_fit(self, measured_depth_mm: Optional[float] = None) -> OffsetMap:
        fw = self.face.latest().face_width_px or 220.0
        self.cfg.offsets.focal_px = 0           # re-derive from hfov unless a tape measurement is given
        m = fit(self.calib_samples, self.cfg.offsets.joints, fw, focal_px(self.cfg), measured_depth_mm)
        m.save(self.cfg.path(self.cfg.offsets.joint_map))
        self.cfg.offsets.px_per_mm_at_nominal = m.px_per_mm
        self.offset_map = m
        self.offsets.map = m
        self._use_map_geometry(m)
        self.log.event("calib_fit", detail_px_per_mm=round(m.px_per_mm, 3), residual_deg=round(m.residual_deg, 2),
                       focal_px=round(m.focal_px, 1), nominal_depth_mm=round(m.nominal_depth_mm),
                       depth=m.has_depth, warnings=m.warnings)
        return m


def versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ("xarm", "hidapi", "mediapipe", "opencv-python", "fastapi", "uvicorn", "pydantic", "faster-whisper",
              "elevenlabs", "google-genai", "pyserial", "sounddevice", "numpy"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out
