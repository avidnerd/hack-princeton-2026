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
from .perception.plate import PlateWatcher, asks_whats_on_plate, match_food
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
        self.voice = VoiceListener(cfg, self.on_voice, self.log)
        self.voice.on_unaddressed = self.on_unaddressed_voice
        self.voice.hint_fn = self.voice_hint
        self.plate_camera = None
        if cfg.plate.enabled and cfg.plate.camera_index >= 0 and opts.camera:
            self.plate_camera = Camera(cfg.camera.model_copy(update={"index": cfg.plate.camera_index}))
        cam = self.plate_camera or self.camera
        self.plate = PlateWatcher(cfg, lambda: (lambda f: f.image if f else None)(cam.latest()),
                                  active_fn=lambda: self.fsm.state.value == "LISTENING")   # only during a session
        self.fsm.plate = self.plate
        from .brain.aim_check import AimCheck
        self.aim_check = AimCheck(self)
        self.setup_state: dict = {}                # what Setup has done this run (see brain/photo_setup.status)
        from .brain.planner import Planner
        self.planner = Planner(None, self.log)
        from .perception.photo_map import FoodSurface
        self.food_surface = FoodSurface()
        self.fsm.food_surface = self.food_surface

        def anchor(bowl: str):
            from .arm.scoop_style import scoop_anchor
            name = self.plate.scoop_for(bowl)
            return scoop_anchor(cfg.kinematics, self.lib.items[name].poses) if self.lib.has(name) else None
        self.plate.anchor_fn = anchor
        from .brain.table_auto import TableAutoCal
        self.table_auto = TableAutoCal(self)
        self.plate.autocal_status_fn = self.table_auto.status
        self.plate.spoon_frame_fn = lambda: (lambda f: f.image if f else None)(self.camera.latest())
        self.recorder = Recorder(cfg, self.arm, self.lib.dir, self.profiles.current.name)
        self.calib_samples: list[dict] = []
        self.summary_outbox: list[dict] = []
        self.started_at = time.time()

    REQUEST_WORDS: tuple = ("have", "want", "give", "bite", "some", "please", "more", "eat", "try", "next")

    def on_voice(self, cmd: Optional[str], text: str, conf: float) -> None:
        """Voice: stop first, always. Then plate questions and food requests ("a bite of the applesauce"),
        then plain commands. cmd is None when the wake word was heard but no command keyword was."""
        if cmd == "stop":
            self.fsm.post("stop", source="voice")
            return
        p = self.plate
        if not self.cfg.voice.requests:          # button-only bites: voice can STOP, and (select) pick the food
            if self.cfg.voice.select and p.enabled:
                if asks_whats_on_plate(text):
                    self.speaker.say_text(p.describe())
                    return
                bowl = p.find(text)
                if bowl and len(text.split()) <= 7:        # "Spoonful, the rice": fast path, no model needed
                    self.fsm.post("select", source="voice", bowl=bowl)
                elif self.ask_planner(text):
                    pass                                   # Gemini plans; the answer is spoken when it's back
                elif bowl:
                    self.fsm.post("select", source="voice", bowl=bowl)
                elif cmd is None and any(w in text.lower() for w in self.REQUEST_WORDS + ("switch", "change")):
                    self.speaker.say_text("I don't see that on the plate. " + p.describe())
            return
        if p.enabled:
            if asks_whats_on_plate(text):
                self.speaker.say_text(p.describe())
                return
            bowl = p.find(text)
            if bowl and cmd in (None, "bite"):
                self.fsm.post("bite", source="voice", bowl=bowl)
                return
            if cmd is None and any(w in text.lower() for w in self.REQUEST_WORDS):
                self.speaker.say_text("I don't see that on the plate. " + p.describe())
                return
        if cmd:
            self.fsm.post(cmd, source="voice")

    def voice_hint(self) -> str:
        """The foods on the table, as a plain list. No wake word and no command in it: on noise the recogniser can
        write its hint out, and a list of foods is not a request."""
        if not self.cfg.voice.hint:
            return ""
        return ", ".join(f for f in self.plate.labels().values() if f)

    def on_unaddressed_voice(self, text: str, conf: float) -> None:
        """Said without "Spoonful" (or the word was cut off). Picking a food is harmless, since nothing moves until
        GO, so a clear food name still counts. Everything else is ignored."""
        if self.cfg.voice.requests or not self.cfg.voice.select or not self.plate.enabled:
            return
        named = [b for b, food in self.plate.labels().items() if food and match_food(text, {b: food})]
        if len(named) == 1:                      # exactly one food named; a list of foods is not a choice
            self.fsm.post("select", source="voice", bowl=named[0])

    def planner_scene(self) -> list[dict]:
        out = []
        for b in self.plate.status()["bowls"]:
            out.append({"name": b["name"], "food": b["food"], "amount": b["amount"], "texture": b["texture"],
                        "empty": b["empty"], "taught": self.lib.has(b["scoop"])})
        return out

    def ask_planner(self, text: str) -> bool:
        """Open-ended request -> Gemini plan -> verified -> executed (speech, which bowl next, a queue of bites).
        Never moves the arm directly. False if no model is available (caller falls back to simple matching)."""
        if self.planner.generate is None:
            self.planner.generate = self.plate.text_generate
        if not self.planner.available:
            return False

        def done(steps, problems):
            said = False
            has_say = any(st["skill"] == "say" for st in steps)   # a decline is only spoken if nothing else is
            for st in steps:
                if st["skill"] == "say":
                    self.speaker.say_text(st["text"])
                    said = True
                elif st["skill"] == "select":
                    self.fsm.post("select", source="planner", bowl=st["bowl"], quiet=said)
                elif st["skill"] == "queue":
                    self.fsm.post("queue", source="planner", bowls=st["bowls"])
                elif st["skill"] == "decline" and not said and not has_say:
                    self.speaker.say_text(st["reason"] or "I can't do that")
                    said = True
            if not steps:
                self.speaker.say_text("Sorry, I didn't get that")

        def failed(e):
            self.log.event("plan_failed", detail_error=repr(e)[:200])
            bowl = self.plate.find(text)
            if bowl:
                self.fsm.post("select", source="voice", bowl=bowl)
            else:
                self.speaker.say_text("Sorry, I couldn't think that through")
        return self.planner.plan_async(text, self.planner_scene(), done, failed)

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
        if self.plate_camera:
            self.plate_camera.start()
        self.plate.start()
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
        for c in (self.watchdog, self.voice, self.esp32, self.plate):
            c.stop()
        if self.plate_camera:
            self.plate_camera.stop()
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
