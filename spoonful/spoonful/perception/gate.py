"""The consent gate. No I/O; deterministic given (face, fsm_state, now) and its own small memory.

Deny reasons (exact strings, shown on screen and logged):
  stop_latched, paused, frame_stale, no_face, face_too_brief, looking_away, mouth_outside_window, mouth_closed
Permit requires: face present for >= presence_hold_s, |yaw - yaw_center| <= yaw_max (yaw_center = where the diner
looks to face the arm), |pitch - offset| <= pitch_max, mouth inside the delivery window, mouth open (jawOpen, or the
lip-gap ratio as fallback) >= threshold for >= mouth_open_hold_s, frame age <=
frame_max_age_ms. Hysteresis: once permitted, the mouth may close to hysteresis_fraction x threshold (and the
hold timer is not re-required) before denial, so the gate does not flutter mid-delivery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import FaceCfg
from .face import FaceState

REASONS = ("stop_latched", "paused", "frame_stale", "no_face", "face_too_brief", "looking_away",
           "mouth_outside_window", "mouth_closed")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""          # "" when ok

    def __bool__(self) -> bool:
        return self.ok


class Gate:
    def __init__(self, cfg: FaceCfg):
        self.cfg = cfg
        self.present_since: Optional[float] = None
        self.open_since: Optional[float] = None
        self.permitted = False
        self.last = Verdict(False, "no_face")

    def reset(self) -> None:
        self.present_since = self.open_since = None
        self.permitted = False

    def in_window(self, face: FaceState) -> bool:
        if not face.mouth_center_px:
            return False
        sx = face.frame_w / 1280.0 if face.frame_w else 1.0
        sy = face.frame_h / 720.0 if face.frame_h else 1.0
        x, y = face.mouth_center_px
        w = self.cfg.delivery_window_px
        return w.x[0] * sx <= x <= w.x[1] * sx and w.y[0] * sy <= y <= w.y[1] * sy

    def openness(self, face: FaceState) -> tuple[float, float]:
        """(value, threshold). jawOpen comes from the model's 3D fit, so it still reads with the head turned; the
        2D lip-gap ratio is the fallback when the model gives no blendshapes."""
        if self.cfg.mouth_signal == "jaw" and face.jaw_open >= 0:
            return face.jaw_open, self.cfg.jaw_open_threshold
        return face.mouth_open_ratio, self.cfg.mouth_open_ratio_threshold

    def mouth_open(self, face: FaceState) -> bool:
        v, th = self.openness(face)
        if self.permitted:
            th *= self.cfg.hysteresis_fraction
        return v >= th

    def permit(self, face: FaceState, fsm_state: str, now: float) -> Verdict:
        v = self._evaluate(face, fsm_state, now)
        self.permitted = v.ok
        self.last = v
        return v

    def _evaluate(self, face: FaceState, fsm_state: str, now: float) -> Verdict:
        c = self.cfg
        # timers track the face regardless of state, so a resume does not have to re-wait
        if face.present:
            self.present_since = self.present_since if self.present_since is not None else face.ts
        else:
            self.present_since = None
        v, th = self.openness(face)
        raw_open = face.present and v >= th
        if raw_open:
            self.open_since = self.open_since if self.open_since is not None else face.ts
        elif not (self.permitted and self.mouth_open(face)):
            self.open_since = None

        if fsm_state == "STOPPED":
            return Verdict(False, "stop_latched")
        if fsm_state == "PAUSED":
            return Verdict(False, "paused")
        if face.ts == 0 or (now - face.ts) * 1000 > c.frame_max_age_ms:
            return Verdict(False, "frame_stale")
        if not face.present:
            return Verdict(False, "no_face")
        if now - self.present_since < c.presence_hold_s:
            return Verdict(False, "face_too_brief")
        if abs(face.yaw_deg - c.yaw_center_deg) > c.yaw_max_deg or abs(face.pitch_deg - c.pitch_offset_deg) > c.pitch_max_deg:
            return Verdict(False, "looking_away")
        if not self.in_window(face):
            return Verdict(False, "mouth_outside_window")
        if not self.mouth_open(face):
            return Verdict(False, "mouth_closed")
        if not self.permitted and (self.open_since is None or now - self.open_since < c.mouth_open_hold_s):
            return Verdict(False, "mouth_closed")
        return Verdict(True, "")
