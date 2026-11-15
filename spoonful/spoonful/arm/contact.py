"""Contact detection without current sensing: commanded-vs-actual position error.

The LX-15D has no current feedback, so a blocked joint shows up as the pot lagging the setpoint. Servos lag a
little when moving, so the allowance grows with commanded speed. Error above the allowance on any arm joint for
contact_hold_s = contact. The FSM decides what to do (near face: back off; elsewhere: park + FAULT).
"""
from __future__ import annotations

from typing import Mapping, Optional

from ..config import ARM_JOINTS, Config
from . import kinematics as K


class Contact(RuntimeError):
    def __init__(self, joint: str, err_deg: float, in_face_zone: bool, grew_deg: Optional[float] = None):
        super().__init__(f"contact on {joint}: {err_deg:.1f} deg error"
                         + (f" (grew {grew_deg:.1f} deg)" if grew_deg is not None else ""))
        self.joint, self.err_deg, self.in_face_zone = joint, err_deg, in_face_zone
        self.grew_deg = grew_deg            # set when the plan's lag_limit_deg tripped, not an absolute threshold


class ContactMonitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._since: dict[str, float] = {}
        self._grow_since: dict[str, float] = {}

    def reset(self) -> None:
        self._since.clear()
        self._grow_since.clear()

    def allowance(self, speed_deg_s: float, extra_deg: float = 0.0, joint: str = "") -> float:
        a = self.cfg.arm
        base = a.contact_error_deg_joint.get(joint, a.contact_error_deg)
        return base + a.contact_grace_deg * speed_deg_s / 10.0 + max(0.0, min(5.0, extra_deg))

    def check(self, now: float, commanded: Mapping[str, float], actual: Mapping[str, float],
              speed_deg_s: float, extra_deg: float = 0.0) -> Optional[Contact]:
        for j in ARM_JOINTS:
            if j not in actual or j not in commanded:
                continue
            allow = self.allowance(speed_deg_s, extra_deg, j)
            err = abs(commanded[j] - actual[j])
            if err > allow:
                since = self._since.setdefault(j, now)
                if now - since >= self.cfg.arm.contact_hold_s:
                    return Contact(j, err, K.in_face_zone(self.cfg.kinematics, actual))
            else:
                self._since.pop(j, None)
        return None

    def check_growth(self, now: float, commanded: Mapping[str, float], actual: Mapping[str, float],
                     base: Mapping[str, float], limit_deg: float, speed_deg_s: float) -> Optional[Contact]:
        """Plans with lag_limit_deg: `base` is each watched joint's signed lag (commanded - actual) when the plan
        started, so steady sag is not counted. A lag that has changed by more than limit_deg plus the speed grace
        for contact_hold_s is a contact. Used as well as check(), never instead of it."""
        a = self.cfg.arm
        allow = max(0.0, limit_deg) + a.contact_grace_deg * speed_deg_s / 10.0
        for j, b in base.items():
            if j not in actual or j not in commanded:
                continue
            grew = abs(commanded[j] - actual[j] - b)
            if grew > allow:
                since = self._grow_since.setdefault(j, now)
                if now - since >= a.contact_hold_s:
                    return Contact(j, abs(commanded[j] - actual[j]), K.in_face_zone(self.cfg.kinematics, actual), grew)
            else:
                self._grow_since.pop(j, None)
        return None
