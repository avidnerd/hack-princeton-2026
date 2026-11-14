"""Session lifecycle: start, counters, end-of-session summary for the caregiver (app + optional Photon text)."""
from __future__ import annotations

import time
from typing import Optional

from ..log import Log


class Session:
    def __init__(self, log: Log):
        self.log = log
        self.id: Optional[int] = None
        self.profile = ""
        self.started = 0.0
        self.bites = self.taken = self.pauses = self.faults = 0
        self.last_bite_done: Optional[float] = None
        self.last_summary: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.id is not None

    def start(self, profile: str) -> int:
        if self.active:
            return self.id
        self.id = self.log.start_session(profile)
        self.profile, self.started = profile, time.time()
        self.bites = self.taken = self.pauses = self.faults = 0
        self.last_bite_done = None
        self.log.event("session_start", detail_profile=profile, session=self.id)
        return self.id

    def record_bite(self, outcome: str, ts_done: float) -> None:
        self.bites += 1
        if outcome == "taken":
            self.taken += 1
        self.last_bite_done = ts_done

    def summary(self, pace_label: str = "") -> str:
        mins = max(1, round((time.time() - self.started) / 60)) if self.started else 0
        declined = self.bites - self.taken
        parts = [f"Spoonful: {self.profile} ate {self.taken} bite{'s' if self.taken != 1 else ''} in {mins} min"]
        if declined:
            parts.append(f"{declined} offered and not taken")
        parts.append(f"{self.pauses} pause{'s' if self.pauses != 1 else ''}")
        parts.append("no faults" if not self.faults else f"{self.faults} fault{'s' if self.faults != 1 else ''}")
        s = ", ".join(parts) + "."
        if pace_label:
            s += f" {pace_label[0].upper()}{pace_label[1:]}."
        return s

    def end(self, pace_label: str = "") -> Optional[str]:
        if not self.active:
            return None
        text = self.summary(pace_label)
        self.log.end_session(self.id, self.bites, self.pauses, self.faults, text)
        self.log.event("session_end", detail_summary=text, session=self.id)
        self.last_summary = text
        self.id = None
        return text
