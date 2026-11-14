"""Bite-interval model, learned per person from the log.

- Before min_bites_for_model intervals: default_interval_s.
- After: median of the diner's own request intervals (bites they asked for), x profile.pace_scale.
- With the piezo: a swallow burst after the last bite clears it; offer at swallow + swallow_offset_s.
Shown on screen as "learned pace: every N s".
"""
from __future__ import annotations

import statistics
import time
from typing import Optional

from ..config import Config
from .profiles import Profile

MAX_GAP_S = 120.0   # gaps longer than this are breaks, not pace


def intervals(bites: list[dict]) -> list[float]:
    """bites: rows from the log (any order). Uses diner-initiated requests within one session."""
    rows = sorted((b for b in bites if b.get("ts_request") and b.get("source") != "pace"),
                  key=lambda b: b["ts_request"])
    out = []
    for a, b in zip(rows, rows[1:]):
        if a.get("session_id") != b.get("session_id"):
            continue
        start = a.get("ts_done") or a["ts_request"]
        gap = b["ts_request"] - start
        if 0 < gap < MAX_GAP_S:
            out.append(gap)
    return out


class Pace:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.interval_s = cfg.pace.default_interval_s
        self.learned = False
        self.n = 0
        self.last_swallow: Optional[float] = None
        self._env_high = False
        self.envelope = 0.0

    def refit(self, bites: list[dict], profile: Profile) -> float:
        iv = intervals(bites)
        self.n = len(iv)
        c = self.cfg.pace
        if len(iv) >= c.min_bites_for_model:
            base, self.learned = statistics.median(iv[-20:]), True
        else:
            base, self.learned = c.default_interval_s, False
        self.interval_s = max(c.min_interval_s, base * profile.pace_scale)
        return self.interval_s

    def on_piezo(self, envelope: float, now: Optional[float] = None) -> bool:
        """Rising edge over the threshold = one swallow. Returns True on a new swallow."""
        now = now or time.time()
        self.envelope = envelope
        high = envelope >= self.cfg.pace.swallow_threshold
        rising = high and not self._env_high
        self._env_high = high
        if rising:
            self.last_swallow = now
        return rising

    def suggest_next_offer_time(self, last_done: Optional[float]) -> Optional[float]:
        if last_done is None:
            return None
        if self.last_swallow and self.last_swallow > last_done:
            return max(self.last_swallow + self.cfg.pace.swallow_offset_s, last_done + self.cfg.pace.min_interval_s)
        return last_done + self.interval_s

    def status(self) -> dict:
        return {"interval_s": round(self.interval_s, 1), "learned": self.learned, "n": self.n,
                "label": f"learned pace: every {self.interval_s:.0f} s" if self.learned
                else f"pace: every {self.interval_s:.0f} s (default)",
                "envelope": round(self.envelope)}
