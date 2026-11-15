"""ArmBackend: the only layer that talks to hardware. Units are raw servo units (0..1000), ids are servo ids.

Implementations: SimBackend (here), XarmBackend (hiwonder_xarm.py), RawHidBackend (hiwonder_raw.py).
Nothing in here enforces safety; arm.py does, before anything reaches a backend.
"""
from __future__ import annotations

import abc
import math
import random
import threading
import time
from typing import Callable, Optional


class BackendError(RuntimeError):
    pass


class ArmBackend(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def connect(self) -> None: ...

    def close(self) -> None:
        pass

    @abc.abstractmethod
    def read_all(self, ids: list[int]) -> dict[int, Optional[int]]:
        """One position per id, None where the read failed."""

    @abc.abstractmethod
    def move(self, pairs: list[tuple[int, int]], duration_ms: int) -> None:
        """Command each servo to reach units over duration_ms. Must not block for the duration."""

    @abc.abstractmethod
    def unload(self, ids: list[int]) -> None:
        """Torque off."""

    @abc.abstractmethod
    def voltage(self) -> Optional[float]: ...


class SimBackend(ArmBackend):
    """Servo simulator: first-order tracking of the commanded target with a speed cap and ±1 unit pot noise.

    Hooks for tests and dev: `hand_move(id, units)` moves an unloaded servo (teach in sim),
    `jam(id)` / `block(id, units)` simulate contact (the servo cannot pass that position), `fail_reads` simulates HID loss.
    `sag(id, units, lag_s)` makes a servo settle short of its command and/or trail it; `obstruct(top_mm, ...)` puts a
    solid in the way of the spoon tip (a cup rim, the food, the bottom). Both are off until called.
    """
    name = "sim"

    def __init__(self, ids: list[int], start_units: Optional[dict[int, int]] = None,
                 max_units_s: float = 900.0, noise: int = 1, seed: int = 0):
        self.ids = ids
        self.pos = {i: float((start_units or {}).get(i, 500)) for i in ids}
        self._seg: dict[int, tuple[float, float, float, float]] = {}  # id -> (t0, p0, t1, p1)
        self.loaded = {i: False for i in ids}
        self.max_units_s = max_units_s
        self.noise = noise
        self.rng = random.Random(seed)
        self.blocks: dict[int, float] = {}
        self.jammed: set[int] = set()
        self.fail_reads = False
        self.sags: dict[int, float] = {}            # id -> units the servo settles away from its command
        self.lags: dict[int, float] = {}            # id -> time constant (s) it trails a moving command by
        self.obstructions: list[tuple[float, float, float, float, float]] = []   # (top, x, y, r_min, r_max) mm
        self.tip_fn: Optional[Callable[[dict[int, float]], tuple[float, float, float]]] = None  # units -> tip mm
        self.volts = 7.4
        self.sent: list[tuple[float, list[tuple[int, int]], int]] = []
        self._lock = threading.Lock()
        self._last = time.monotonic()

    def connect(self) -> None:
        pass

    def _advance(self) -> None:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        moved = dict(self.pos)
        for i in self.ids:
            if not self.loaded[i] or i not in self._seg or i in self.jammed:
                continue
            t0, p0, t1, p1 = self._seg[i]
            goal = p1 if now >= t1 else p0 + (p1 - p0) * (now - t0) / max(1e-6, t1 - t0)
            want = goal + self.sags.get(i, 0.0) - self.pos[i]
            if self.lags.get(i, 0.0) > 0:
                want *= 1.0 - math.exp(-dt / self.lags[i])
            step = max(-self.max_units_s * dt, min(self.max_units_s * dt, want))
            new = self.pos[i] + step
            if i in self.blocks:
                b = self.blocks[i]
                if (self.pos[i] <= b <= new) or (new <= b <= self.pos[i]):
                    new = b
            moved[i] = new
        self.pos = self._stop_at_solid(moved) if self.obstructions and self.tip_fn else moved

    def _in_solid(self, pos: dict[int, float]) -> bool:
        x, y, z = self.tip_fn(pos)
        return any(z < top and r0 <= math.hypot(x - ox, y - oy) <= r1 for top, ox, oy, r0, r1 in self.obstructions)

    def _stop_at_solid(self, new: dict[int, float]) -> dict[int, float]:
        """The tip cannot enter an obstruction: the whole arm stops at the last point of this step that is still
        outside (all joints together, a simplification). A tip that is already inside one may only rise."""
        cur = self.pos

        def at(f: float) -> dict[int, float]:
            return {i: cur[i] + (new[i] - cur[i]) * f for i in cur}

        if self._in_solid(cur):
            return new if self.tip_fn(new)[2] > self.tip_fn(cur)[2] else cur
        n = max(1, math.ceil(max(abs(new[i] - cur[i]) for i in cur)))   # walk the step a servo unit at a time: a
        hit = next((k for k in range(1, n + 1) if self._in_solid(at(k / n))), None)   # servo catching up must not
        if hit is None:                                                 # jump through a thin wall
            return new
        lo, hi = (hit - 1) / n, hit / n
        for _ in range(10):
            mid = (lo + hi) / 2
            if self._in_solid(at(mid)):
                hi = mid
            else:
                lo = mid
        return at(lo)

    def read_all(self, ids):
        with self._lock:
            if self.fail_reads:
                return {i: None for i in ids}
            self._advance()
            return {i: int(round(self.pos[i] + self.rng.randint(-self.noise, self.noise))) for i in ids}

    def move(self, pairs, duration_ms):
        with self._lock:
            self._advance()
            now = time.monotonic()
            self.sent.append((time.time(), list(pairs), duration_ms))
            if len(self.sent) > 5000:
                del self.sent[:1000]
            for i, u in pairs:
                self.loaded[i] = True
                self._seg[i] = (now, self.pos[i], now + duration_ms / 1000.0, float(u))

    def unload(self, ids):
        with self._lock:
            self._advance()
            for i in ids:
                self.loaded[i] = False
                self._seg.pop(i, None)

    def voltage(self):
        return self.volts

    # --- test/dev hooks ---
    def hand_move(self, sid: int, units: float) -> None:
        with self._lock:
            if self.loaded[sid]:
                return  # a loaded servo resists the hand
            self.pos[sid] = float(units)

    def jam(self, sid: int, on: bool) -> None:
        """Simulated stall: the servo cannot move at all (commanded-vs-actual error grows -> contact)."""
        with self._lock:
            self._advance()
            (self.jammed.add if on else self.jammed.discard)(sid)

    def block(self, sid: int, units: Optional[float]) -> None:
        with self._lock:
            if units is None:
                self.blocks.pop(sid, None)
            else:
                self.blocks[sid] = float(units)

    def sag(self, sid: int, units: float = 0.0, lag_s: float = 0.0) -> None:
        """Simulated load: the servo settles `units` away from its command (signed, steady) and trails a moving
        command with time constant `lag_s` (lag = speed x lag_s). sag(sid) switches both off."""
        with self._lock:
            self._advance()
            self.sags[sid], self.lags[sid] = float(units), float(lag_s)
            if not units:
                self.sags.pop(sid)
            if lag_s <= 0:
                self.lags.pop(sid)

    def obstruct(self, top_mm: Optional[float], x: float = 0.0, y: float = 0.0, r_min: float = 0.0,
                 r_max: Optional[float] = None) -> None:
        """Simulated solid the spoon TIP cannot enter: everything below top_mm (table frame, mm) whose horizontal
        distance from (x, y) is within [r_min, r_max]; r_max None = everywhere (a floor). A cup rim is a ring
        (r_min = opening radius, r_max = outer radius, top = rim height); food or the bottom is a disc (r_min 0).
        Several can be set; obstruct(None) removes them all. Needs tip_fn (Arm sets it). Only the tip is modelled,
        not the rest of the spoon."""
        with self._lock:
            self._advance()
            if top_mm is None:
                self.obstructions.clear()
            else:
                self.obstructions.append((float(top_mm), float(x), float(y), float(r_min),
                                          math.inf if r_max is None else float(r_max)))


def make_backend(kind: str, ids: list[int], start_units: Optional[dict[int, int]] = None) -> ArmBackend:
    if kind == "sim":
        return SimBackend(ids, start_units)
    if kind == "xarm":
        from .hiwonder_xarm import XarmBackend
        return XarmBackend()
    if kind == "raw":
        from .hiwonder_raw import RawHidBackend
        return RawHidBackend()
    raise ValueError(f"unknown arm backend {kind!r}")
