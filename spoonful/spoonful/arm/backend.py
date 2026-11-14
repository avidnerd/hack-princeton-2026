"""ArmBackend: the only layer that talks to hardware. Units are raw servo units (0..1000), ids are servo ids.

Implementations: SimBackend (here), XarmBackend (hiwonder_xarm.py), RawHidBackend (hiwonder_raw.py).
Nothing in here enforces safety; arm.py does, before anything reaches a backend.
"""
from __future__ import annotations

import abc
import random
import threading
import time
from typing import Optional


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
        for i in self.ids:
            if not self.loaded[i] or i not in self._seg or i in self.jammed:
                continue
            t0, p0, t1, p1 = self._seg[i]
            goal = p1 if now >= t1 else p0 + (p1 - p0) * (now - t0) / max(1e-6, t1 - t0)
            step = max(-self.max_units_s * dt, min(self.max_units_s * dt, goal - self.pos[i]))
            new = self.pos[i] + step
            if i in self.blocks:
                b = self.blocks[i]
                if (self.pos[i] <= b <= new) or (new <= b <= self.pos[i]):
                    new = b
            self.pos[i] = new

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
