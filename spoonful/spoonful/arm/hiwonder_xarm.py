"""ArmBackend over the `xarm` PyPI package (verified against xarm 0.0.4 source).

Quirks of 0.0.4 (see NOTES.md):
- getPosition(int) works; a multi read needs a list of xarm.Servo objects (not ints) and fills them in place.
- Controller._recv() indexes the HID report without checking it is non-empty: an empty 50 ms read raises
  IndexError. We retry and report None instead.
- Its degree helpers assume 250 deg over 1000 units. We only ever pass integer units.
- setPosition(list) uses count=len(list) in one CMD_SERVO_MOVE (0x03) frame; servoOff uses CMD 0x14 (20).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from .backend import ArmBackend, BackendError


def _s16(v: int) -> int:
    """Just past the 0 end a servo reports e.g. 65533 for -3: decode as signed 16-bit."""
    return v - 65536 if v > 32767 else v


class XarmBackend(ArmBackend):
    name = "xarm"

    def __init__(self, port: str = "USB", retries: int = 2):
        self.port = port
        self.retries = retries
        self.c = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            import xarm
            self._xarm = xarm
            self.c = xarm.Controller(self.port)
        except Exception as e:
            raise BackendError(f"xarm.Controller({self.port!r}) failed: {e!r}") from e

    def close(self) -> None:
        try:
            if self.c is not None:
                self.c._device.close()
        except Exception:
            pass

    def read_all(self, ids):
        servos = [self._xarm.Servo(i) for i in ids]
        with self._lock:
            for _ in range(self.retries + 1):
                try:
                    self.c.getPosition(servos)
                    return {s.servo_id: _s16(int(s.position)) for s in servos}
                except Exception:
                    time.sleep(0.005)
            # fall back to single reads so one dead servo doesn't blind us to the rest
            out: dict[int, Optional[int]] = {}
            for i in ids:
                try:
                    out[i] = _s16(int(self.c.getPosition(i)))
                except Exception:
                    out[i] = None
            return out

    def move(self, pairs, duration_ms):
        pairs = [(int(i), int(max(0, min(1000, u)))) for i, u in pairs]
        with self._lock:
            try:
                self.c.setPosition(pairs, duration=int(duration_ms), wait=False)
            except Exception as e:
                raise BackendError(f"setPosition failed: {e!r}") from e

    def unload(self, ids):
        with self._lock:
            try:
                self.c.servoOff([int(i) for i in ids])
            except Exception as e:
                raise BackendError(f"servoOff failed: {e!r}") from e

    def voltage(self):
        with self._lock:
            for _ in range(self.retries + 1):
                try:
                    v = self.c.getBatteryVoltage()
                    if v is not None:
                        return v
                except Exception:
                    time.sleep(0.005)
        return None
