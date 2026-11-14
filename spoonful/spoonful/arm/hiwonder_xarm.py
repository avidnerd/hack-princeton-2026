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

    CMD_GET_POSITION = 21
    READ_TIMEOUT_S = 0.15
    SINGLE_TIMEOUT_S = 0.06

    @property
    def round_robin(self) -> bool:
        """The Arm reads one joint per tick over USB HID (see _read_positions_hid)."""
        return self.c is not None and not getattr(self.c, "_is_serial", True)

    def read_one(self, sid: int) -> Optional[int]:
        with self._lock:
            try:
                r = self._read_positions_hid([int(sid)], self.SINGLE_TIMEOUT_S)
            except Exception as e:
                raise BackendError(f"read servo {sid}: {e!r}") from e
        return r[int(sid)] if r else None

    def _read_positions_hid(self, ids, timeout_s: Optional[float] = None) -> Optional[dict[int, int]]:
        """getPosition, done properly over USB HID. The library waits only 50 ms for the reply and then fails; on
        this controller a 6-servo reply often takes longer (about 45% of reads failed, 2026-11-14), and that late
        reply was then taken as the answer to the NEXT request. Here: drain stale reports, send, wait up to
        READ_TIMEOUT_S for the reply to THIS command, and map positions by the servo ids in the reply."""
        dev = self.c._device
        while dev.read(64, 0):                       # drop late replies to earlier requests
            pass
        self.c._send(self.CMD_GET_POSITION, bytearray([len(ids), *ids]))
        deadline = time.monotonic() + (timeout_s or self.READ_TIMEOUT_S)
        while time.monotonic() < deadline:
            rep = dev.read(64, 10)
            if not rep or len(rep) < 5 or rep[0] != 0x55 or rep[1] != 0x55 or rep[3] != self.CMD_GET_POSITION:
                continue
            data = rep[4:4 + rep[2]]
            n = data[0] if data else 0
            out = {}
            for k in range(n):
                sid, lo, hi = data[1 + 3 * k], data[2 + 3 * k], data[3 + 3 * k]
                out[sid] = _s16(hi * 256 + lo)
            if all(i in out for i in ids):
                return out
        return None

    def read_all(self, ids):
        ids = [int(i) for i in ids]
        with self._lock:
            if not getattr(self.c, "_is_serial", True):
                for _ in range(self.retries + 1):
                    try:
                        r = self._read_positions_hid(ids)
                    except Exception:
                        r = None
                    if r is not None:
                        return {i: r[i] for i in ids}
                return {i: None for i in ids}
            servos = [self._xarm.Servo(i) for i in ids]
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
            # One servo per command: this controller ignored a multi-servo servoOff([6,5,4,3,2]) (arm stayed stiff,
            # 2026-11-14) but obeys single ones. Gripper not in `ids` keeps holding the spoon.
            try:
                for i in ids:
                    self.c.servoOff(int(i))
                    time.sleep(0.02)
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
