"""ArmBackend speaking the Hiwonder HID protocol directly (fallback if the xarm package misbehaves).

Output report: [report_id=0, 0x55, 0x55, len(payload)+2, cmd, *payload]
  CMD_SERVO_MOVE=3            payload: count, time_lo, time_hi, (id, pos_lo, pos_hi) * count
  CMD_GET_BATTERY_VOLTAGE=15  reply payload: mv_lo, mv_hi
  CMD_MULT_SERVO_UNLOAD=20    payload: count, id * count
  CMD_MULT_SERVO_POS_READ=21  payload: count, id * count ; reply: count, (id, pos_lo, pos_hi) * count
"""
from __future__ import annotations

import threading
from typing import Optional

from .backend import ArmBackend, BackendError

VID, PID = 0x0483, 0x5750
MOVE, VOLT, UNLOAD, POSREAD = 3, 15, 20, 21


class RawHidBackend(ArmBackend):
    name = "raw"

    def __init__(self, timeout_ms: int = 60):
        self.timeout_ms = timeout_ms
        self.d = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            import hid
            self.d = hid.device()
            self.d.open(VID, PID)
            self.d.set_nonblocking(0)
        except Exception as e:
            raise BackendError(f"HID open {VID:04x}:{PID:04x} failed: {e!r}") from e

    def close(self) -> None:
        try:
            self.d.close()
        except Exception:
            pass

    def _send(self, cmd: int, payload: list[int] = ()) -> None:
        try:
            self.d.write([0, 0x55, 0x55, len(payload) + 2, cmd, *payload])
        except Exception as e:
            raise BackendError(f"HID write failed: {e!r}") from e

    def _recv(self, cmd: int) -> Optional[list[int]]:
        for _ in range(3):  # skip stale replies to other commands
            r = self.d.read(64, self.timeout_ms)
            if not r:
                return None
            if len(r) >= 4 and r[0] == 0x55 and r[1] == 0x55 and r[3] == cmd:
                return list(r[4:4 + r[2] - 2])
        return None

    def read_all(self, ids):
        with self._lock:
            self._send(POSREAD, [len(ids), *ids])
            r = self._recv(POSREAD)
        out: dict[int, Optional[int]] = {i: None for i in ids}
        if r:
            n = r[0]
            for k in range(n):
                base = 1 + 3 * k
                if base + 2 < len(r):
                    out[r[base]] = r[base + 1] | (r[base + 2] << 8)
        return out

    def move(self, pairs, duration_ms):
        ms = int(duration_ms)
        payload = [len(pairs), ms & 0xFF, (ms >> 8) & 0xFF]
        for i, u in pairs:
            u = int(max(0, min(1000, u)))
            payload += [int(i), u & 0xFF, (u >> 8) & 0xFF]
        with self._lock:
            self._send(MOVE, payload)

    def unload(self, ids):
        with self._lock:
            self._send(UNLOAD, [len(ids), *[int(i) for i in ids]])

    def voltage(self):
        with self._lock:
            self._send(VOLT)
            r = self._recv(VOLT)
        return None if not r or len(r) < 2 else (r[0] | (r[1] << 8)) / 1000.0
