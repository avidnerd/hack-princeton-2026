"""Watchdog thread (20 Hz). Independent of the FSM so it still acts when the FSM is wedged.

- no FSM tick for 250 ms           -> arm.hold(), then a slow park (park is exempt from the heartbeat guard)
- no camera frame for 500 ms while the arm is moving in a bite      -> FSM fault camera_dropout (hold + park)
- no arm position read for 1 s while torque is on                    -> hold, FSM fault arm_read
Every trip is logged with its cause. The arm thread separately refuses setpoints without a fresh heartbeat.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from ..arm.arm import Arm
from ..log import Log


class Watchdog:
    FSM_TIMEOUT_S = 0.25
    CAMERA_TIMEOUT_S = 0.5
    ARM_READ_TIMEOUT_S = 1.0     # one joint per tick: a joint's reading is up to ~0.25 s old by design

    def __init__(self, fsm, arm: Arm, camera, log: Log):
        self.fsm, self.arm, self.camera, self.log = fsm, arm, camera, log
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.trips: list[tuple[float, str]] = []
        self._fsm_tripped = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _trip(self, cause: str) -> None:
        self.trips.append((time.time(), cause))
        self.log.fault("watchdog", cause=cause)

    def check(self, now: float) -> Optional[str]:
        from .fsm import BITE_STATES, S
        if now - self.fsm.last_tick > self.FSM_TIMEOUT_S:
            if not self._fsm_tripped:
                self._fsm_tripped = True
                self._trip("fsm_stalled")
                self.arm.hold("watchdog fsm")
                if self.arm.torque_on:
                    self.arm.park()
            return "fsm_stalled"
        self._fsm_tripped = False
        moving = self.arm.busy and self.fsm.state in BITE_STATES
        if moving and self.camera is not None and self.camera.age_s() > self.CAMERA_TIMEOUT_S:
            self._trip("camera_dropout")
            self.arm.hold("watchdog camera")
            self.fsm.post("fault", kind="camera_dropout")
            return "camera_dropout"
        if self.arm.torque_on and self.arm.read_age_s() > self.ARM_READ_TIMEOUT_S and self.fsm.state != S.FAULT:
            self._trip("arm_read")
            self.arm.hold("watchdog arm read")
            self.fsm.post("fault", kind="arm_read")
            return "arm_read"
        return None

    def _loop(self) -> None:
        time.sleep(1.0)  # let everything start
        while not self._stop.is_set():
            try:
                self.check(time.monotonic())
            except Exception as e:
                self.log.fault("watchdog_exception", error=repr(e))
            time.sleep(0.05)
