"""The GO button, heard by Windows itself.

The arcade button is a Bluetooth keyboard that types PageDown. The browser page only hears it while that page has
the keyboard: click anywhere else (another window, the address bar, a text box) and presses are lost. This asks
Windows for the key's state directly, so a press counts whichever window is in front.

Only PageDown / PageUp (what the button and presenter clickers send). Enter stays a page-only shortcut: it is typed
all the time for other reasons.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional

VK_PRIOR, VK_NEXT = 0x21, 0x22          # PageUp, PageDown
POLL_S = 0.02
MIN_GAP_S = 0.25                        # one press, however the switch bounces


class GoKey:
    def __init__(self, on_press: Callable[[], None], state_fn: Optional[Callable[[int], bool]] = None):
        self.on_press = on_press
        self._state = state_fn or self._windows_state()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.presses = 0
        self.last_at = 0.0

    @staticmethod
    def _windows_state() -> Optional[Callable[[int], bool]]:
        if sys.platform != "win32":
            return None
        import ctypes
        get = ctypes.windll.user32.GetAsyncKeyState
        return lambda vk: bool(get(vk) & 0x8000)

    @property
    def available(self) -> bool:
        return self._state is not None

    def start(self) -> None:
        if self._state is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="go-key", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll(self, was_down: bool, now: float) -> bool:
        """One look at the keys. Returns whether a key is down; calls on_press on the way down."""
        down = self._state(VK_NEXT) or self._state(VK_PRIOR)
        if down and not was_down and now - self.last_at >= MIN_GAP_S:
            self.last_at = now
            self.presses += 1
            self.on_press()
        return down

    def _run(self) -> None:
        down = True                     # a key held while starting up is not a press
        while not self._stop.wait(POLL_S):
            try:
                down = self.poll(down, time.monotonic())
            except Exception:
                down = False
