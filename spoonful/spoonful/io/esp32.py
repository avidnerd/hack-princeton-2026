"""ESP32-S3 serial peripheral: arcade STOP button + piezo swallow envelope. Newline JSON at 115200:
  {"btn":1} press, {"btn":0} release, {"piezo":<envelope>} at 20 Hz, {"hb":1} every 1 s.
Port auto-detect by Espressif USB VID (0x303A), a usbmodem/usbserial name, or (Windows) a COM port whose description
mentions USB (CP210x / CH340 bridges); set esp32.port in config.yaml to skip detection. Absent device = feature disabled,
never a crash; reconnects every 2 s.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

from ..config import Esp32Cfg

ESPRESSIF_VID = 0x303A


def find_port() -> Optional[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid == ESPRESSIF_VID:
            return p.device
    for p in ports:
        if any(k in p.device for k in ("usbmodem", "usbserial", "ttyACM", "ttyUSB")):
            return p.device
        if p.device.upper().startswith("COM") and any(k in (p.description or "")
                                                      for k in ("USB", "CP210", "CH340", "CH910")):
            return p.device
    return None


class Esp32:
    def __init__(self, cfg: Esp32Cfg, on_button: Callable[[], None], on_piezo: Callable[[float], None]):
        self.cfg = cfg
        self.on_button, self.on_piezo = on_button, on_piezo
        self.port: Optional[str] = None
        self.connected = False
        self.last_hb = 0.0
        self.last_envelope = 0.0
        self.presses = 0
        self._stop = threading.Event()

    def start(self) -> None:
        if self.cfg.enabled:
            threading.Thread(target=self._loop, name="esp32", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def handle_line(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except (ValueError, TypeError):
            return
        if not isinstance(msg, dict):
            return
        if msg.get("btn") == 1:
            self.presses += 1
            self.on_button()
        if "piezo" in msg:
            self.last_envelope = float(msg["piezo"])
            self.on_piezo(self.last_envelope)
        if "hb" in msg:
            self.last_hb = time.time()

    def _loop(self) -> None:
        import serial
        while not self._stop.is_set():
            port = self.cfg.port if self.cfg.port != "auto" else find_port()
            if not port:
                self.connected = False
                time.sleep(2)
                continue
            try:
                with serial.Serial(port, self.cfg.baud, timeout=0.5) as s:
                    self.port, self.connected = port, True
                    self.last_hb = time.time()
                    while not self._stop.is_set():
                        raw = s.readline()
                        if raw:
                            self.handle_line(raw.decode(errors="ignore").strip())
                        if time.time() - self.last_hb > self.cfg.heartbeat_timeout_s:
                            break   # silent device: reopen
            except Exception:
                pass
            self.connected = False
            time.sleep(2)

    def status(self) -> dict:
        return {"enabled": self.cfg.enabled, "connected": self.connected, "port": self.port,
                "presses": self.presses, "envelope": round(self.last_envelope),
                "hb_age_s": round(time.time() - self.last_hb, 1) if self.last_hb else None}
