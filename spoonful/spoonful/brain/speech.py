"""Speech policy (R8): the arm says only these lines, only at state changes the diner needs.

No praise, no narration, never anything about perception. 3 s cooldown, never the same line twice in a row
("Stopped" bypasses the cooldown: it confirms a safety action).
Backends: ElevenLabs with every line cached to disk at startup (no network during the demo), or the OS voice
(`say`: macOS `say`, Windows SAPI through PowerShell, Linux espeak/spd-say). Cached mp3s play with afplay on
macOS, PowerShell MediaPlayer on Windows, ffplay/mpg123 elsewhere.
Verified against elevenlabs 2.77.0: ElevenLabs(api_key=).text_to_speech.convert(voice_id, text=, model_id=,
output_format=) -> Iterator[bytes].
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ..config import Config

LINES = {
    "ready": "Ready when you are",
    "here": "Here you go",
    "wait": "I'll wait",
    "paused": "Paused",
    "break": "Taking a break",
    "last": "That's the last one",
    "stopped": "Stopped",
}
BYPASS_COOLDOWN = {"stopped"}

WINDOWS = sys.platform == "win32"
# Text and paths go through environment variables so quotes in the lines ("I'll wait") never need escaping.
_PS_SPEAK = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             "$s.Rate = [int]$env:SPOONFUL_RATE; $s.Speak($env:SPOONFUL_TEXT)")
_PS_PLAY = ("Add-Type -AssemblyName PresentationCore; $p = New-Object System.Windows.Media.MediaPlayer; "
            "$p.Open([uri]$env:SPOONFUL_FILE); $p.Play(); $i = 0; "
            "while (-not $p.NaturalDuration.HasTimeSpan -and $i -lt 40) { Start-Sleep -Milliseconds 50; $i++ }; "
            "if ($p.NaturalDuration.HasTimeSpan) { Start-Sleep -Milliseconds ($p.NaturalDuration.TimeSpan.TotalMilliseconds + 100) }")


def _powershell() -> Optional[str]:
    return shutil.which("powershell") or shutil.which("pwsh")


def tts_command(text: str, voice: str, rate: float, platform: str = sys.platform) -> Optional[tuple[list[str], dict]]:
    """(argv, extra env) that speaks `text` with the OS voice, or None if this machine has none."""
    if platform == "darwin" and shutil.which("say"):
        return ["say", "-v", voice, "-r", str(int(175 * rate)), text], {}
    if platform == "win32" and _powershell():
        sapi_rate = max(-10, min(10, round((rate - 1.0) * 10)))      # SAPI: -10..10, 0 = normal
        return ([_powershell(), "-NoProfile", "-NonInteractive", "-Command", _PS_SPEAK],
                {"SPOONFUL_TEXT": text, "SPOONFUL_RATE": str(sapi_rate)})
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            return [exe, "-s", str(int(175 * rate)), text], {}
    if shutil.which("spd-say"):
        return ["spd-say", "-w", text], {}
    return None


def play_command(path: Path, platform: str = sys.platform) -> Optional[tuple[list[str], dict]]:
    """(argv, extra env) that plays an mp3, or None."""
    if platform == "darwin" and shutil.which("afplay"):
        return ["afplay", str(path)], {}
    if platform == "win32" and _powershell():
        return ([_powershell(), "-NoProfile", "-NonInteractive", "-Command", _PS_PLAY],
                {"SPOONFUL_FILE": str(Path(path).resolve())})
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)], {}
    if shutil.which("mpg123"):
        return ["mpg123", "-q", str(path)], {}
    return None


def has_os_voice() -> bool:
    return tts_command("x", "", 1.0) is not None


class Speaker:
    def __init__(self, cfg: Config, log=None):
        self.cfg = cfg
        self.log = log
        self.backend = cfg.speech.backend
        self.cache_dir = cfg.path(cfg.speech.cache_dir)
        self.last_key: Optional[str] = None
        self.last_t = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.muted_fn: Callable[[], bool] = lambda: False   # profile.speech == False
        self.history: list[tuple[float, str]] = []
        if self.backend == "say" and not has_os_voice():
            self.backend = "none"

    # ---- cache ----
    def _cache_path(self, key: str) -> Path:
        s = self.cfg.speech
        h = hashlib.sha1(f"{LINES[key]}|{s.elevenlabs_voice_id}|{s.elevenlabs_model}".encode()).hexdigest()[:10]
        return self.cache_dir / f"{key}_{h}.mp3"

    def prewarm(self) -> dict:
        """Generate every line once. Returns {key: 'cached'|'generated'|'error ...'}; falls back to `say`."""
        if self.backend != "elevenlabs":
            return {}
        key_env = os.environ.get("ELEVENLABS_API_KEY")
        out = {}
        missing = [k for k in LINES if not self._cache_path(k).exists()]
        if missing and not key_env:
            self.backend = "say" if has_os_voice() else "none"
            return {"error": "ELEVENLABS_API_KEY not set; using say"}
        client = None
        for k in LINES:
            p = self._cache_path(k)
            if p.exists():
                out[k] = "cached"
                continue
            try:
                if client is None:
                    from elevenlabs.client import ElevenLabs
                    client = ElevenLabs(api_key=key_env)
                audio = client.text_to_speech.convert(
                    self.cfg.speech.elevenlabs_voice_id, text=LINES[k],
                    model_id=self.cfg.speech.elevenlabs_model, output_format="mp3_44100_128")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"".join(audio))
                out[k] = "generated"
            except Exception as e:
                out[k] = f"error {e!r}"
        if any(v.startswith("error") for v in out.values()):
            self.backend = "say" if has_os_voice() else "none"
        return out

    # ---- policy ----
    def allowed(self, key: str, now: float) -> bool:
        if key not in LINES or not self.cfg.speech.enabled or self.muted_fn():
            return False
        if key == self.last_key:
            return False
        if key not in BYPASS_COOLDOWN and now - self.last_t < self.cfg.speech.cooldown_s:
            return False
        return True

    def say(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if not self.allowed(key, now):
                return False
            self.last_key, self.last_t = key, now
            self.history.append((time.time(), LINES[key]))
            self._play(key)
        if self.log:
            self.log.event("speech", detail_line=LINES[key])
        return True

    def _play(self, key: str) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        cmd = None
        if self.backend == "elevenlabs" and self._cache_path(key).exists():
            cmd = play_command(self._cache_path(key))
        if cmd is None and self.backend in ("say", "elevenlabs"):
            cmd = tts_command(LINES[key], self.cfg.speech.voice, self.cfg.speech.rate)
        if cmd is None:
            return
        argv, env = cmd
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {}
        try:
            self._proc = subprocess.Popen(argv, env={**os.environ, **env}, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, **kw)
        except OSError:
            pass
