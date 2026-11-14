"""Voice commands: wake phrase + command, or push-to-talk from the UI (no wake phrase needed).

Mic -> energy VAD -> utterance (<= 4 s) -> faster-whisper (local, preloaded) or OpenAI -> parse_command.
Verified against faster-whisper 1.2.1: WhisperModel(size, device="cpu", compute_type="int8");
model.transcribe(np.float32 array, language="en", beam_size=1) -> (segments iterator, info); Segment.avg_logprob.
Unknown or low-confidence -> ignored silently. "stop" is latched like the button (FSM side).
"""
from __future__ import annotations

import io
import math
import os
import queue
import re
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np

from ..config import Config, VoiceCfg


def normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_command(text: str, cfg: VoiceCfg, require_wake: bool = True) -> Optional[str]:
    """Pure. Returns a command key from cfg.commands, or None."""
    t = normalise(text)
    if not t:
        return None
    rest = t
    woke = False
    for alias in sorted(cfg.wake_aliases, key=len, reverse=True):
        i = t.find(alias)
        if i >= 0:
            woke, rest = True, t[i + len(alias):].strip()
            break
    if not woke:
        if cfg.bare_stop and re.search(r"\bstop\b", t):
            return "stop"
        if require_wake:
            return None
    # "stop" wins over everything if it appears at all
    if re.search(r"\bstop\b", rest):
        return "stop"
    best: Optional[tuple[int, str]] = None
    for cmd, phrases in cfg.commands.items():
        for ph in phrases:
            m = re.search(rf"\b{re.escape(normalise(ph))}\b", rest)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), cmd)
    return best[1] if best else None


class Transcriber:
    def __init__(self, cfg: VoiceCfg):
        self.cfg = cfg
        self.model = None
        if cfg.backend == "faster_whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel(cfg.model, device="cpu", compute_type="int8")

    def __call__(self, audio: np.ndarray) -> tuple[str, float]:
        if self.cfg.backend == "faster_whisper":
            segs, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                            condition_on_previous_text=False)
            segs = list(segs)
            if not segs:
                return "", 0.0
            text = " ".join(s.text for s in segs)
            conf = math.exp(sum(s.avg_logprob for s in segs) / len(segs))
            return text, conf
        return self._openai(audio)

    def _openai(self, audio: np.ndarray) -> tuple[str, float]:
        import httpx
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.cfg.sample_rate)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        r = httpx.post("https://api.openai.com/v1/audio/transcriptions",
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                       files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                       data={"model": "whisper-1", "language": "en"}, timeout=10)
        r.raise_for_status()
        return r.json().get("text", ""), 0.9


class VoiceListener:
    def __init__(self, cfg: Config, on_command: Callable[[str, str, float], None], log=None):
        self.cfg = cfg
        self.on_command = on_command
        self.log = log
        self.enabled = cfg.voice.enabled
        self.error: Optional[str] = None
        self.ptt = False
        self._q: "queue.Queue[tuple[np.ndarray, bool]]" = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self.last_text = ""
        self.transcriber: Optional[Transcriber] = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self.transcriber = Transcriber(self.cfg.voice)
            import sounddevice  # noqa: F401
        except Exception as e:
            self.error, self.enabled = f"voice disabled: {e!r}", False
            return
        threading.Thread(target=self._capture, name="voice-mic", daemon=True).start()
        threading.Thread(target=self._recognise, name="voice-asr", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def set_ptt(self, on: bool) -> None:
        self.ptt = on

    def _capture(self) -> None:
        import sounddevice as sd
        v = self.cfg.voice
        block = int(v.sample_rate * 0.05)
        buf: list[np.ndarray] = []
        speaking, silence, ptt_seen = False, 0.0, False
        try:
            with sd.InputStream(samplerate=v.sample_rate, channels=1, dtype="float32", blocksize=block) as st:
                while not self._stop.is_set():
                    x, _ = st.read(block)
                    x = x[:, 0]
                    loud = float(np.sqrt(np.mean(x ** 2))) > v.vad_rms
                    if self.ptt:
                        ptt_seen = True
                        buf.append(x)
                        continue
                    if ptt_seen:      # PTT released: send everything recorded while held
                        self._emit(buf, True)
                        buf, ptt_seen, speaking = [], False, False
                        continue
                    if loud:
                        speaking, silence = True, 0.0
                        buf.append(x)
                    elif speaking:
                        buf.append(x)
                        silence += 0.05
                        if silence >= 0.6:
                            self._emit(buf, False)
                            buf, speaking = [], False
                    if len(buf) * 0.05 > 4.0:
                        self._emit(buf, False)
                        buf, speaking = [], False
        except Exception as e:
            self.error = f"mic error: {e!r}"

    def _emit(self, buf: list[np.ndarray], ptt: bool) -> None:
        if len(buf) * 0.05 < 0.3:
            return
        try:
            self._q.put_nowait((np.concatenate(buf), ptt))
        except queue.Full:
            pass

    def _recognise(self) -> None:
        while not self._stop.is_set():
            try:
                audio, ptt = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text, conf = self.transcriber(audio)
            except Exception as e:
                self.error = f"asr error: {e!r}"
                continue
            self.last_text = text
            cmd = parse_command(text, self.cfg.voice, require_wake=not ptt)
            if cmd and conf < self.cfg.voice.min_confidence and cmd != "stop":
                cmd = None
            if self.log:
                self.log.voice(text, cmd, round(conf, 3))
            if cmd:
                self.on_command(cmd, text, conf)

    def status(self) -> dict:
        return {"enabled": self.enabled, "error": self.error, "ptt": self.ptt, "last_text": self.last_text}
