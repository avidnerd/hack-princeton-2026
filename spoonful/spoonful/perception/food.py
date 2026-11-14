"""Optional VLM food check (Saturday evening, only if everything else passes).

Every interval_s in LISTENING, crop bowl_roi and ask Gemini for {spoon_loaded, bowl_empty, sweet_side}.
Verified against google-genai 3.2.0: genai.Client(api_key=...).models.generate_content(model=, contents=[
types.Part.from_bytes(data=, mime_type=), prompt], config=types.GenerateContentConfig(response_mime_type=...)).
No key / no network -> the module reports disabled and nothing else changes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

from ..config import Config

PROMPT = ("You are looking at a bowl of food in front of a feeding robot. Answer only JSON with keys "
          '"bowl_empty" (true if no food is left that a spoon could pick up), "spoon_loaded" (true if the spoon '
          'visible has food on it, null if no spoon visible), "sweet_side" ("left", "right" or null if one bowl).')


class FoodWatcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = cfg.food.enabled and bool(os.environ.get("GEMINI_API_KEY"))
        self.last: dict = {}
        self.empty_count = 0
        self._last_t = 0.0
        self._busy = False
        self.client = None
        if self.enabled:
            try:
                from google import genai
                self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            except Exception as e:
                self.enabled = False
                self.last = {"error": repr(e)}

    def maybe_check(self, image, now: float) -> None:
        if not self.enabled or self._busy or now - self._last_t < self.cfg.food.interval_s or image is None:
            return
        self._last_t = now
        self._busy = True
        threading.Thread(target=self._check, args=(image.copy(),), daemon=True).start()

    def _check(self, image) -> None:
        try:
            import cv2
            from google.genai import types
            x, y, w, h = self.cfg.food.bowl_roi
            crop = image[y:y + h, x:x + w]
            ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            resp = self.client.models.generate_content(
                model=self.cfg.food.model,
                contents=[types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            d = json.loads(resp.text)
            d["ts"] = time.time()
            self.last = d
            self.empty_count = self.empty_count + 1 if d.get("bowl_empty") else 0
        except Exception as e:
            self.last = {"error": repr(e), "ts": time.time()}
        finally:
            self._busy = False

    @property
    def bowl_empty(self) -> bool:
        return self.empty_count >= 2

    def status(self) -> dict:
        return {"enabled": self.enabled, **self.last, "bowl_empty_confirmed": self.bowl_empty}
