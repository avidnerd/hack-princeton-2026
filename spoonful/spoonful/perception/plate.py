"""Plate awareness: what food is in which bowl, and which bowl a spoken request means.

Bowls sit at fixed, taught spots (config `plate.bowls`, each with an image box `roi` and a taught scoop path). Every
`interval_s` the watcher crops each bowl from the plate camera and asks Gemini what is in it and how much is left. A
request such as "Spoonful, a bite of the applesauce" is matched against those labels to pick the bowl; the FSM then
runs that bowl's taught scoop. Vision decides WHAT and WHERE; the arm only ever runs taught motion.

No key / no network / vision error: the labels fall back to `label` in config or to labels the operator typed in the
UI, and everything else keeps working. Models are tried in order (`model`, then `fallback_models`) with a per-call
timeout; the one that last worked is tried first. Verified against google-genai 3.2.0.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Callable, Optional

from ..arm.scoop_style import TEXTURES, texture_from_name
from .table import TableCalib
from ..config import Config

PROMPT = (
    "You are the eyes of a feeding robot. Each image is a close-up of one spot on the table, labelled with its id. "
    "First decide whether the image clearly shows a bowl, cup or container with food in it. If it does not (an "
    "empty table, a phone, a hand, clothing, an empty container, or you are unsure), give food null and amount "
    "\"empty\". Do not guess. Only if you clearly see food: name it in 1 to 3 plain lowercase words a person would "
    "say when asking for it (for example \"applesauce\", \"rice\", \"yogurt\"), say how much a spoon could still "
    "pick up: \"full\", \"half\", \"low\" or \"empty\", and how hard a spoon must push to scoop it, as "
    "\"texture\": \"liquid\" (soup, broth), \"soft\" (applesauce, yogurt, pudding), \"thick\" (mashed potatoes, "
    "oatmeal, rice) or \"firm\" (jello, tofu, banana, pasta). Answer only JSON: a list of objects with keys "
    "\"bowl\", \"food\", \"amount\", \"texture\"."
)
AMOUNTS = ("full", "half", "low", "empty")
BOWL_TOUCH_PROMPT = (
    "A robot arm's spoon is dipped into one bowl, cup or container of food in this picture. Return the box of THAT "
    "container (the one the spoon is in) and the food in it. Answer only JSON {\"box_2d\": [ymin, xmin, ymax, xmax] "
    "normalised to 0-1000, \"food\": 1-3 lowercase words or null}, or {\"box_2d\": null} if you can't see it."
)
TIP_PROMPT = (
    "A white plastic spoon is taped to the gripper of a small teal robot arm. Point to the very tip of that spoon: "
    "the end of the spoon's bowl farthest from the gripper. Only if you can clearly see that spoon tip, answer "
    "{\"point\": [y, x]} with coordinates normalised to 0-1000. If the tip is outside the picture, hidden, or you "
    "are not sure which object it is, answer {\"point\": null}. Answer only JSON."
)
SPOON_PROMPT = (
    "A small robot arm is holding a spoon out toward a person. Look at the spoon's bowl. Answer only JSON: "
    "{\"spoon_visible\": true or false, \"food_on_spoon\": true or false, \"amount\": \"none\", \"little\" "
    "or \"good\"}."
)
LOCATE_PROMPT = (
    "Find every bowl, cup or container with food in this picture of a table. For each one return "
    "{\"food\": a 1 to 3 word lowercase name or null, \"box_2d\": [ymin, xmin, ymax, xmax]} with coordinates "
    "normalised to 0-1000. Answer only a JSON list."
)
PLATE_QUESTIONS = (r"what'?s on (my|the) plate", r"what (do i|is there|have i got|can i) (have|eat)?",
                   r"what (food|foods) ", r"what'?s there", r"what is on")


def squash(text: str) -> str:
    """'Apple sauce!' -> 'applesauce' (spaces and punctuation removed, lowercase)."""
    return re.sub(r"[^a-z]", "", text.lower())


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and w[-3] in "sxz":
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def match_food(text: str, labels: dict[str, Optional[str]]) -> Optional[str]:
    """Pure. Which bowl does `text` ask for? labels: bowl -> food label (or None).

    Whole label (spaces ignored) beats a single word of it, longer beats shorter. 'apple sauce' matches
    'applesauce'; 'potato' matches 'mashed potatoes'. Ambiguous single-word matches return None.
    """
    t_sq = squash(text)
    t_words = {singular(w) for w in words(text)}
    best: list[tuple[int, str]] = []
    for bowl, food in labels.items():
        if not food:
            continue
        f_sq = squash(food)
        if f_sq and f_sq in t_sq:
            best.append((100 + len(f_sq), bowl))
            continue
        hits = [w for w in (singular(x) for x in words(food)) if len(w) >= 4 and w in t_words]
        if hits:
            best.append((max(len(w) for w in hits), bowl))
    if not best:
        return None
    best.sort(reverse=True)
    if len(best) > 1 and best[0][0] == best[1][0] and best[0][0] < 100:
        return None
    return best[0][1]


def asks_whats_on_plate(text: str) -> bool:
    t = " " + text.lower().replace("’", "'") + " "
    return any(re.search(p, t) for p in PLATE_QUESTIONS)


def describe(foods: list[str]) -> str:
    foods = [f for f in foods if f]
    if not foods:
        return "I don't see any food yet"
    if len(foods) == 1:
        return f"I see {foods[0]}"
    return "I see " + ", ".join(foods[:-1]) + " and " + foods[-1]


class PlateWatcher:
    QUOTA_PAUSE_S = 120
    def __init__(self, cfg: Config, frame_fn: Callable[[], Optional[object]],
                 vision_fn: Optional[Callable[[list[tuple[str, bytes]]], list[dict]]] = None,
                 active_fn: Callable[[], bool] = lambda: True,
                 locate_fn: Optional[Callable[[bytes], list[dict]]] = None,
                 spoon_fn: Optional[Callable[[bytes], dict]] = None,
                 point_fn: Optional[Callable[[bytes], dict]] = None):
        """frame_fn -> latest BGR image (numpy) or None. vision_fn(crops) -> [{"bowl","food","amount"}], where
        crops = [(bowl_name, jpeg_bytes)]; default is Gemini. active_fn: only spend vision calls when True."""
        self.cfg, self.frame_fn, self.active_fn = cfg, frame_fn, active_fn
        p = cfg.plate
        self.enabled = p.enabled and bool(p.bowls)
        self.bowls: dict[str, dict] = {
            b.name: {"food": b.label or None, "amount": None, "source": "config" if b.label else None,
                     "empty_count": 0, "ts": None, "vision_texture": None} for b in p.bowls}
        self.manual_texture: dict[str, str] = {}
        self.table = TableCalib(cfg.path(p.track.calib_file))
        self.anchor_fn: Callable[[str], Optional[tuple[float, float]]] = lambda bowl: None   # set by System
        self.detections: list[dict] = []
        self.autocal_status_fn: Callable[[], dict] = lambda: {}
        self._locate: Optional[Callable] = locate_fn
        self._spoon: Optional[Callable] = spoon_fn
        self._point: Optional[Callable] = point_fn
        self._bowl_touch: Optional[Callable] = None
        self.text_generate: Optional[Callable[[str], object]] = None    # Gemini, text in -> JSON out (planner)
        self.spoon_frame_fn: Callable[[], Optional[object]] = frame_fn   # System: the face camera (sees the spoon)
        self.manual: dict[str, str] = {}
        self.error: Optional[str] = None
        self.last_check = 0.0
        self.checks = 0
        self._busy = False
        self._stop = threading.Event()
        self._vision = vision_fn
        self.vision_name = "custom" if vision_fn else "none"
        self.models = [m for m in dict.fromkeys([p.model, *p.fallback_models]) if m]
        self.model_used: Optional[str] = None
        if self.enabled and vision_fn is None and os.environ.get("GEMINI_API_KEY"):
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                                      http_options=types.HttpOptions(timeout=int(p.timeout_s * 1000)))
                self._vision = lambda crops: self._gemini(client, crops)
                self._locate = self._locate or (lambda jpg: self._gemini_locate(client, jpg))
                self.text_generate = lambda prompt: self._generate(client, [prompt])
                self._bowl_touch = lambda jpg: self._generate(client, [
                    types.Part.from_bytes(data=jpg, mime_type="image/jpeg"), BOWL_TOUCH_PROMPT])
                self._point = self._point or (lambda jpg: self._generate(client, [
                    types.Part.from_bytes(data=jpg, mime_type="image/jpeg"), TIP_PROMPT]))
                self._spoon = self._spoon or (lambda jpg: self._generate(client, [
                    types.Part.from_bytes(data=jpg, mime_type="image/jpeg"), SPOON_PROMPT]))
                self.vision_name = "gemini"
            except Exception as e:
                self.error = f"gemini unavailable: {e!r}"

    def reset_bowls(self) -> None:
        """config plate.bowls changed (Quick setup): start the bowl table over."""
        p = self.cfg.plate
        self.enabled = p.enabled and bool(p.bowls)
        self.bowls = {b.name: {"food": b.label or None, "amount": None, "source": "config" if b.label else None,
                               "empty_count": 0, "ts": None, "vision_texture": None} for b in p.bowls}
        self.manual.clear()
        self.manual_texture.clear()
        self.last_check = 0.0                 # look at the new bowls straight away

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.enabled and self._vision:
            threading.Thread(target=self._loop, name="plate", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.active_fn() and time.time() - self.last_check >= self.cfg.plate.interval_s:
                self.check_now()
            time.sleep(0.25)

    # ------------------------------------------------------------------ vision
    def crops(self, image) -> list[tuple[str, bytes]]:
        import cv2
        h, w = image.shape[:2]
        sx, sy = w / 1280.0, h / 720.0          # rois are in 1280x720 reference coords, like the rest of config
        out = []
        for b in self.cfg.plate.bowls:
            x, y, bw, bh = b.roi
            st = self.bowls.get(b.name) or {}
            if st.get("pos_px") and st.get("pos_ts") and time.time() - st["pos_ts"] < self.cfg.plate.track.max_age_s:
                x, y = st["pos_px"][0] - bw / 2, st["pos_px"][1] - bh / 2     # the bowl was found elsewhere: look there
            x0, y0 = max(0, int(x * sx)), max(0, int(y * sy))
            x1, y1 = min(w, int((x + bw) * sx)), min(h, int((y + bh) * sy))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            ok, jpg = cv2.imencode(".jpg", image[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                out.append((b.name, jpg.tobytes()))
        return out

    def _gemini(self, client, crops: list[tuple[str, bytes]]) -> list[dict]:
        from google.genai import types
        contents: list = []
        for name, jpg in crops:
            contents += [f"Bowl id: {name}", types.Part.from_bytes(data=jpg, mime_type="image/jpeg")]
        contents.append(PROMPT)
        order = ([self.model_used] if self.model_used else []) + [m for m in self.models if m != self.model_used]
        errors = []
        for model in order:
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                d = json.loads(resp.text)
            except Exception as e:                   # overloaded (503), deadline (504), retired (404), bad JSON
                errors.append(f"{model}: {str(e)[:80]}")
                continue
            self.model_used = model
            break
        else:
            raise RuntimeError("all models failed: " + " | ".join(errors))
        if isinstance(d, dict):
            d = d.get("bowls") or [d]
        return d if isinstance(d, list) else []

    def _generate(self, client, contents) -> object:
        """generate_content with the model fallback list (see _gemini)."""
        from google.genai import types
        order = ([self.model_used] if self.model_used else []) + [m for m in self.models if m != self.model_used]
        errors = []
        for model in order:
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                d = json.loads(resp.text)
            except Exception as e:
                errors.append(f"{model}: {str(e)[:80]}")
                continue
            self.model_used = model
            return d
        raise RuntimeError("all models failed: " + " | ".join(errors))

    def _gemini_locate(self, client, jpg: bytes) -> list[dict]:
        from google.genai import types
        d = self._generate(client, [types.Part.from_bytes(data=jpg, mime_type="image/jpeg"), LOCATE_PROMPT])
        return d if isinstance(d, list) else []

    def locate(self, image) -> None:
        """Where each bowl is on the table now. Detections -> table mm (homography) -> nearest to each bowl's taught
        spot (a detection whose food matches the bowl's label wins)."""
        import cv2
        if not (self.cfg.plate.track.enabled and self.table.ready and self._locate):
            return
        h, w = image.shape[:2]
        ok, jpg = cv2.imencode(".jpg", cv2.resize(image, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        dets = []
        for r in self._locate(jpg.tobytes()) or []:
            b = r.get("box_2d") if isinstance(r, dict) else None
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            ymin, xmin, ymax, xmax = (float(v) for v in b)
            px = ((xmin + xmax) / 2 * 1.28, (ymin + ymax) / 2 * 0.72)       # 0-1000 -> 1280x720 reference
            mm = self.table.to_table(px)
            if mm:
                dets.append({"food": (r.get("food") or None), "px": px, "mm": mm, "box": [ymin, xmin, ymax, xmax]})
        self.detections = dets
        now = time.time()
        reach = 2 * self.cfg.plate.track.max_shift_mm
        for name, b in self.bowls.items():
            anchor = self.anchor_fn(name)
            if not anchor or not dets:
                continue

            def score(d):
                dist = math.dist(d["mm"], anchor)
                same = d["food"] and b["food"] and match_food(d["food"], {name: b["food"]}) == name
                return dist - (1000 if same else 0)
            best = min(dets, key=score)
            if math.dist(best["mm"], anchor) <= reach:
                b.update(pos_mm=best["mm"], pos_px=best["px"], pos_ts=now)

    def box_of_touched_bowl(self) -> Optional[dict]:
        """{"roi": (x, y, w, h) in 1280x720 reference, "food": str|None} for the container the spoon is dipped in."""
        import cv2
        if self._bowl_touch is None:
            return None
        img = self.frame_fn()
        if img is None:
            return None
        ok, jpg = cv2.imencode(".jpg", cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        try:
            d = self._bowl_touch(jpg.tobytes()) if ok else None
        except Exception as e:
            self.error = f"bowl box: {e!r}"[:300]
            return None
        b = d.get("box_2d") if isinstance(d, dict) else None
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            return None
        ymin, xmin, ymax, xmax = (float(v) for v in b)
        x0, y0, x1, y1 = xmin * 1.28, ymin * 0.72, xmax * 1.28, ymax * 0.72
        pad = 12
        return {"roi": (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(x1 - x0 + 2 * pad), int(y1 - y0 + 2 * pad)),
                "food": (d.get("food") or None)}

    def point_spoon_tip(self) -> Optional[tuple[float, float]]:
        """Pixel (1280x720 reference) of the spoon tip in the PLATE camera picture, or None."""
        import cv2
        if self._point is None:
            return None
        img = self.frame_fn()
        if img is None:
            return None
        img = cv2.resize(img, (1280, 720))
        self.last_point_frame = img
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        try:
            d = self._point(jpg.tobytes()) if ok else None
        except Exception as e:
            self.error = f"spoon tip: {e!r}"[:300]
            return None
        p = d.get("point") if isinstance(d, dict) else None
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            return None
        y, x = float(p[0]), float(p[1])
        return x * 1.28, y * 0.72

    @property
    def can_check_spoon(self) -> bool:
        return self._spoon is not None and self.cfg.plate.spoon_check.enabled

    def check_spoon(self) -> Optional[dict]:
        """{"spoon_visible", "food_on_spoon", "amount"} from the face camera, or None (no camera/vision/answer)."""
        import cv2
        if not self.can_check_spoon:
            return None
        try:
            img = self.spoon_frame_fn()
            if img is None:
                return None
            ok, jpg = cv2.imencode(".jpg", cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])
            d = self._spoon(jpg.tobytes()) if ok else None
            return d if isinstance(d, dict) else None
        except Exception as e:
            self.error = f"spoon check: {e!r}"[:300]
            return None

    def shift_for(self, bowl: str) -> Optional[tuple[float, float]]:
        """(dx, dy) mm from the taught spot to where the bowl is now; None = not located (use the taught spot)."""
        b = self.bowls.get(bowl) or {}
        anchor = self.anchor_fn(bowl)
        if not (self.cfg.plate.track.enabled and anchor and b.get("pos_mm") and b.get("pos_ts")):
            return None
        if time.time() - b["pos_ts"] > self.cfg.plate.track.max_age_s:
            return None
        return b["pos_mm"][0] - anchor[0], b["pos_mm"][1] - anchor[1]

    def check_now(self) -> bool:
        """One vision pass. Returns True if labels were updated. Never raises."""
        if not self._vision or self._busy:
            return False
        self._busy = True
        self.last_check = time.time()
        try:
            img = self.frame_fn()
            if img is None:
                self.error = "no plate camera frame"
                return False
            crops = self.crops(img)
            if not crops:
                self.error = "bowl boxes are outside the image"
                return False
            self.apply(self._vision(crops))
            self.locate(img)
            self.error = None
            self.checks += 1
            return True
        except Exception as e:
            self.error = f"vision: {e!r}"[:300]
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                self.last_check = time.time() + self.QUOTA_PAUSE_S      # out of quota: stop asking for a while
                self.error = f"Gemini quota used up; pausing vision {self.QUOTA_PAUSE_S} s"
            return False
        finally:
            self._busy = False

    def apply(self, results: list[dict]) -> None:
        now = time.time()
        for r in results:
            if not isinstance(r, dict):
                continue
            name = str(r.get("bowl", ""))
            if name not in self.bowls:
                continue
            b = self.bowls[name]
            food = r.get("food")
            food = str(food).strip().lower() if food else None
            amount = r.get("amount") if r.get("amount") in AMOUNTS else None
            fixed = name in self.manual or b["source"] == "config"     # a typed or configured label is not overridden
            if not fixed and food:
                b["food"], b["source"] = food, self.vision_name
            tex = r.get("texture")
            b["vision_texture"] = tex if tex in TEXTURES and food and not fixed else None
            b["amount"], b["ts"] = amount, now
            empty = amount == "empty" or (food is None and not fixed)
            b["empty_count"] = b["empty_count"] + 1 if empty else 0

    # ------------------------------------------------------------------ queries
    def set_label(self, bowl: str, food: Optional[str], texture: Optional[str] = None) -> None:
        """Operator override (UI). Empty food clears the food override. texture: liquid|soft|thick|firm,
        "" = automatic, None = leave as is (and then an empty food with a texture only sets the texture)."""
        if bowl not in self.bowls:
            raise KeyError(bowl)
        if texture is not None:
            if texture and texture not in TEXTURES:
                raise ValueError(f"texture must be one of {TEXTURES}")
            if texture:
                self.manual_texture[bowl] = texture
            else:
                self.manual_texture.pop(bowl, None)
            if not food:
                return
        if food:
            self.manual[bowl] = food.strip().lower()
            self.bowls[bowl].update(food=self.manual[bowl], source="operator", empty_count=0)
        else:
            self.manual.pop(bowl, None)
            cfg_label = next((b.label for b in self.cfg.plate.bowls if b.name == bowl), "")
            self.bowls[bowl].update(food=cfg_label or None, source="config" if cfg_label else None)

    def labels(self) -> dict[str, Optional[str]]:
        return {k: v["food"] for k, v in self.bowls.items()}

    def find(self, text: str) -> Optional[str]:
        return match_food(text, self.labels())

    def is_empty(self, bowl: str) -> bool:
        b = self.bowls.get(bowl)
        return bool(b) and b["empty_count"] >= self.cfg.plate.empty_confirm

    def food_of(self, bowl: Optional[str]) -> Optional[str]:
        return self.bowls[bowl]["food"] if bowl in self.bowls else None

    def texture_of(self, bowl: str) -> tuple[str, str]:
        """(texture, why). Operator choice, then what vision saw, then the food's name, then the default."""
        if bowl in self.manual_texture:
            return self.manual_texture[bowl], "operator"
        b = self.bowls.get(bowl) or {}
        if b.get("vision_texture") and b.get("source") == self.vision_name:
            return b["vision_texture"], "vision"
        t = texture_from_name(b.get("food"))
        if t:
            return t, "food name"
        return self.cfg.plate.default_texture, "default"

    def scoop_for(self, bowl: str) -> str:
        b = next(b for b in self.cfg.plate.bowls if b.name == bowl)
        return b.scoop or f"scoop_{b.name}"

    def describe(self) -> str:
        return describe([v["food"] for k, v in self.bowls.items() if v["food"] and not self.is_empty(k)])

    def status(self) -> dict:
        return {"enabled": self.enabled, "vision": self.vision_name if self._vision else "none", "error": self.error,
                "model": self.model_used, "table": self.table.status(),
                "autocal": self.autocal_status_fn(),
                "shifts": {k: (lambda s: [round(s[0]), round(s[1])] if s else None)(self.shift_for(k))
                           for k in self.bowls},
                "checks": self.checks, "last_check_age_s": round(time.time() - self.last_check, 1) if self.last_check else None,
                "bowls": [{"name": k, "food": v["food"], "amount": v["amount"], "source": v["source"],
                           "empty": self.is_empty(k), "scoop": self.scoop_for(k), "texture": self.texture_of(k)[0],
                           "texture_why": self.texture_of(k)[1]} for k, v in self.bowls.items()],
                "summary": self.describe()}
