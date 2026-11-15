"""One overhead photo -> a metric map of the table in the arm's own coordinates.

The arm first holds its spoon tip at a known spot (PHOTO_TIP_MM) just above the table. The operator takes ONE photo
from above (phone, roughly straight down) with the arm base, the spoon and the bowls in it, and uploads it. Vision
finds three kinds of things in that photo: the centre of the arm's round base, the spoon tip, and every bowl (box,
food, how full, texture). The robot knows where its base is (0, 0) and where it put the spoon tip, so those two
points fix the photo's scale, rotation and offset (a similarity transform on the table plane). Every bowl's centre
then has a position the arm can reach for, with no teaching.

From how full each bowl looks it also predicts the food surface height (PRIOR_BOWL_MM deep bowl); the touch probe
measures it on the first scoop and the estimate is corrected bite by bite (see FoodSurface).
"""
from __future__ import annotations

import base64
import math
from typing import Callable, Optional

PHOTO_TIP_MM = (170.0, 0.0, 30.0)      # where the arm holds the spoon tip for the photo (in front of the base)
PRIOR_BOWL_MM = 55.0                   # typical cup/bowl depth when nothing else is known
FILL = {"full": 0.85, "half": 0.5, "low": 0.2, "empty": 0.0}

PROMPT = (
    "This photo was taken from above a table. On it are a small teal robot arm on a round base, a white plastic spoon "
    "held by the arm, and some bowls or cups of food. Answer only JSON with:\n"
    "  \"base\": [y, x] the centre of the robot arm's round base,\n"
    "  \"tip\": [y, x] the very tip of the spoon held by the arm (far end of the spoon's bowl),\n"
    "  \"bowls\": a list of {\"box_2d\": [ymin, xmin, ymax, xmax], \"food\": 1-3 lowercase words, \"amount\": "
    "\"full\"|\"half\"|\"low\"|\"empty\", \"texture\": \"liquid\"|\"soft\"|\"thick\"|\"firm\"} for every container "
    "with food.\n"
    "All coordinates normalised to 0-1000 of the image height (y) and width (x). Use null for anything you can't see. "
    "Don't list the robot, the spoon, a phone or other non-food things as bowls."
)


def _px(pt, w: int, h: int) -> Optional[tuple[float, float]]:
    if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
        return None
    y, x = float(pt[0]), float(pt[1])
    return x / 1000.0 * w, y / 1000.0 * h


def similarity(base_px, tip_px, tip_mm) -> Callable[[tuple[float, float]], tuple[float, float]]:
    """Pixel -> table mm from two matching points. Pixels have y pointing down, so v is negated first (a camera
    looking down at the table sees it without mirroring). Base is at (0, 0) mm."""
    pb = complex(base_px[0], -base_px[1])
    pt = complex(tip_px[0], -tip_px[1])
    mt = complex(tip_mm[0], tip_mm[1])
    if abs(pt - pb) < 10:
        raise ValueError("base and spoon tip are too close together in the photo")
    a = mt / (pt - pb)                           # rotation + scale
    def f(p):
        m = a * (complex(p[0], -p[1]) - pb)
        return m.real, m.imag
    f.mm_per_px = abs(a)
    return f


def analyze(jpg: bytes, w: int, h: int, generate: Callable[[list], object], tip_mm=PHOTO_TIP_MM[:2]) -> dict:
    """Vision on one overhead photo -> {"bowls": [{name, food, amount, texture, mm, px, box}], "mm_per_px", ...}."""
    d = generate(jpg)
    if not isinstance(d, dict):
        raise ValueError("vision gave no answer")
    base, tip = _px(d.get("base"), w, h), _px(d.get("tip"), w, h)
    if base is None or tip is None:
        raise ValueError("couldn't find " + ("the arm's base" if base is None else "the spoon tip") + " in the photo")
    to_mm = similarity(base, tip, tip_mm)
    bowls = []
    for i, b in enumerate(d.get("bowls") or []):
        bx = b.get("box_2d") if isinstance(b, dict) else None
        if not (isinstance(bx, (list, tuple)) and len(bx) == 4):
            continue
        ymin, xmin, ymax, xmax = (float(v) for v in bx)
        c = ((xmin + xmax) / 2 / 1000 * w, (ymin + ymax) / 2 / 1000 * h)
        x, y = to_mm(c)
        diam = max((xmax - xmin) / 1000 * w, (ymax - ymin) / 1000 * h) * to_mm.mm_per_px
        bowls.append({"food": (b.get("food") or None), "amount": b.get("amount"), "texture": b.get("texture"),
                      "px": c, "box": [xmin, ymin, xmax, ymax], "mm": (x, y), "diameter_mm": diam})
    bowls.sort(key=lambda b: math.atan2(b["mm"][1], b["mm"][0]), reverse=True)   # left to right, seen from the arm
    for i, b in enumerate(bowls):
        b["name"] = f"bowl{i + 1}"
    return {"bowls": bowls, "mm_per_px": to_mm.mm_per_px, "base_px": base, "tip_px": tip}


# ---------------------------------------------------------------- consensus over rotated views
# A vision model answers confidently even when it is guessing. Asking it about the same photo rotated 0/90/180/270
# deg and mapping every answer back gives four independent-ish readings: real things land in the same place every
# time, guesses wander or appear only once. On a real overhead photo (2026-11-14) the spoon tip agreed to +-7 px and
# the cups to +-2..6 px across the four views, while three phantom "peanut butter" bowls appeared in only one view.

ROTATIONS = (0, 90, 180, 270)
MIN_VOTES = 3                    # a bowl must be seen in at least this many of the 4 views to be used
UNSURE_VOTES = 2                 # seen in this many: reported as "unsure, please confirm", not scooped


def _unrotate(rot: int, W: int, H: int):
    """Pixel in the image rotated clockwise by rot -> pixel in the original W x H image."""
    if rot == 0:
        return lambda x, y: (x, y)
    if rot == 90:
        return lambda x, y: (y, H - x)
    if rot == 180:
        return lambda x, y: (W - x, H - y)
    return lambda x, y: (W - y, x)                                   # 270


def _norm_answer(d) -> dict:
    if isinstance(d, list) and d and isinstance(d[0], dict):        # sometimes wrapped in a list
        d = d[0]
    return d if isinstance(d, dict) else {}


def _median(pts):
    xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
    n = len(pts)
    return (xs[n // 2] + xs[(n - 1) // 2]) / 2, (ys[n // 2] + ys[(n - 1) // 2]) / 2


def consensus(answers: list[tuple[int, dict, int, int]], W: int, H: int) -> dict:
    """answers: [(rotation, vision answer, rotated width, rotated height)] -> agreed points in original pixels."""
    pts = {"base": [], "tip": []}
    dets = []                                                        # (view index, centre, box, food, amount, texture)
    for i, (rot, d, w, h) in enumerate(answers):
        d = _norm_answer(d)
        un = _unrotate(rot, W, H)
        for key in ("base", "tip"):
            p = _px(d.get(key), w, h)
            if p is not None:
                pts[key].append(un(*p))
        for b in d.get("bowls") or []:
            bx = b.get("box_2d") if isinstance(b, dict) else None
            if not (isinstance(bx, (list, tuple)) and len(bx) == 4):
                continue
            ymin, xmin, ymax, xmax = (float(v) for v in bx)
            c1 = un(xmin / 1000 * w, ymin / 1000 * h)
            c2 = un(xmax / 1000 * w, ymax / 1000 * h)
            box = (min(c1[0], c2[0]), min(c1[1], c2[1]), max(c1[0], c2[0]), max(c1[1], c2[1]))
            dets.append((i, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box, b.get("food"), b.get("amount"),
                         b.get("texture")))
    out = {}
    for key, ps in pts.items():
        if len(ps) >= MIN_VOTES:
            m = _median(ps)
            out[key] = {"px": m, "votes": len(ps), "spread_px": max(math.dist(m, p) for p in ps)}
        else:
            out[key] = {"px": None, "votes": len(ps), "spread_px": None}
    clusters: list[list] = []
    for d in sorted(dets, key=lambda d: d[0]):
        size = max(d[2][2] - d[2][0], d[2][3] - d[2][1])
        home = next((c for c in clusters if math.dist(_median([e[1] for e in c]), d[1]) <= max(30.0, 0.5 * size)
                     and d[0] not in {e[0] for e in c}), None)
        (home.append(d) if home is not None else clusters.append([d]))
    bowls = []
    from collections import Counter
    for c in clusters:
        votes = len({e[0] for e in c})
        if votes < UNSURE_VOTES:
            continue
        centre = _median([e[1] for e in c])
        boxes = [e[2] for e in c]
        box = tuple(sorted(b[k] for b in boxes)[len(boxes) // 2] for k in range(4))
        names = Counter((e[3] or "").strip().lower() for e in c if e[3])
        amounts = Counter(e[4] for e in c if e[4])
        textures = Counter(e[5] for e in c if e[5])
        bowls.append({"px": centre, "box_px": box, "votes": votes, "views": len(answers),
                      "spread_px": max(math.dist(centre, e[1]) for e in c),
                      "food": names.most_common(1)[0][0] if names else None,
                      "names_seen": sorted(names), "amount": amounts.most_common(1)[0][0] if amounts else None,
                      "texture": textures.most_common(1)[0][0] if textures else None,
                      "sure": votes >= MIN_VOTES})
    out["bowls"] = bowls
    return out


def refine_circle(img, px, r_min: float, r_max: float, max_move: float) -> tuple[tuple[float, float], bool]:
    """From above, the arm's turntable and the cups are circles: find the circle nearest the vision answer and use its
    centre (a few px instead of vision's +-10-15). Returns (centre, refined?). Keeps the vision answer when no circle
    of the right size is close enough."""
    import cv2
    import numpy as np
    H, W = img.shape[:2]
    pad = int(r_max * 1.6)
    x0, y0 = max(0, int(px[0]) - pad), max(0, int(px[1]) - pad)
    x1, y1 = min(W, int(px[0]) + pad), min(H, int(px[1]) + pad)
    crop = cv2.medianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), 5)
    if crop.size == 0:
        return px, False
    found = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, r_min), param1=100, param2=30,
                             minRadius=max(5, int(r_min)), maxRadius=int(r_max))
    if found is None:
        return px, False
    best = min((math.dist((cx + x0, cy + y0), px), (cx + x0, cy + y0)) for cx, cy, _ in found[0])
    return (best[1], True) if best[0] <= max_move else (px, False)


def analyze_robust(img, generate: Callable[[bytes], object], tip_mm=PHOTO_TIP_MM[:2], views=ROTATIONS) -> dict:
    """The photo, asked about in 4 rotations in parallel; only what the views agree on is used. Each bowl gets a
    vote count and an uncertainty in mm. Bowls seen in only 2 views come back as unsure (not scooped)."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    H, W = img.shape[:2]
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def one(rot):
        im = img if rot == 0 else cv2.rotate(img, codes[rot])
        h, w = im.shape[:2]
        jpg = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        try:
            return rot, generate(jpg), w, h
        except Exception:
            return rot, {}, w, h
    with ThreadPoolExecutor(max_workers=len(views)) as ex:
        answers = list(ex.map(one, views))
    agreed = consensus(answers, W, H)
    for bw in agreed["bowls"]:                                       # cups: snap to the rim circle
        bx = bw["box_px"]
        r = max(bx[2] - bx[0], bx[3] - bx[1]) / 2
        c, ok = refine_circle(img, bw["px"], 0.7 * r, 1.25 * r, 0.35 * r)
        if ok:
            bw["px"], bw["spread_px"], bw["refined"] = c, min(bw["spread_px"], 4.0), True
    if agreed["base"]["px"] is None or agreed["tip"]["px"] is None:
        what = "the arm's base" if agreed["base"]["px"] is None else "the spoon tip"
        raise ValueError(f"the views don't agree on {what}: retake the photo from straight above, arm and spoon in view")
    to_mm = similarity(agreed["base"]["px"], agreed["tip"]["px"], tip_mm)
    k = to_mm.mm_per_px
    ref_unc = max(agreed["base"]["spread_px"], agreed["tip"]["spread_px"]) * k
    bowls = []
    for b in agreed["bowls"]:
        x, y = to_mm(b["px"])
        bx = b["box_px"]
        bowls.append({**b, "mm": (x, y), "uncertainty_mm": round(b["spread_px"] * k + ref_unc, 1),
                      "diameter_mm": max(bx[2] - bx[0], bx[3] - bx[1]) * k})
    bowls.sort(key=lambda b: math.atan2(b["mm"][1], b["mm"][0]), reverse=True)
    n = 0
    for b in bowls:
        if b["sure"]:
            n += 1
            b["name"] = f"bowl{n}"
        else:
            b["name"] = None
    return {"bowls": bowls, "mm_per_px": k, "base_px": agreed["base"]["px"], "tip_px": agreed["tip"]["px"],
            "base_votes": agreed["base"]["votes"], "tip_votes": agreed["tip"]["votes"],
            "reference_uncertainty_mm": round(ref_unc, 1)}


def predicted_surface_mm(amount: Optional[str], bowl_depth: float = PRIOR_BOWL_MM) -> float:
    """Height of the food surface above the table, from how full the bowl looks."""
    return 8.0 + FILL.get(amount or "half", 0.5) * (bowl_depth - 8.0)     # 8 mm: the bowl's own bottom


class FoodSurface:
    """Per bowl: belief about the food surface height (mm above the table) - predicted from the photo, corrected by
    every touch. A tiny world model: predict -> act -> feel -> update (exponential update, like a 1-D Kalman filter
    with a fixed gain), and the history of prediction errors shows it learning."""

    GAIN = 0.6

    def __init__(self):
        self.est: dict[str, float] = {}
        self.history: dict[str, list[dict]] = {}

    def predict(self, bowl: str, amount: Optional[str] = None) -> float:
        if bowl not in self.est:
            self.est[bowl] = predicted_surface_mm(amount)
        return self.est[bowl]

    def felt(self, bowl: str, touched_mm: float) -> dict:
        pred = self.est.get(bowl, touched_mm)
        self.est[bowl] = pred + self.GAIN * (touched_mm - pred)
        rec = {"predicted_mm": round(pred, 1), "felt_mm": round(touched_mm, 1), "error_mm": round(touched_mm - pred, 1),
               "new_estimate_mm": round(self.est[bowl], 1)}
        self.history.setdefault(bowl, []).append(rec)
        return rec

    def status(self) -> dict:
        return {b: {"estimate_mm": round(v, 1), "history": self.history.get(b, [])[-8:]} for b, v in self.est.items()}


def decode_upload(data_url: str) -> bytes:
    """'data:image/jpeg;base64,....' (from the browser) -> raw bytes."""
    return base64.b64decode(data_url.split(",", 1)[1] if "," in data_url else data_url)
