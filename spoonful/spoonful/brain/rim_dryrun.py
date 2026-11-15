"""Dry run of the rim-aware scoop: a picture of what the planner would do at one bowl, without the arm.

`verdict(cfg, bowl)` runs arm/cup.plan_entry on the configured cup and spoon and says, in plain words, whether a
safe entry exists and why not. `render(...)` draws the same plan: a top view, a side view cut through the cup
centre along the arm's line, and the text. Both views have the arm on the left. If the bowl still has a saved
legacy scoop (scoop_<bowl>), its tip path and the places it would touch this cup are drawn too ("old path").

Reads config and the trajectory files only. It never imports, starts or moves the arm and opens no camera. The
picture is the planner's prediction from the numbers in config; nothing here has been checked on the real arm, and
servo sag is not in it (the planned poses are drawn as commanded).
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from ..arm import cup as U
from ..arm.trajectory import Library
from ..config import Config

FONT, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
INK, GREY, LIGHT = (40, 40, 40), (120, 120, 120), (190, 190, 190)
RED, ORANGE, BROWN = (0, 0, 220), (0, 140, 255), (40, 80, 140)
WALL, BAND, SAFE = (70, 70, 70), (205, 228, 255), (215, 240, 215)
POSE_COL = {"hover": (210, 120, 0), "pivot (entry angle)": (160, 150, 0), "dip (lowest)": (0, 130, 0),
            "end of stroke": (130, 0, 90), "out": GREY, "clear": LIGHT}
WHY = {"unreachable": "the arm cannot hold that pose within its joint limits",
       "clearance": "the spoon comes too close to the cup on the way in",
       "retract": "the spoon comes too close to the cup on the way out",
       "approach": "no clear way from the start pose to above the cup"}
REASON = {"unmeasured": "a cup or spoon dimension is missing",
          "too_narrow": "the opening is too narrow for the spoon with the margin",
          "unreachable": "the arm cannot reach a safe entry over this cup",
          "no_clear_path": "every entry comes too close to the cup"}
FOLLOW_FRAC, FOLLOW_MM = 0.15, 2.0     # a followed move widens the cup's uncertainty (perception/rim.py's measured error)
W, LEFT, PANEL_W, TOP_H, SIDE_H = 1280, 760, 730, 280, 420


# ---------------------------------------------------------------- the plan, with no arm
def _library(cfg: Config) -> Optional[Library]:
    """The saved trajectories of the real arm, read-only. None if they can't be read."""
    try:
        return Library(cfg, cfg.path(cfg.arm.trajectories_dir))
    except Exception:
        return None


def _case(cfg: Config, bowl_name: str, observation=None) -> SimpleNamespace:
    """Everything the verdict and the picture need: the cup, the spoon, the plan or the refusal."""
    rim = cfg.plate.rim
    bowl = next((b for b in cfg.plate.bowls if b.name == bowl_name), None)
    c = SimpleNamespace(name=bowl_name, bowl=bowl, cup=None, spoon=U.Spoon.from_cfg(cfg.spoon), plan=None, err=None,
                        lib=_library(cfg), start=None, moved_mm=0.0, obs=observation, margin=rim.clearance_mm,
                        want_angle=45.0, want_drag=20.0)
    if bowl is None or bowl.cup is None:
        return c
    cup = U.Cup.from_cfg(bowl.cup, rim.default_rim_mm)
    if observation is not None and getattr(observation, "status", "") == "moved":
        c.moved_mm = math.hypot(observation.dx, observation.dy)
        cup = cup.moved(observation.dx, observation.dy,
                        uncertainty=cup.uncertainty + FOLLOW_FRAC * c.moved_mm + FOLLOW_MM)
    c.cup, c.margin = cup, rim.clearance_mm + cup.uncertainty
    c.want_angle = float(bowl.attack.get("entry_angle_deg", 45.0))
    c.want_drag = float(bowl.attack.get("drag_mm", 20.0))
    if c.lib is not None and c.lib.has("rest"):          # a scoop starts from rest; without it, no approach is drawn
        c.start = c.lib.pose("rest")
    try:
        from .rim_scoop import plan_deepest      # the same search the live scoop uses
        c.plan = plan_deepest(cfg.kinematics, cfg.arm.soft_limits_deg, cfg.arm.home_pose, cup, c.spoon,
                              _depth(bowl.cup, rim), clearance_mm=rim.clearance_mm,
                              hover_above_rim_mm=rim.hover_above_rim_mm, want_angle_deg=c.want_angle,
                              want_drag_mm=c.want_drag, start=c.start, dig_mm=_dig(bowl.attack, cup),
                              headroom_mm=rim.sag_headroom_mm)
    except U.NoSafeEntry as e:
        c.err = e
    return c


def _depth(cup_cfg, rim):
    from .rim_scoop import want_depth
    return want_depth(cup_cfg, rim)


def _dig(attack: dict, cup) -> float:
    from .rim_scoop import dig_mm          # the same rule the live scoop uses
    return dig_mm(attack, cup)


def _rejected(rej: list[dict], margin: float, top: int = 6) -> list[str]:
    """The most telling dropped candidates, in plain words: the fit first, then the entries that got as far as arm
    poses (the planner tries the best first), then entry angles with no placing that keeps the margin."""
    out = [r["why"] for r in rej if r.get("stage") == "fit"]
    for r in (r for r in rej if r.get("stage") == "arm"):
        out.append(f"{r['angle_deg']:.0f} deg, tip {r['offset_mm']:.0f} mm past centre, down to {r['dip_z']:.0f} mm, "
                   f"drag {r['drag_mm']:.0f} mm: {WHY.get(r['why'], r['why'])}")
    for r in (r for r in rej if r.get("stage") == "geometry" and not r.get("kept")):
        best = f" (best gap {r['best_mm']:.1f} mm, needs {margin:.1f})" if r.get("best_mm") is not None else ""
        out.append(f"{r['angle_deg']:.0f} deg entry: no placing inside the cup keeps the margin{best}"
                   if r["why"] == "clearance" else f"{r['angle_deg']:.0f} deg entry: {r['why'].replace('_', ' ')}")
    return out[:top]


def _verdict(cfg: Config, c: SimpleNamespace) -> dict:
    rim, food = cfg.plate.rim, (c.bowl.label if c.bowl else "")
    who = f"{c.name} ({food})" if food else c.name
    v = {"bowl": c.name, "food": food, "has_cup": c.cup is not None, "feasible": False, "reason": "", "detail": "",
         "refusals": [], "rejected": [], "cup": None, "spoon_measured": cfg.spoon.measured,
         "want_angle_deg": c.want_angle, "want_drag_mm": c.want_drag, "clearance_mm": rim.clearance_mm,
         "margin_mm": c.margin, "entry_angle_deg": None, "drag_mm": None, "dip_z_mm": None,
         "min_clearance_mm": None, "approach_poses": 0, "observation": None}
    if c.cup is None:
        v["reason"] = "no_bowl" if c.bowl is None else "no_cup"
        v["detail"] = f"no bowl named {c.name}" if c.bowl is None else "no cup geometry: map the table first"
        v["text"] = f"{who}: {v['detail']}"
        return v
    cc = c.bowl.cup
    v["cup"] = {"source": cc.source, "confirmed": cc.confirmed, "uncertainty_mm": c.cup.uncertainty,
                "rim_mm": c.cup.rim, "rim_measured": c.cup.rim_measured, "x_mm": c.cup.x, "y_mm": c.cup.y,
                "opening_mm": 2 * c.cup.radius}
    if c.obs is not None:
        v["observation"] = {"status": c.obs.status, "reason": getattr(c.obs, "reason", ""),
                            "confidence": getattr(c.obs, "confidence", 0.0), "moved_mm": c.moved_mm}
        if c.obs.status == "lost" and rim.require_view:
            v["refusals"].append(f"the camera does not see the opening ({c.obs.reason or 'lost'})")
    if rim.require_confirmed and not cc.confirmed:
        v["refusals"].append("the cup centre is not confirmed with the spoon")
    if c.plan is not None:
        p = c.plan
        v.update(feasible=True, entry_angle_deg=p.entry_angle_deg, drag_mm=p.drag_mm, dip_z_mm=p.dip_z,
                 min_clearance_mm=p.min_clearance_mm, approach_poses=len(p.approach),
                 rejected=_rejected(p.rejected, c.margin, 3))
        v["text"] = (f"{who}: safe entry found: {p.entry_angle_deg:.0f} deg below level, drag {p.drag_mm:.0f} mm, "
                     f"tip down to {p.dip_z:.0f} mm above the table, smallest gap {p.min_clearance_mm:.1f} mm "
                     f"(needs {c.margin:.1f})")
    else:
        e = c.err
        v.update(reason=e.reason, detail=e.detail or REASON.get(e.reason, ""), rejected=_rejected(e.rejected, c.margin))
        v["text"] = f"{who}: NO SAFE ENTRY ({e.reason}): {v['detail']}"
    if v["refusals"]:
        v["text"] += "; a real scoop would refuse: " + "; ".join(v["refusals"])
    return v


def verdict(cfg: Config, bowl_name: str, observation=None) -> dict:
    """What plan_entry says for this bowl's cup, as data and as one line ("text"). feasible / reason are the
    planner's; "refusals" lists what else would stop a real scoop (not confirmed, camera lost)."""
    return _verdict(cfg, _case(cfg, bowl_name, observation))


# ---------------------------------------------------------------- geometry for the picture
def _scan(kin, poses: list[dict], cup: U.Cup, spoon: U.Spoon, step_deg: float = 1.0):
    """The move through poses the way the arm makes it (joint-linear): joint samples, tip of each, the smallest
    clearance of each and the spoon point that is at."""
    P = U._q(poses)
    Qs = [P[:1]]
    for k in range(len(P) - 1):
        n = max(1, int(math.ceil(float(np.abs(P[k + 1] - P[k]).max()) / step_deg)))
        Qs.append(P[k] + (P[k + 1] - P[k]) * (np.arange(1, n + 1) / n)[:, None])
    Q = np.concatenate(Qs)
    c, (X, Y, Z) = U._clear_q(kin, Q, cup, spoon)
    i, j = np.arange(len(Q)), c.argmin(axis=1)
    return Q, U._fk(kin, Q)[0], c[i, j], np.stack([X[i, j], Y[i, j], Z[i, j]], axis=1), j


def _pose(Q_row) -> dict:
    return {j: float(v) for j, v in zip(U.FKJ, Q_row)}


def _ghost(cup: U.Cup, spoon: U.Spoon, angle: float, hover_above: float, margin: float):
    """Not a plan: the spoon going straight down at the cup centre at the asked angle, to show where it would be
    too close when there is no safe entry. (lowest body, touching points, points inside the margin)."""
    line, yaw = math.radians(90.0 + angle), math.atan2(cup.y, cup.x)
    low = U._low(spoon, 90.0 + angle)
    z = np.arange(cup.rim + hover_above - low, cup.floor - low - 0.01, -2.0)
    tip = np.stack([np.full(len(z), cup.x), np.full(len(z), cup.y), z], axis=1)
    c, (X, Y, Z) = U._clear(tip, np.full(len(z), line), np.full(len(z), yaw), cup, spoon)
    _, _, w, part = U._template(spoon)
    body = [U.BodyPoint(float(X[-1, i]), float(Y[-1, i]), float(Z[-1, i]), float(w[i]), U.PARTS[part[i]], yaw)
            for i in range(len(w))]
    pts = np.stack([X, Y, Z], axis=2)
    return body, pts[c < 0], pts[(c >= 0) & (c < margin)]


def _split(pts: list, spoon: U.Spoon):
    """Body slices -> (the spoon's line from the tip back, the bowl's underside), each ordered from the tip."""
    s, d, _, _ = U._template(spoon)
    idx = sorted(range(len(pts)), key=lambda i: s[i])
    return [pts[i] for i in idx if d[i] == 0], [pts[i] for i in idx if d[i] > 0]


class _View:
    """A panel of the canvas with a mm -> px mapping (h to the right, v up). Drawing is clipped to the panel."""

    def __init__(self, canvas, x0, y0, w, h, scale, h_lo, v_lo):
        self.img, self.h, self.s, self.h_lo, self.v_lo = canvas[y0:y0 + h, x0:x0 + w], h, scale, h_lo, v_lo

    def px(self, a, v) -> tuple[int, int]:
        lim = 30000.0                                   # far-off points stay drawable (OpenCV clips the line)
        x, y = (a - self.h_lo) * self.s, self.h - (v - self.v_lo) * self.s
        return int(round(max(-lim, min(lim, x)))), int(round(max(-lim, min(lim, y))))

    def line(self, pts, col, thick=1, closed=False):
        if len(pts) > 1:
            cv2.polylines(self.img, [np.array([self.px(*p) for p in pts], np.int32)], closed, col, thick, AA)

    def rect(self, a0, v0, a1, v1, col):
        cv2.rectangle(self.img, self.px(a0, v1), self.px(a1, v0), col, -1)

    def dots(self, pts, col, r=2, fill=True, step=1):
        for p in pts[::step]:
            cv2.circle(self.img, self.px(*p), r, col, -1 if fill else 1, AA)

    def label(self, text, a, v, col=INK, dx=0, dy=0, scale=0.4):
        x, y = self.px(a, v)
        cv2.putText(self.img, text, (x + dx, y + dy), FONT, scale, (255, 255, 255), 3, AA)
        cv2.putText(self.img, text, (x + dx, y + dy), FONT, scale, col, 1, AA)


def _wrap(text: str, width_px: int, scale: float) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        t = f"{cur} {word}".strip()
        if cur and cv2.getTextSize(t, FONT, scale, 1)[0][0] > width_px:
            out.append(cur)
            cur = "   " + word
        else:
            cur = t
    return out + [cur] if cur else out


# ---------------------------------------------------------------- the picture
def _lines(cfg: Config, c: SimpleNamespace, v: dict, old: Optional[dict]) -> list[tuple[str, tuple, bool]]:
    """The text block: (text, colour, heading?)."""
    rim, sp, L = cfg.plate.rim, cfg.spoon, []

    def add(t, col=INK, head=False):
        L.append((t, col, head))

    add(f"Dry run: {c.name}" + (f" ({v['food']})" if v["food"] else ""), INK, True)
    add("Planner's prediction from config. Not checked on the real arm; servo sag is not included.", GREY)
    if c.cup is None:
        add(v["detail"], RED, True)
        return L
    cup, cc = c.cup, c.bowl.cup
    add("Cup", INK, True)
    add(f"centre x {cup.x:.0f}, y {cup.y:.0f} mm; opening {2 * cup.radius:.0f} mm wide; wall {cup.wall:.0f} mm; "
        f"inside bottom {cup.floor:.0f} mm above the table")
    if cup.rim_measured:
        add(f"rim height: {cup.rim:.0f} mm, measured")
    else:
        add(f"rim height: {cup.rim:.0f} mm ASSUMED, not measured", RED)
    add("position: confirmed with the spoon over it" if cc.confirmed else
        "position: NOT confirmed with the spoon", INK if cc.confirmed else RED)
    add(f"source: {cc.source or 'not recorded'}; position uncertainty: +-{cup.uncertainty:.0f} mm")
    add("Spoon", INK, True)
    add(("measured: " if sp.measured else "PLACEHOLDER numbers, not measured: ")
        + f"bowl {sp.bowl_length_mm:.0f} x {sp.bowl_width_mm:.0f} mm, {sp.bowl_depth_mm:.0f} mm deep; "
          f"neck {sp.neck_length_mm:.0f} x {sp.neck_width_mm:.0f} mm", INK if sp.measured else RED)
    add("Camera", INK, True)
    o = v["observation"]
    if o is None:
        add("not consulted in this dry run", GREY)
    else:
        add(f"{o['status']}" + (f" ({o['reason']})" if o["reason"] else "") + f", confidence {o['confidence']:.2f}",
            RED if o["status"] == "lost" else INK)
        if o["status"] == "moved":
            add(f"the cup moved {o['moved_mm']:.0f} mm: planned at the new place, uncertainty widened to "
                f"+-{cup.uncertainty:.0f} mm")
    add("Plan", INK, True)
    add(f"required gap: {c.margin:.1f} mm = {rim.clearance_mm:.1f} clearance + {cup.uncertainty:.1f} position "
        f"uncertainty; asked for {c.want_angle:.0f} deg entry, {c.want_drag:.0f} mm drag")
    if v["feasible"]:
        add("SAFE ENTRY FOUND", (0, 130, 0), True)
        add(f"entry angle: {v['entry_angle_deg']:.0f} deg below level")
        add(f"drag toward the arm: {v['drag_mm']:.0f} mm")
        add(f"dip height: tip {v['dip_z_mm']:.0f} mm above the table ({cup.rim - v['dip_z_mm']:.0f} mm below the rim)")
        add(f"smallest gap to the cup: {v['min_clearance_mm']:.1f} mm")
        add("approach from rest: " + ("not drawn (no rest pose saved)" if c.start is None else
                                      "direct" if not v["approach_poses"] else
                                      f"{v['approach_poses']} extra pose(s) to stay clear of the cup"))
    else:
        add(f"NO SAFE ENTRY: {v['reason']}", RED, True)
        add(v["detail"], RED)
        if c.err.reason != "unmeasured":
            add(f"grey spoon: NOT a plan. Straight down at the centre at {c.want_angle:.0f} deg, to show where "
                f"it would be too close.", GREY)
    for r in v["refusals"]:
        add(f"A real scoop would refuse: {r}", RED)
    if v["rejected"]:
        add("Entries tried and dropped", INK, True)
        for r in v["rejected"]:
            add("- " + r, GREY)
    if old is not None:
        add(f"Old path ({old['name']}, the saved scoop)", BROWN, True)
        add((f"touches the cup: {-old['min_mm']:.1f} mm into it with the spoon's {old['part']}" if old["min_mm"] < 0
             else f"smallest gap {old['min_mm']:.1f} mm ({old['part']}); the new plan requires {c.margin:.1f}"),
            RED if old["min_mm"] < c.margin else INK)
    add("Key", INK, True)
    if c.plan is not None:
        for name, col in POSE_COL.items():
            add(f"spoon at: {name}", col)
        add("black line: path of the spoon tip (grey: approach)")
    add("red dot: predicted collision;  orange ring: closer than the required gap", RED)
    if old is not None:
        add("brown dots: old path of the tip (brown spoon: its closest pose)", BROWN)
    return L


def _message(lines: list[str], w: int = 760, h: int = 220):
    img = np.full((h, w, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (20, 50 + 34 * i), FONT, 0.7 if i == 0 else 0.55, INK if i == 0 else RED, 1, AA)
    return img


def render(cfg: Config, bowl_name: str, observation=None, sighting_img=None):
    """BGR picture of the dry run for one bowl (see the module docstring). observation: a perception.rim
    Observation, if the camera was asked; sighting_img: the camera picture with RimObserver.draw's overlay already
    on it, shown as given. No arm, no camera, nothing is moved or written."""
    c = _case(cfg, bowl_name, observation)
    v = _verdict(cfg, c)
    if c.cup is None:
        return _message([f"Dry run: {bowl_name}", v["detail"]])
    kin, cup, spoon, rim = cfg.kinematics, c.cup, c.spoon, cfg.plate.rim
    drawable = c.err is None or c.err.reason != "unmeasured"      # unmeasured: a NaN or zero size, no shape to draw

    # what to draw: the plan's bodies and tip path, or the ghost; the old path against the same cup
    bodies, tip_path, app_path, hits, near, old, mark = [], None, None, [], [], None, None
    if c.plan is not None:
        p = c.plan
        names = ["hover", "pivot (entry angle)", "dip (lowest)"] + (["end of stroke"] if p.stroke else []) \
            + ["out", "clear"]
        keys = [p.hover, p.oriented, p.dip] + ([p.stroke[-1]] if p.stroke else []) + [p.out, p.clear]
        bodies = [(n, U.spoon_body(kin, q, spoon), POSE_COL[n], 2 if n in ("hover", "dip (lowest)", "end of stroke")
                   else 1) for n, q in zip(names, keys)]
        _, tip_path, cl, pts, _ = _scan(kin, p.poses(), cup, spoon)
        hits, near = list(pts[cl < 0]), list(pts[(cl >= 0) & (cl < c.margin)])
        mark = (pts[int(np.argmin(cl))], float(cl.min()))
        if c.start is not None:
            _, app_path, cl, pts, _ = _scan(kin, [c.start, *p.approach, p.hover], cup, spoon)
            hits += list(pts[cl < 0])
            near += list(pts[(cl >= 0) & (cl < c.margin)])
    elif drawable:
        body, h, n = _ghost(cup, spoon, c.want_angle, rim.hover_above_rim_mm, c.margin)
        bodies, hits, near = [("ghost", body, GREY, 1)], list(h), list(n)
    scoop = (c.bowl.scoop or f"scoop_{c.name}")
    if drawable and c.lib is not None and c.lib.has(scoop):
        try:
            Q, tips, cl, pts, j = _scan(kin, c.lib.path(scoop), cup, spoon)
            k = int(np.argmin(cl))
            old = {"name": scoop, "tips": tips, "hits": pts[cl < 0], "min_mm": float(cl[k]),
                   "part": U.PARTS[U._template(spoon)[3][int(j[k])]], "body": U.spoon_body(kin, _pose(Q[k]), spoon)}
        except Exception:
            old = None                                   # an unreadable old path is only a missing comparison

    text = _lines(cfg, c, v, old)
    tw = W - LEFT - 30
    rows = []
    for t, col, head in text:
        rows += [(s, col, head) for s in _wrap(t, tw, 0.5 if head else 0.42)]
    text_h = 30 + sum(26 if head else 18 for _, _, head in rows)
    inset = None
    if sighting_img is not None and getattr(sighting_img, "size", 0):
        f = tw / sighting_img.shape[1]
        inset = cv2.resize(sighting_img, (tw, max(1, int(round(sighting_img.shape[0] * f)))), interpolation=cv2.INTER_AREA)
        if inset.ndim == 2:
            inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
    H = max(TOP_H + SIDE_H + 110, text_h + (inset.shape[0] + 40 if inset is not None else 0) + 10)
    img = np.full((H, W, 3), 255, np.uint8)

    y = 28
    for s, col, head in rows:
        y += 26 if head else 18
        cv2.putText(img, s, (LEFT + 15, y), FONT, 0.5 if head else 0.42, col, 1, AA)
    if inset is not None:
        cv2.putText(img, "Camera view, with the opening as the camera found it", (LEFT + 15, y + 26), FONT, 0.42,
                    INK, 1, AA)
        img[y + 34:y + 34 + inset.shape[0], LEFT + 15:LEFT + 15 + tw] = inset[:, :, :3]
    cv2.line(img, (LEFT, 10), (LEFT, H - 10), LIGHT, 1)
    if not drawable:
        cv2.putText(img, "Nothing to draw: a cup or spoon dimension is missing (see the text).", (20, 60), FONT,
                    0.55, RED, 1, AA)
        return img

    # one frame for both views: `a` along the arm's line through the cup centre (arm on the left), `b` across it
    yaw = math.atan2(cup.y, cup.x)
    cy, sy = math.cos(yaw), math.sin(yaw)

    def ab(x, y):
        return (x - cup.x) * cy + (y - cup.y) * sy, -(x - cup.x) * sy + (y - cup.y) * cy

    Ro = cup.radius + cup.wall
    every = [pt for _, b, _, _ in bodies for pt in b]
    a_lo = max(-(Ro + 150.0), min([ab(pt.x, pt.y)[0] for pt in every] + [-(Ro + 30.0)]) - 10.0)
    a_hi = Ro + 30.0
    z_hi = min(cup.rim + 200.0, max([pt.z for pt in every] + [cup.rim + rim.hover_above_rim_mm + 30.0]) + 12.0)
    s = min(PANEL_W / (a_hi - a_lo), TOP_H / (2 * (Ro + 14.0)), SIDE_H / (z_hi + 10.0), 5.0)
    top = _View(img, 15, 40, PANEL_W, TOP_H, s, a_lo, -TOP_H / 2 / s)
    side = _View(img, 15, TOP_H + 85, PANEL_W, SIDE_H, s, a_lo, -10.0)
    cv2.putText(img, "TOP VIEW (looking down). The arm is to the left.", (15, 28), FONT, 0.5, INK, 1, AA)
    cv2.putText(img, "SIDE VIEW (cut through the cup centre along the arm's line). Heights in mm above the table.",
                (15, TOP_H + 73), FONT, 0.5, INK, 1, AA)

    def circle(r, col, thick):
        cv2.circle(top.img, top.px(0, 0), max(0, int(round(r * s))), col, thick, AA)

    # ---- top view
    circle(Ro, WALL, -1)
    circle(cup.radius, (255, 255, 255), -1)
    safe = cup.safe_radius(spoon.bowl_width / 2, rim.clearance_mm)
    if safe > 0:
        circle(safe, SAFE, -1)
        circle(safe, (0, 130, 0), 1)
    circle(cup.radius, WALL, 1)
    cv2.putText(top.img, f"dark ring: cup wall ({cup.wall:.0f} mm) around the {2 * cup.radius:.0f} mm opening", (8, 18),
                FONT, 0.4, WALL, 1, AA)
    cv2.putText(top.img, f"green: where the middle of the spoon's widest part may be (radius {safe:.0f} mm)"
                if safe > 0 else "no room: the spoon's widest part does not fit with the required gap", (8, 34),
                FONT, 0.4, (0, 130, 0) if safe > 0 else RED, 1, AA)
    cv2.putText(top.img, "toward the arm", (8, TOP_H - 22), FONT, 0.4, GREY, 1, AA)
    cv2.arrowedLine(top.img, (120, TOP_H - 12), (8, TOP_H - 12), GREY, 1, AA, tipLength=0.1)
    cv2.line(top.img, (PANEL_W - 20 - int(20 * s), TOP_H - 12), (PANEL_W - 20, TOP_H - 12), INK, 2)
    cv2.putText(top.img, "20 mm", (PANEL_W - 20 - int(20 * s), TOP_H - 18), FONT, 0.4, INK, 1, AA)

    def footprint(pts, col, thick):
        line, _ = _split(pts, spoon)
        left = [ab(q.x - q.half_width * math.sin(q.yaw), q.y + q.half_width * math.cos(q.yaw)) for q in line]
        right = [ab(q.x + q.half_width * math.sin(q.yaw), q.y - q.half_width * math.cos(q.yaw)) for q in line]
        top.line(left + right[::-1], col, thick, closed=True)

    def profile(pts, col, thick):
        line, under = _split(pts, spoon)
        la = [(ab(q.x, q.y)[0], q.z) for q in line]
        side.line(la, col, thick)
        if under:
            side.line([la[0]] + [(ab(q.x, q.y)[0], q.z) for q in under] + [la[len(under) + 1]], col, thick)

    # ---- side view: table, margin band, cup
    side.rect(a_lo, -10.0, a_lo + PANEL_W / s, 0.0, (225, 225, 225))
    m = c.margin
    for sgn in (-1, 1):
        lo, hi = sorted((sgn * (cup.radius - m), sgn * (Ro + m)))
        side.rect(lo, 0.0, hi, cup.rim + m, BAND)
    side.rect(-cup.radius, cup.floor, cup.radius, cup.floor + m, BAND)
    for sgn in (-1, 1):
        lo, hi = sorted((sgn * cup.radius, sgn * Ro))
        side.rect(lo, 0.0, hi, cup.rim, WALL)
    side.rect(-cup.radius, 0.0, cup.radius, cup.floor, WALL)
    side.line([(a_lo, 0.0), (a_lo + PANEL_W / s, 0.0)], INK, 1)
    side.label("table (0 mm)", a_lo, 0, INK, 4, 12)
    side.label(f"rim {cup.rim:.0f} mm" + ("" if cup.rim_measured else " (ASSUMED)"), Ro, cup.rim,
               WALL if cup.rim_measured else RED, 6, 12)
    side.label(f"inside bottom {cup.floor:.0f} mm", Ro, cup.floor, WALL, 6, 4)
    side.label(f"pale band: required gap {m:.1f} mm", Ro, cup.rim + m, (0, 110, 200), 6, -8)
    cv2.putText(side.img, "toward the arm", (8, 18), FONT, 0.4, GREY, 1, AA)

    # ---- old path, then the plan on top of it
    if old is not None:
        t = old["tips"]
        top.dots([ab(x, y) for x, y, _ in t], BROWN, 1)
        side.dots([(ab(x, y)[0], z) for x, y, z in t], BROWN, 1)
        profile(old["body"], BROWN, 1)
        footprint(old["body"], BROWN, 1)
        k = int(np.argmin(t[:, 2]))
        side.label("old path", ab(t[k, 0], t[k, 1])[0], t[k, 2], BROWN, -70, 14)
    for path, col in ((app_path, LIGHT), (tip_path, INK)):
        if path is not None:
            top.line([ab(x, y) for x, y, _ in path], col, 1)
            side.line([(ab(x, y)[0], z) for x, y, z in path], col, 1)
    for name, body, col, thick in bodies:
        profile(body, col, thick)
        if name in ("hover", "dip (lowest)", "end of stroke", "ghost"):
            footprint(body, col, thick)
    if c.plan is not None:
        ea = ab(*c.plan.entry_xy)
        cv2.drawMarker(top.img, top.px(*ea), RED, cv2.MARKER_CROSS, 14, 2)
        top.label("entry point (tip goes down here)", ea[0], ea[1], INK, 10, -8)
    old_hits = list(old["hits"]) if old is not None else []
    for view, f in ((top, lambda q: ab(q[0], q[1])), (side, lambda q: (ab(q[0], q[1])[0], q[2]))):
        view.dots([f(q) for q in near], ORANGE, 3, fill=False, step=max(1, len(near) // 150))
        view.dots([f(q) for q in hits + old_hits], RED, 3, step=max(1, (len(hits) + len(old_hits)) // 300))
    if mark is not None:                                 # the label sits in the corner, joined to the place by a line
        q, mm = mark
        at, corner = side.px(ab(q[0], q[1])[0], q[2]), (PANEL_W - 235, 22)
        cv2.circle(side.img, at, 6, INK, 1, AA)
        cv2.line(side.img, at, (corner[0] + 60, corner[1] + 5), INK, 1, AA)
        cv2.putText(side.img, f"smallest gap {mm:.1f} mm (circled)", corner, FONT, 0.45, INK, 1, AA)
    what = "this plan" if c.plan is not None else "the grey spoon"
    counts = f"red dots (predicted collisions): {len(hits)} on {what}" + (
        f", {len(old_hits)} on the old path" if old is not None else "")
    cv2.putText(side.img, counts, (8, 36), FONT, 0.45, RED if hits or old_hits else INK, 1, AA)
    for view in (top, side):
        cv2.rectangle(view.img, (0, 0), (PANEL_W - 1, view.h - 1), LIGHT, 1)
    return img


# ---------------------------------------------------------------- command line
def run(cfg: Config, bowls: Optional[list[str]] = None, out: Optional[str] = None) -> list[str]:
    """python -m spoonful dryrun: one picture and one verdict line per bowl. Default file: data/dryrun_<bowl>.png;
    with --out and several bowls the bowl's name is added to the file name."""
    names = list(bowls) if bowls else [b.name for b in cfg.plate.bowls]
    lines = [] if names else ["no bowls in config: map the table first"]
    for name in names:
        if out:
            path = Path(out)
            if len(names) > 1:
                path = path.with_name(f"{path.stem}_{name}{path.suffix or '.png'}")
        else:
            path = cfg.path("data") / f"dryrun_{name}.png"
        ok, buf = cv2.imencode(path.suffix or ".png", render(cfg, name))
        if not ok:
            lines.append(f"{name}: could not encode the picture as {path.suffix}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())
        lines.append(f"{verdict(cfg, name)['text']}  -> {path}")
    return lines
