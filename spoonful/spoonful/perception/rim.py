"""Is the cup's opening still where it was when the operator confirmed it? OpenCV only: no Gemini, no arm.

The table camera (the phone) is not calibrated and its pose is unknown, so this module never says where a cup IS in
the arm's frame. It answers a smaller question that a single uncalibrated view can answer: the rim of a cup is a
circle, seen as an ellipse; that ellipse was stored when the operator confirmed the spoon over the cup (the
reference); has it moved since, and if so by how many millimetres?

Pixels become millimetres with the ellipse itself as the ruler (the rim radius is known):
  semi-major axis px = the radius seen across the view      -> px per mm for a sideways (lateral) move
  semi-minor axis px = the radius seen along the view,
                       foreshortened by sin(elevation)      -> px per mm for a move toward/away from the camera
A move along the ellipse's major axis is lateral, a move along its minor axis is depth (up in the picture = further
from the camera). A ViewCalib (one angle) then turns camera lateral/depth into arm x/y.

This is the WEAK-PERSPECTIVE approximation: the cup is treated as small compared with its distance from the camera,
and the move as small compared with that distance too. It is exact only in the limit. It breaks when
  - the cup is close to the camera or the lens is wide (strong perspective): the near half of the rim is drawn
    larger than the far half, so the ellipse centre is not the picture of the rim centre and the axes are not the
    radius. The error grows like (radius / distance);
  - the move is long: the ruler changes along the way (the mean of the two rulers is used, which is first order);
  - the cup is far from where the ViewCalib was learned: "depth" means "along the line of sight to this cup", and
    lines of sight to cups at different places in the picture point in different table directions. One azimuth for
    the whole picture is wrong by the angle between them (several degrees across a phone's picture), and a move
    is then reported turned by that angle;
  - the camera is low: the minor axis is a few pixels, so one pixel is many millimetres of depth. Below
    `min_axis_ratio` depth is not reported at all (depth_ok False) and a cup that moved in depth is refused.
That is why the result is only used to follow small moves (max_follow_mm), and anything else is "lost". In the
synthetic pictures (tests/test_rim.py: cup about 0.5 m from the camera, two cups about 150 px apart) a followed
move came out wrong by roughly a tenth to a sixth of its length plus 1-2 mm; whoever follows a move should widen the
cup's uncertainty by at least that.

Nothing here has been run on the real table or on a picture of a real cup. Every threshold was chosen on those
synthetic pictures.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

REASONS = ("no_frame", "no_ellipse", "ambiguous", "partly_hidden", "too_flat", "low_contrast")
MIN_ARC = 0.55             # least fraction of the outline with an edge under it, for a known cup still in its place
FULL_ARC = 0.85            # ...and for a first sighting, or a cup seen somewhere new
SIMILAR = 0.8              # a second candidate scoring this close to the best one (elsewhere) = ambiguous
FLAT_RATIO = 0.12          # b/a below this (or b under MIN_B_PX) can't be told from a line
MIN_A_PX, MIN_B_PX = 10.0, 4.0
HERE = 0.1                 # "still in that very place": within this fraction of the rim radius
SIZE_CHANGE = 0.25         # the opening can't grow or shrink by more than this between two looks
CANNY = (30, 80)
SUPPORT_PX = 2.0           # an edge this close to the ellipse, running along it, supports it
MIN_RUN = 0.1              # ...but an arc shorter than this fraction of the outline is not counted as support
THUMB_W = 240              # width of the picture kept to tell a moved camera from a moved cup
BACKGROUND_RESPONSE = 0.25  # phase-correlation peak needed before the background shift is believed


@dataclass
class RimSighting:
    """The opening as an ellipse in image px. a >= b are the semi-axes; angle_deg is the major axis from the image
    x axis, in (-90, 90]. coverage: fraction of the ellipse (by arc length) with an edge under it."""
    cx: float
    cy: float
    a: float
    b: float
    angle_deg: float
    confidence: float
    stamp: float
    coverage: float


class RimNotFound(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# ---------------------------------------------------------------- ellipse helpers (cx, cy, a, b, theta rad)
def _norm(fit) -> Optional[tuple]:
    """cv2.fitEllipse's answer -> (cx, cy, a, b, theta) with a >= b and theta in (-pi/2, pi/2]."""
    (cx, cy), (w, h), ang = fit
    if not all(map(math.isfinite, (cx, cy, w, h, ang))) or min(w, h) <= 0:
        return None
    if w < h:
        w, h, ang = h, w, ang + 90.0
    ang = (ang + 90.0) % 180.0 - 90.0
    return cx, cy, w / 2, h / 2, math.radians(90.0 if ang == -90.0 else ang)


def _fit(pts) -> Optional[tuple]:
    import cv2
    if len(pts) < 6:
        return None
    try:
        return _norm(cv2.fitEllipse(np.ascontiguousarray(pts, np.float32)))
    except cv2.error:
        return None


def _radial(P, e):
    """For points P (N x 2): distance to the ellipse along the ray from its centre, and the ellipse's unit normal
    there."""
    cx, cy, a, b, th = e
    c, s = math.cos(th), math.sin(th)
    dx, dy = P[:, 0] - cx, P[:, 1] - cy
    qx, qy = c * dx + s * dy, -s * dx + c * dy
    rho = np.hypot(qx / a, qy / b) + 1e-9
    dist = np.hypot(qx, qy) * np.abs(1 - 1 / rho)
    nx, ny = qx / (a * a), qy / (b * b)
    n = np.hypot(nx, ny) + 1e-12
    return dist, np.column_stack([(c * nx - s * ny) / n, (s * nx + c * ny) / n])


def _refine(P, G, e) -> Optional[tuple]:
    """Refit to the edge pixels that lie on the ellipse and run along it, in a narrowing band. This is what pulls a
    rough guess (from one arc, or from the expected shape) onto the whole visible rim."""
    for tol in (0.2 * e[3] + 2.0, 4.0, 2.5):
        dist, n = _radial(P, e)
        m = (dist <= tol) & (np.abs((n * G).sum(1)) >= 0.7)
        if m.sum() < 12:
            return None
        e = _fit(P[m])
        if e is None or e[2] > 4000:
            return None
    return e


def _support(dist_map, gx, gy, e, n: int = 180):
    """(coverage, mean distance of the supported samples to their edge, support in each quarter of the ellipse: the
    two ends of the major axis, then the two long sides). A sample of the ellipse is supported when
    an edge lies within SUPPORT_PX of it and the picture's gradient there points across the ellipse (so texture
    lines that merely cross the ellipse don't count)."""
    cx, cy, a, b, th = e
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = math.cos(th), math.sin(th)
    ex, ey = a * np.cos(t), b * np.sin(t)
    x, y = cx + c * ex - s * ey, cy + s * ex + c * ey
    nx, ny = b * np.cos(t), a * np.sin(t)
    nn = np.hypot(nx, ny)
    nx, ny = (c * nx - s * ny) / nn, (s * nx + c * ny) / nn
    w = np.hypot(a * np.sin(t), b * np.cos(t))                     # arc length each sample stands for
    H, W = dist_map.shape
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi, yi = np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)
    d = dist_map[yi, xi]
    g0, g1 = gx[yi, xi], gy[yi, xi]
    along = np.abs(g0 * nx + g1 * ny) / (np.hypot(g0, g1) + 1e-6)
    ok = inside & (d <= SUPPORT_PX) & (along >= 0.7)
    ok = ok | (np.roll(ok, 1) & np.roll(ok, -1))                   # a single missed sample doesn't break an arc
    if not ok.all() and ok.any():
        # only arcs of some length count: a rim that is partly hidden still shows as a few long arcs, while lines
        # that touch or cross the ellipse (the cup's sides and base, a pattern on the table) give short scraps
        first = int(np.argmin(ok))
        r_ok, r_w = np.roll(ok, -first), np.roll(w, -first)
        edge = np.flatnonzero(np.diff(np.concatenate([[False], r_ok, [False]]).astype(np.int8)))
        for i0, i1 in zip(edge[::2], edge[1::2]):
            if r_w[i0:i1].sum() < MIN_RUN * w.sum():
                r_ok[i0:i1] = False
        ok = np.roll(r_ok, first)
    k = math.sqrt(0.5)
    parts = [float(ok[m].mean()) for m in (np.cos(t) > k, np.cos(t) < -k, np.sin(t) > k, np.sin(t) < -k)]
    return float((w * ok).sum() / w.sum()), (float(d[ok].mean()) if ok.any() else SUPPORT_PX), parts


def _prep(crop):
    """Edges of a BGR crop, taken per pixel from whichever colour channel changes most (food against a white cup
    can differ in colour more than in brightness)."""
    import cv2
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    gxs, gys = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    if gxs.ndim == 3:
        k = (gxs * gxs + gys * gys).argmax(2)[..., None]
        gxs, gys = np.take_along_axis(gxs, k, 2)[..., 0], np.take_along_axis(gys, k, 2)[..., 0]
    edges = cv2.Canny(gxs.astype(np.int16), gys.astype(np.int16), *CANNY)
    ys, xs = np.nonzero(edges)
    P = np.column_stack([xs, ys]).astype(np.float32)
    G = np.column_stack([gxs[ys, xs], gys[ys, xs]])
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-6)
    dist_map = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return edges, P, G, dist_map, gxs, gys


def _arcs(edges) -> tuple[list, list]:
    """(whole contours, smooth pieces of them cut at sharp corners), longest first. The outline of an opening that
    something crosses is an arc plus a straight chord; cutting at the corners gives the arc back."""
    import cv2
    whole, pieces = [], []
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    k = 4
    for c in cnts:
        p = c[:, 0, :].astype(np.float32)
        if len(p) < 24:
            continue
        whole.append(p)
        d1, d2 = p - np.roll(p, k, 0), np.roll(p, -k, 0) - p
        turn = np.abs(np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(1)))
        corner = turn > math.radians(50)
        if not corner.any():
            continue
        first = int(np.argmax(corner))
        p, corner = np.roll(p, -first, 0), np.roll(corner, -first)
        for seg in np.split(p, np.flatnonzero(corner)):
            if len(seg) >= 16 + k:
                pieces.append(seg[k:])
    key = lambda q: -len(q)
    return sorted(whole, key=key), sorted(pieces, key=key)


def _vote(edges, expect: RimSighting) -> list[tuple]:
    """Where could an ellipse of the expected shape sit? Slide its outline over the edge picture (three sizes) and
    keep the places where much of it lands on edges. Works when contours are broken or something crosses the rim."""
    import cv2
    E = cv2.dilate(edges, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
    out = []
    for k in (0.87, 1.0, 1.15):
        a, b = expect.a * k, expect.b * k
        half = int(a + 4)
        if 2 * half + 1 > min(E.shape):
            continue
        tmpl = np.zeros((2 * half + 1, 2 * half + 1), np.float32)
        cv2.ellipse(tmpl, (half, half), (int(round(a)), int(round(b))), expect.angle_deg, 0, 360, 1.0, 2)
        r = cv2.matchTemplate(E, tmpl, cv2.TM_CCORR) / float(tmpl.sum())
        for _ in range(4):
            _, val, _, (px, py) = cv2.minMaxLoc(r)
            if val < 0.35:
                break
            out.append((px + half, py + half, a, b, math.radians(expect.angle_deg)))
            cv2.circle(r, (px, py), max(3, int(b * 0.6)), 0.0, -1)
    return out


@dataclass
class _Cand:
    score: float                   # coverage x match to the expected size and aspect
    conf: float
    cov: float
    e: tuple                       # (cx, cy, a, b, theta) in image px
    here: bool                     # still where `expect` was (only judged when a reach is given)
    enough: bool                   # enough of its outline shows for it to be believed
    match: float = 1.0             # 1 = exactly the expected size and aspect


def _same(e, f) -> bool:
    """Two fits of the same ring (for example the inner and outer edge of the lip)."""
    return math.hypot(e[0] - f[0], e[1] - f[1]) <= max(3.0, 0.25 * min(e[3], f[3])) and abs(e[2] - f[2]) <= 0.15 * f[2]


def find_rim(img, roi=None, expect: Optional[RimSighting] = None, min_confidence: float = 0.55,
             ignore=(), stamp: Optional[float] = None, reach_px: Optional[float] = None) -> RimSighting:
    """The opening of the one cup in `roi` (x, y, w, h in the px of `img`; None = whole picture), as a fitted
    ellipse. Raises RimNotFound(reason) rather than guess: the answer is always an ellipse fitted to rim edges,
    never the middle of a box or a blob.

    Edges -> contours, cut into smooth arcs -> ellipse fits of contours, arcs and pairs of arcs (plus, with
    `expect`, the places where an ellipse of the expected shape lands on edges) -> each refitted to the edge pixels
    lying along it -> scored by the fraction of the ellipse that has an edge under it, and by size and aspect
    against `expect`. A rim seen from above the table is wider than tall with a roughly level major axis (or the
    axis `expect` had). Two different ellipses that score alike are "ambiguous".

    How much of the outline must show: FULL_ARC, because a first sighting, or a sighting somewhere new, has
    nothing to be held against, and a half-seen ellipse is as likely to be the outline of the cup's body, the part
    of the opening beside whatever crosses it, or the cup's BASE under the straight edge of whatever hides the rim
    (all of these came up in the synthetic pictures, with the right size and aspect). Only with `reach_px` (which
    the observer passes) is a partly hidden opening accepted, down to MIN_ARC and at reduced confidence, and only
    where `expect` was: ellipses centred further than reach_px from there are not candidates (measured as a
    sideways move; up/down the picture counts a/b times as much), and one still in that very place is the answer
    whatever else is near. Less than that is "partly_hidden".
    `ignore`: sightings of OTHER known cups; an ellipse sitting on one of them is not a candidate."""
    if img is None or getattr(img, "size", 0) == 0:
        raise RimNotFound("no_frame")
    H, W = img.shape[:2]
    if roi is None:
        x0, y0, x1, y1, min_a = 0, 0, W, H, MIN_A_PX
    else:
        x, y, w, h = (float(v) for v in roi)
        pad = max(6.0, 0.1 * max(w, h))
        x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
        x1, y1 = min(W, int(x + w + pad)), min(H, int(y + h + pad))
        # the box is around one cup, so its opening spans a good part of it: crumbs and print on the cup don't
        min_a = max(MIN_A_PX, 0.25 * w) if expect is None else MIN_A_PX
    if x1 - x0 < 24 or y1 - y0 < 16:
        raise RimNotFound("no_frame", "the cup's part of the picture is outside the frame")
    edges, P, G, dist_map, gx, gy = _prep(img[y0:y1, x0:x1])
    if len(P) < 40:
        raise RimNotFound("low_contrast", "almost no edges in this part of the picture")

    whole, pieces = _arcs(edges)
    pieces = pieces[:12]
    groups = whole[:40] + pieces + [np.vstack([p, q]) for i, p in enumerate(pieces[:9]) for q in pieces[i + 1:9]]
    seeds = [e for e in map(_fit, groups) if e is not None]
    if expect is not None:
        seeds += _vote(edges, expect)
    ch, cw = edges.shape
    lean = math.radians(expect.angle_deg) if expect is not None else 0.0
    cands, flat, seen = [], [], set()
    for e in seeds:
        if not (6 <= e[2] <= cw and 2 <= e[3] and -e[2] < e[0] < cw + e[2] and -e[2] < e[1] < ch + e[2]):
            continue
        e = _refine(P, G, e)
        if e is None:
            continue
        key = tuple(int(round(v / 2)) for v in e[:4])
        if key in seen:
            continue
        seen.add(key)
        cx, cy, a, b, th = e
        if a < min_a or not (0 <= cx - 0.9 * a and cx + 0.9 * a <= cw and 0 <= cy < ch):
            continue                                              # too small, or mostly outside this crop
        cov, resid, parts = _support(dist_map, gx, gy, e)
        if cov < 0.3 or max(parts[:2]) < 0.6:
            continue                                              # two near-parallel lines: sides but no round end
        if b / a < 0.85 and abs((th - lean + math.pi / 2) % math.pi - math.pi / 2) > math.radians(30):
            continue                                              # a tilted ellipse is not a rim on the table
        match = 1.0
        if expect is not None:
            size, aspect = abs(math.log(a / expect.a)), abs(math.log((b / a) / (expect.b / expect.a)))
            if size > 0.4 or aspect > 0.35:
                continue
            match = 1.0 - 0.6 * size - 0.4 * aspect
        full = (cx + x0, cy + y0, a, b, th)
        here = False
        if expect is not None:
            d, (u, v) = np.array([full[0] - expect.cx, full[1] - expect.cy]), _axes(expect)
            # distance counted as a sideways move: a px up or down the picture is a/b times as many mm
            away = math.hypot(d @ u, (d @ v) * expect.a / expect.b)
            if reach_px is not None and away > reach_px:
                continue
            here = reach_px is not None and away <= max(2.0, HERE * expect.a)
        if any(_same(full, (o.cx, o.cy, o.a, o.b, 0.0)) for o in ignore):
            continue
        conf = cov * match * min(1.0, max(0.6, 1.15 - resid / SUPPORT_PX * 0.3))
        c = _Cand(cov * match, conf, cov, full, here, cov >= (MIN_ARC if here else FULL_ARC), match)
        (flat if (b < MIN_B_PX or b / a < FLAT_RATIO) else cands).append(c)

    cands.sort(key=lambda c: -c.score)
    best = next((c for c in cands if c.enough), None)
    stay = next((c for c in cands if c.enough and c.here), None)
    if stay is not None and (stay.match >= best.match - 0.04
                             or not (_nested(stay.e, best.e) or _nested(best.e, stay.e))):
        # Still in its place: that is the answer even if a neighbouring cup looks better, or if the outline of its
        # own body shows more completely. But a ring of the wrong shape in the old place does not outvote a
        # better-shaped ring of the same cup a little way off: the cup moved a little, and its contents (or bits
        # of its edges) now sit where its opening was.
        best = stay
    if best is None:
        if any(c.enough for c in flat):
            raise RimNotFound("too_flat", "the camera is too low to see into the cup")
        if cands:
            raise RimNotFound("partly_hidden", f"only {cands[0].cov:.0%} of an opening is visible")
        raise RimNotFound("no_ellipse")
    if not best.here:
        # One cup shows several rings, one inside the other: the visible food, the opening, the outline of the
        # body. Seen from above the table the opening is the highest of them in the picture.
        family = [c for c in cands if c is best or _nested(c.e, best.e) or _nested(best.e, c.e)]
        if expect is None:
            good = [c for c in family if c.enough and c.score >= SIMILAR * best.score]
            top = min(c.e[1] for c in good)
            # (the part of an opening beside something thin that crosses it is a good ring too, at the same height:
            # of those, the widest is the opening)
            best = max((c for c in good if c.e[1] <= top + max(2.0, 0.08 * c.e[3])), key=lambda c: c.e[2])
        if any(not c.enough and c.cov >= 0.6 * best.cov and c.e[2] >= 0.6 * best.e[2]
               and c.e[1] < best.e[1] - max(3.0, 0.25 * best.e[3]) for c in family):
            raise RimNotFound("partly_hidden", "a half-seen ring above the best whole one may be the real opening")
        for c in cands:
            if c is best or c.score < SIMILAR * best.score or _same(c.e, best.e):
                continue
            if _nested(c.e, best.e) or _nested(best.e, c.e):
                continue                                          # another ring of the same cup
            if not c.enough:
                raise RimNotFound("partly_hidden", "a half-seen ring here fits as well as the best whole one")
            raise RimNotFound("ambiguous", "two openings look alike here")
    if best.conf < min_confidence:
        raise RimNotFound("partly_hidden" if best.cov < FULL_ARC else "low_contrast", f"confidence {best.conf:.2f}")
    cx, cy, a, b, th = best.e
    return RimSighting(float(cx), float(cy), float(a), float(b), math.degrees(th), round(float(best.conf), 3),
                       time.time() if stamp is None else stamp, round(float(best.cov), 3))


def _nested(inner, outer) -> bool:
    """`inner` is a smaller ellipse centred inside `outer`. (Two cups side by side, or one behind the other, are
    centred at least a whole opening apart.)"""
    if inner[2] > outer[2]:
        return False
    c, s = math.cos(outer[4]), math.sin(outer[4])
    dx, dy = inner[0] - outer[0], inner[1] - outer[1]
    return math.hypot((c * dx + s * dy) / outer[2], (-s * dx + c * dy) / outer[3]) < 1.0


# ---------------------------------------------------------------- pixels -> millimetres
def _axes(s: RimSighting):
    """Unit vectors in the image: lateral (along the major axis, pointing right) and depth (along the minor axis,
    pointing up the picture = away from the camera). A near-round ellipse has no reliable axis: use the image's."""
    th = math.radians(s.angle_deg) if s.b / s.a < 0.9 else 0.0
    u = np.array([math.cos(th), math.sin(th)])
    if u[0] < 0:
        u = -u
    return u, np.array([u[1], -u[0]])


def _ruler(p: RimSighting, q: RimSighting, rp: float, rq: float):
    """px per mm across the view and along it, as the mean of two sightings."""
    return (p.a / rp + q.a / rq) / 2, (p.b / rp + q.b / rq) / 2


def displacement_mm(ref: RimSighting, now: RimSighting, radius_mm: float, min_axis_ratio: float
                    ) -> tuple[float, float, bool]:
    """(lateral_mm, depth_mm, depth_ok): how far the opening moved from `ref` to `now`, in the camera's own two
    table directions. lateral > 0 = to the right in the picture; depth > 0 = away from the camera.

    The ellipse is the ruler (weak perspective, see the module docstring). depth_ok is False when the opening is
    seen flatter than min_axis_ratio: depth_mm is then only a hint that something changed along the view (one px is
    many mm) and must not be used as a measurement."""
    u, v = _axes(ref)
    d = np.array([now.cx - ref.cx, now.cy - ref.cy])
    across, along = _ruler(ref, now, radius_mm, radius_mm)
    ok = min(ref.b / ref.a, now.b / now.a) >= min_axis_ratio
    return float(d @ u / across), float(d @ v / along), bool(ok)


@dataclass
class ViewCalib:
    """Camera lateral/depth -> arm x/y. azimuth_deg: the table direction (atan2(y, x) in the arm's frame) the camera
    looks along, i.e. which way "further from the camera" is. mirrored: the picture is flipped left-right (a
    selfie-style stream). residual_mm: how badly the evidence it came from disagreed with itself."""
    azimuth_deg: float
    mirrored: bool
    source: str
    residual_mm: float
    stamp: float

    def to_table(self, lateral_mm: float, depth_mm: float) -> tuple[float, float]:
        az = math.radians(self.azimuth_deg)
        lat = -lateral_mm if self.mirrored else lateral_mm
        # looking along +x, the right of the picture is -y
        return (depth_mm * math.cos(az) + lat * math.sin(az), depth_mm * math.sin(az) - lat * math.cos(az))


def calib_from_pair(ref_a: RimSighting, cup_a_xy, radius_a: float, ref_b: RimSighting, cup_b_xy, radius_b: float,
                    *, min_axis_ratio: float = 0.25, tol_frac: float = 0.2, min_sep_mm: float = 50.0,
                    mirrored: bool = False, stamp: Optional[float] = None) -> Optional[ViewCalib]:
    """Two confirmed cups in one picture: the vector between them as the camera sees it (lateral, depth, in mm by
    the two ellipses' rulers) against the same vector in the arm's frame gives the azimuth. None (not trusted) when
    the two lengths differ by more than tol_frac, when the cups are too close together to give a direction, or when
    either opening is too flat to measure depth.

    One pair cannot tell a mirrored picture from a normal one (either fits with a different angle), so that is an
    input, not a result."""
    if min(ref_a.b / ref_a.a, ref_b.b / ref_b.a) < min_axis_ratio:
        return None
    t = complex(cup_b_xy[0] - cup_a_xy[0], cup_b_xy[1] - cup_a_xy[1])
    if abs(t) < min_sep_mm:
        return None
    ua, va = _axes(ref_a)
    ub, vb = _axes(ref_b)
    u, v = ua + ub, va + vb
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    d = np.array([ref_b.cx - ref_a.cx, ref_b.cy - ref_a.cy])
    across, along = _ruler(ref_a, ref_b, radius_a, radius_b)
    lat, dep = float(d @ u / across), float(d @ v / along)
    seen = math.hypot(lat, dep)
    if seen < 1e-6 or abs(seen - abs(t)) > tol_frac * abs(t):
        return None
    c = complex(dep, lat if mirrored else -lat)                    # table vector = e^(i az) * c
    az = math.degrees(math.atan2(t.imag, t.real) - math.atan2(c.imag, c.real))
    return ViewCalib((az + 180.0) % 360.0 - 180.0, mirrored, "pair", abs(seen - abs(t)),
                     time.time() if stamp is None else stamp)


def calib_from_jacobian(J, yaw_deg: float = 0.0, axis_deg: float = 0.0, stamp: Optional[float] = None
                        ) -> Optional[ViewCalib]:
    """From AimCheck's image Jacobian for a bowl (rows: px per mm of spoon motion further / left / up; only the
    first two are used). yaw_deg: the arm's base yaw at that bowl ("further" is radial, so it points along yaw_deg
    in the table frame). axis_deg: direction of the openings' major axis in the picture (0 = level camera).

    None when the camera barely sees one table direction (same rule as aim_check.solve_offset) or when the two
    directions it implies are far from square to each other, which a weak-perspective view of a table would give.
    The Jacobian must be in px of an undistorted-aspect picture (AimCheck works on a 1280x720 resize)."""
    J = np.asarray(J, float)
    if J.ndim != 2 or J.shape[0] < 2 or J.shape[1] != 2 or not np.isfinite(J[:2]).all():
        return None
    M = J[:2].T                                                   # px = M @ (further_mm, left_mm)
    S = np.linalg.svd(M, compute_uv=False)
    if S[0] < 0.05 or S[1] < 0.2 * S[0]:
        return None
    th = math.radians(axis_deg)
    Mi = np.linalg.inv(M)
    depth, lateral = Mi @ np.array([math.sin(th), -math.cos(th)]), Mi @ np.array([math.cos(th), math.sin(th)])
    depth, lateral = depth / np.linalg.norm(depth), lateral / np.linalg.norm(lateral)
    skew = abs(float(depth @ lateral))
    if skew > math.sin(math.radians(25)):
        return None
    mirrored = depth[0] * lateral[1] - depth[1] * lateral[0] > 0    # the picture's right should be the view's right
    az = math.degrees(math.atan2(depth[1], depth[0])) + yaw_deg
    return ViewCalib((az + 180.0) % 360.0 - 180.0, bool(mirrored), "jacobian", 20.0 * skew,
                     time.time() if stamp is None else stamp)


def spoon_over(before, after, s: RimSighting, min_px: int = 120) -> dict:
    """Does the camera see the spoon above this opening? `before` is the picture taken with the arm at rest, `after`
    the one with the spoon hovering. What changed in the band above the opening is the arm; its lowest point is the
    spoon tip (seen from the side). Reports where that is across the opening, in rim half-widths (0 = middle,
    +-1 = the rim's edges): the camera can judge left-right this way, not depth.

    A plain picture difference, no model: anything else that moved in that band (the diner behind the table, a
    shadow) counts too, so this is a second opinion on the encoders, not a measurement."""
    import cv2
    if before is None or after is None or before.shape != after.shape:
        return {"seen": False, "why": "no comparable pictures"}
    g0 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    g1 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    H, W = g0.shape
    x0, x1 = int(max(0, s.cx - 2.5 * s.a)), int(min(W, s.cx + 2.5 * s.a))
    y0, y1 = int(max(0, s.cy - 7.0 * s.a)), int(min(H, s.cy - 0.2 * s.b))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return {"seen": False, "why": "the opening is at the edge of the picture"}
    d = (cv2.absdiff(g0, g1)[y0:y1, x0:x1] > 28).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(d)
    if len(ys) < min_px:
        return {"seen": False, "why": "nothing new above the opening"}
    low = ys.max()
    near = ys >= low - 8
    xt, yt = float(np.median(xs[near])) + x0, float(low + y0)
    off = (xt - s.cx) / max(1.0, s.a)
    return {"seen": True, "tip_px": [round(xt), round(yt)], "off": round(off, 2), "over": abs(off) <= 1.0,
            "changed_px": int(len(ys))}


# ---------------------------------------------------------------- the observer
@dataclass
class Observation:
    status: str                    # "ok" (where it was) | "moved" | "lost"
    bowl: str
    dx: float = 0
    dy: float = 0                  # table mm, from the reference to where the opening is now (0 when "lost")
    confidence: float = 0
    stamp: float = 0
    reason: str = ""               # for "lost": no_frame / stale_frame / no_reference / reference_stale /
                                   #   a RimNotFound reason / moved_unmeasurable / moved_too_far / camera_moved /
                                   #   size_changed
    sighting: Optional[RimSighting] = None


@dataclass
class _Ref:
    sighting: RimSighting
    roi: tuple                     # x, y, w, h of the cup in the picture when confirmed
    cup_xy: tuple
    radius: float
    stamp: float                   # when the operator confirmed (following a move does not renew it)
    thumb: Optional[np.ndarray]    # small grey copy of that picture, to tell a moved camera from a moved cup
    shape: tuple                   # (h, w) of that picture
    followed: bool = False


def _thumb(img):
    import cv2
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (THUMB_W, max(8, round(h * THUMB_W / w))), interpolation=cv2.INTER_AREA).astype(np.float32)


class RimObserver:
    """Keeps, per bowl, the opening as the camera saw it when the operator confirmed the cup, and compares each new
    picture with it. In memory only: a restart, or a phone that was moved, needs a new Confirm.

    frame_fn() -> BGR picture or None. frame_age_fn() -> seconds since that picture arrived; without it, a picture
    object that has been handed over unchanged for longer than view_max_age_s is treated as stale.
    cfg: the app Config (cfg.plate.rim is used) or a RimCfg."""

    def __init__(self, cfg, frame_fn: Callable, frame_age_fn: Optional[Callable[[], float]] = None,
                 clock: Callable[[], float] = time.time):
        self.rc = getattr(getattr(cfg, "plate", None), "rim", cfg)
        self.frame_fn, self.frame_age_fn, self.clock = frame_fn, frame_age_fn, clock
        self.refs: dict[str, _Ref] = {}
        self.calib: Optional[ViewCalib] = None
        self.bowl_calib: dict[str, ViewCalib] = {}
        self.last: dict[str, Observation] = {}
        self.mirrored = False                       # set True if the table camera's stream is flipped left-right
        self._lock = threading.RLock()
        self._frame_seen = (None, 0.0)              # (picture object, when it was first handed over)
        self._moved_thumb: dict[str, tuple] = {}    # bowl -> (stamp, small copy of the picture it was seen moved in)

    # -- kept across restarts of the app (the cups and the phone have not moved just because the program did)
    def persist(self, path) -> int:
        """Load what was saved at `path` (if anything) and save there from now on. References older than
        reference_max_age_h are refused later by observe(), as always; a cup that moved while the app was off is
        seen as moved or lost against its old reference, never silently re-learned. Returns how many were loaded."""
        import pickle
        self._path = path
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self.refs = dict(data.get("refs", {}))
                self.mirrored = bool(data.get("mirrored", self.mirrored))
                try:
                    self.recalibrate()
                except Exception:
                    pass
            return len(self.refs)
        except Exception:
            return 0

    def _save(self) -> None:
        path = getattr(self, "_path", None)
        if path is None:
            return
        import pickle
        try:
            with self._lock:
                data = {"refs": dict(self.refs), "mirrored": self.mirrored}
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass

    # -- references
    def set_reference(self, bowl: str, img, roi, cup_xy, radius_mm: float) -> RimSighting:
        """The operator has just confirmed this cup at cup_xy (arm frame). Store how its opening looks now. Raises
        RimNotFound (and keeps no reference for the bowl) when the camera can't see the opening clearly."""
        with self._lock:
            self.refs.pop(bowl, None)
            self.last.pop(bowl, None)
            self.bowl_calib.pop(bowl, None)
            if img is None:
                raise RimNotFound("no_frame")
            now = self.clock()
            s = find_rim(img, roi, None, self.rc.min_confidence, ignore=[r.sighting for r in self.refs.values()],
                         stamp=now)
            self.refs[bowl] = _Ref(s, tuple(float(v) for v in roi) if roi is not None else self._box(s, 0.0),
                                   (float(cup_xy[0]), float(cup_xy[1])), float(radius_mm), now, _thumb(img),
                                   img.shape[:2])
            # only cups the camera still sees where they were confirmed may teach it the view
            still = [bowl]
            for name, r in self.refs.items():
                if name != bowl and not r.followed:
                    try:
                        o = find_rim(img, self._search_roi(r), r.sighting, self.rc.min_confidence, ignore=[s],
                                     reach_px=self._reach(r))
                    except RimNotFound:
                        continue
                    if math.hypot(o.cx - r.sighting.cx, o.cy - r.sighting.cy) <= 3.0:
                        still.append(name)
            self.recalibrate(only=still)
            self._save()
            return s

    def has_reference(self, bowl: str) -> bool:
        return bowl in self.refs

    def forget(self, bowl: Optional[str] = None) -> None:
        """The cup was re-placed (one bowl: the camera has not moved, so what was learned about the view stays) or
        the camera/table changed (everything, the view calibration too)."""
        with self._lock:
            if bowl is None:
                self.refs, self.last, self.bowl_calib, self.calib, self._moved_thumb = {}, {}, {}, None, {}
            else:
                for d in (self.refs, self.last, self.bowl_calib, self._moved_thumb):
                    d.pop(bowl, None)
            self._save()

    # -- calibration of the view
    def set_calib(self, calib: Optional[ViewCalib], bowl: Optional[str] = None) -> None:
        """Hand over a view calibration (for example calib_from_jacobian). With `bowl`, it is used for that bowl
        only: a Jacobian measured over one cup is right for the line of sight to that cup."""
        with self._lock:
            if bowl is None:
                self.calib = calib
            elif calib is None:
                self.bowl_calib.pop(bowl, None)
            else:
                self.bowl_calib[bowl] = calib

    def recalibrate(self, cups: Optional[dict] = None, only=None) -> Optional[ViewCalib]:
        """Learn the view from every pair of confirmed cups. cups: bowl -> (x, y, radius) in the arm frame (default:
        what set_reference was given). All pairs must give the same azimuth (within 15 deg), else the view is not
        trusted. With no pair to judge by, what was known stays. Pairs that disagree clear a calibration that came
        from pairs; one handed over with set_calib is only replaced by a good answer."""
        with self._lock:
            names = [n for n, r in self.refs.items() if not r.followed and (only is None or n in only)
                     and (cups is None or n in cups)]
            found = []
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    ra, rb = self.refs[a], self.refs[b]
                    xa, ya, rad_a = cups[a] if cups else (*ra.cup_xy, ra.radius)
                    xb, yb, rad_b = cups[b] if cups else (*rb.cup_xy, rb.radius)
                    found.append(calib_from_pair(ra.sighting, (xa, ya), rad_a, rb.sighting, (xb, yb), rad_b,
                                                 min_axis_ratio=self.rc.min_axis_ratio, mirrored=self.mirrored,
                                                 stamp=self.clock()))
            if not found:
                return self.calib
            calib = None
            if all(c is not None for c in found):
                z = sum(complex(math.cos(math.radians(c.azimuth_deg)), math.sin(math.radians(c.azimuth_deg)))
                        for c in found)
                az = math.degrees(math.atan2(z.imag, z.real))
                if all(abs((c.azimuth_deg - az + 180) % 360 - 180) <= 15 for c in found):
                    calib = replace(found[0], azimuth_deg=az, residual_mm=max(c.residual_mm for c in found))
            if calib is not None or self.calib is None or self.calib.source == "pair":
                self.calib = calib
            return calib

    # -- looking
    def _box(self, s: RimSighting, grow_px: float) -> tuple:
        g = grow_px + 0.3 * s.a
        return (s.cx - s.a - g, s.cy - s.a - g, 2 * (s.a + g), 2 * (s.a + g))

    def _reach(self, r: _Ref) -> float:
        """How far (px) this cup's opening is looked for: a little beyond what may be followed, so that "moved too
        far" can be told from "gone"."""
        return 1.3 * self.rc.max_follow_mm * r.sighting.a / r.radius

    def _search_roi(self, r: _Ref) -> tuple:
        return self._box(r.sighting, self._reach(r))

    def _frame(self):
        """(picture, reason it can't be used)."""
        img = self.frame_fn()
        if img is None:
            return None, "no_frame"
        now = self.clock()
        if self.frame_age_fn is not None:
            age = self.frame_age_fn()
        else:
            if self._frame_seen[0] is not img:
                self._frame_seen = (img, now)
            age = now - self._frame_seen[1]
        if age is None or age > self.rc.view_max_age_s:
            return None, "stale_frame"
        return img, ""

    def _camera_moved(self, bowl: str, img, s: RimSighting) -> bool:
        """The cup's picture moved. Did everything else move with it? Other known cups are asked first; with none
        in sight, the rest of the picture (the cups blanked out) is compared with the picture at confirm time."""
        import cv2
        ref = self.refs[bowl]
        shift = np.array([s.cx - ref.sighting.cx, s.cy - ref.sighting.cy])
        near = 0.35 * float(np.linalg.norm(shift)) + 3.0
        others = []
        for name, r in self.refs.items():
            if name == bowl:
                continue
            try:
                o = find_rim(img, self._search_roi(r), r.sighting, self.rc.min_confidence, ignore=[s],
                             reach_px=self._reach(r))
            except RimNotFound:
                continue
            others.append(np.array([o.cx - r.sighting.cx, o.cy - r.sighting.cy]))
        if others:
            return all(float(np.linalg.norm(o - shift)) <= near for o in others)
        if ref.thumb is None or img.shape[:2] != ref.shape:
            return False
        a, b = ref.thumb.copy(), _thumb(img)
        k = THUMB_W / img.shape[1]
        x, y, w, h = ref.roi
        for ox, oy in ((0.0, 0.0), shift):                          # the cup, where it was and where it is
            p0 = (int((x + ox - 0.15 * w) * k), int((y + oy - 0.15 * h) * k))
            p1 = (int((x + ox + 1.15 * w) * k) + 1, int((y + oy + 1.15 * h) * k) + 1)
            for im in (a, b):
                cv2.rectangle(im, p0, p1, float(im.mean()), -1)
        if min(a.std(), b.std()) < 3.0:                             # a plain table says nothing either way
            return False
        (gx, gy), response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
        return response >= BACKGROUND_RESPONSE and float(np.linalg.norm(np.array([gx, gy]) / k - shift)) <= near

    def observe(self, bowl: str) -> Observation:
        """One fresh picture: is this bowl's opening where it was ("ok"), measurably somewhere near ("moved", with
        dx, dy in table mm), or not usable ("lost", with the reason)? Never raises and never fills in a guess."""
        with self._lock:
            try:
                obs = self._observe(bowl)
            except RimNotFound as e:
                obs = Observation("lost", bowl, reason=e.reason, stamp=self.clock())
            except Exception as e:                                  # a vision bug must not become a scoop
                obs = Observation("lost", bowl, reason=f"error:{type(e).__name__}", stamp=self.clock())
            self.last[bowl] = obs
            return obs

    def _observe(self, bowl: str) -> Observation:
        rc, now = self.rc, self.clock()
        lost = lambda reason, s=None: Observation("lost", bowl, reason=reason, stamp=now, sighting=s,
                                                  confidence=s.confidence if s else 0)
        ref = self.refs.get(bowl)
        if ref is None:
            return lost("no_reference")
        if now - ref.stamp > rc.reference_max_age_h * 3600:
            return lost("reference_stale")
        img, why = self._frame()
        if img is None:
            return lost(why)
        others = [r.sighting for n, r in self.refs.items() if n != bowl]
        s = find_rim(img, self._search_roi(ref), ref.sighting, rc.min_confidence, ignore=others, stamp=now,
                     reach_px=self._reach(ref))
        if abs(s.a / ref.sighting.a - 1) > SIZE_CHANGE:
            return lost("size_changed", s)
        lat, dep, depth_ok = displacement_mm(ref.sighting, s, ref.radius, rc.min_axis_ratio)
        calib = self.bowl_calib.get(bowl) or self.calib
        dx, dy = calib.to_table(lat, dep if depth_ok else 0.0) if calib else (0.0, 0.0)
        if math.hypot(lat, dep) <= rc.moved_tol_mm:
            return Observation("ok", bowl, dx, dy, s.confidence, now, sighting=s)
        if self._camera_moved(bowl, img, s):
            return lost("camera_moved", s)
        if not depth_ok and abs(dep) > rc.moved_tol_mm:
            return lost("moved_unmeasurable", s)                   # it moved along a direction this view can't measure
        if math.hypot(lat, dep if depth_ok else 0.0) > rc.max_follow_mm:
            return lost("moved_too_far", s)
        if calib is None:
            return lost("moved_unmeasurable", s)                   # no way to turn the camera's directions into x, y
        self._moved_thumb[bowl] = (now, _thumb(img))
        return Observation("moved", bowl, dx, dy, s.confidence, now, sighting=s)

    def accept_move(self, bowl: str, obs: Observation) -> None:
        """The scoop followed the cup to where `obs` saw it: that place is now the reference. The confirm time is
        kept (following does not make an old confirmation young), and a followed cup no longer teaches the view."""
        with self._lock:
            ref = self.refs.get(bowl)
            if ref is None or obs is None or obs.status != "moved" or obs.sighting is None or obs.bowl != bowl:
                return
            sx, sy = obs.sighting.cx - ref.sighting.cx, obs.sighting.cy - ref.sighting.cy
            x, y, w, h = ref.roi
            at, thumb = self._moved_thumb.pop(bowl, (None, None))
            self.refs[bowl] = replace(ref, sighting=obs.sighting, roi=(x + sx, y + sy, w, h),
                                      cup_xy=(ref.cup_xy[0] + obs.dx, ref.cup_xy[1] + obs.dy), followed=True,
                                      thumb=thumb if at == obs.stamp else None)
            self.last.pop(bowl, None)
            self._save()

    def draw(self, img, bowl: Optional[str] = None):
        """A copy of the picture with, per bowl, the reference opening (grey), the opening as last seen (green where
        it was, yellow moved, red not usable) and a line of text. For the operator to check that the ellipse really
        sits on the rim."""
        import cv2
        out = img.copy()
        colours = {"ok": (60, 200, 60), "moved": (0, 200, 240), "lost": (40, 40, 230)}
        with self._lock:
            items = [(n, r, self.last.get(n)) for n, r in self.refs.items() if bowl in (None, n)]
        for name, r, obs in items:
            col = colours.get(obs.status, (0, 0, 0)) if obs else (255, 255, 255)
            for s, c, thick in ((r.sighting, (170, 170, 170), 1), (obs.sighting if obs else None, col, 2)):
                if s is not None:
                    cv2.ellipse(out, ((s.cx, s.cy), (2 * s.a, 2 * s.b), s.angle_deg), c, thick, cv2.LINE_AA)
                    cv2.drawMarker(out, (int(s.cx), int(s.cy)), c, cv2.MARKER_CROSS, 8, 1)
            text = name if obs is None else (f"{name}: {obs.status} {obs.reason}".strip() + f" {obs.confidence:.2f}")
            at = (int(r.sighting.cx - r.sighting.a), int(max(14, r.sighting.cy - r.sighting.b - 8)))
            cv2.putText(out, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        return out
