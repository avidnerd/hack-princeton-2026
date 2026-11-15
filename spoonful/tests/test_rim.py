"""perception/rim.py against synthetic camera pictures.

The pictures are drawn here with a pinhole camera looking at a table (z = 0, arm frame, mm): cups are cylinders with
a lip, a darker inside wall and food-coloured contents, on a plain or a cluttered table, blurred and noisy. This
checks the geometry and the refusal logic. It says nothing about a real translucent cup under real light.
"""
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from spoonful.perception import rim as R

W, H, F = 960, 540, 800.0               # a phone frame as the app receives it; F px is about a 62 deg wide lens
RAD, RIM = 35.0, 45.0                   # the cup: inner radius of the opening, rim height


class Cam:
    """Pinhole camera at `pos` looking at `look` (arm-frame mm), no roll."""

    def __init__(self, pos, look, f=F):
        self.pos, self.f = np.array(pos, float), f
        fwd = np.array(look, float) - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        self.rot = np.array([right, np.cross(fwd, right), fwd])      # rows: right, down, forward

    def project(self, pts):
        p = (np.atleast_2d(np.asarray(pts, float)) - self.pos) @ self.rot.T
        return np.column_stack([W / 2 + self.f * p[:, 0] / p[:, 2], H / 2 + self.f * p[:, 1] / p[:, 2]])

    def table_xy(self):
        """Per pixel: the table point it looks at (x, y) and whether it looks at the table at all."""
        u, v = np.meshgrid(np.arange(W) - W / 2, np.arange(H) - H / 2)
        ray = (u[..., None] * self.rot[0] + v[..., None] * self.rot[1]) / self.f + self.rot[2]
        hit = ray[..., 2] < -1e-3
        t = np.where(hit, -self.pos[2] / np.where(hit, ray[..., 2], -1.0), 0.0)
        return self.pos[0] + t * ray[..., 0], self.pos[1] + t * ray[..., 1], hit & (t < 3000)

    @property
    def azimuth_deg(self):
        return math.degrees(math.atan2(self.rot[2][1], self.rot[2][0]))


def _poly(cam, pts):
    return np.round(cam.project(pts) * 8).astype(np.int32)           # 1/8 px, for anti-aliased drawing


def _ring(x, y, r, z, n=160):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([x + r * np.cos(t), y + r * np.sin(t), np.full(n, z)])


_TABLES: dict = {}


def draw_table(cam, kind="plain"):
    key = (cam.pos.tobytes(), cam.rot.tobytes(), cam.f, kind)
    if key not in _TABLES:
        _TABLES[key] = _draw_table(cam, kind)
    return _TABLES[key].copy()


def _draw_table(cam, kind):
    x, y, hit = cam.table_xy()
    img = np.full((H, W, 3), (150, 140, 130), np.float32)             # the wall behind the table
    if kind == "plain":
        tab = np.stack([188 + 0.02 * x, 196 + 0.02 * x, 202 + 0.02 * x], -1)
    else:                                                              # wood grain, a checked mat, crumbs, cutlery
        grain = 18 * np.sin(y / 6.0 + 2.5 * np.sin(x / 45.0)) + 8 * np.sin(y / 1.7 + x / 30.0)
        tab = np.stack([95 + grain, 135 + grain, 175 + grain], -1)
        mat = (x > 120) & (x < 330) & (y > -170) & (y < 170)
        check = ((np.floor(x / 22) + np.floor(y / 22)) % 2 == 0)
        tab[mat & check] = (205, 200, 190)
        tab[mat & ~check] = (150, 110, 90)
    img[hit] = tab[hit]
    img = np.clip(img, 0, 255).astype(np.uint8)
    if kind != "plain":
        rng = np.random.default_rng(5)
        for cx, cy in rng.uniform([60, -260], [400, 260], (60, 2)):
            cv2.fillPoly(img, [_poly(cam, _ring(cx, cy, rng.uniform(2, 5), 0, 12))],
                         tuple(int(c) for c in rng.integers(40, 240, 3)), cv2.LINE_AA, 3)
        fork = np.array([[150, -150, 0], [300, -120, 0], [301, -128, 0], [151, -158, 0]], float)
        cv2.fillPoly(img, [_poly(cam, fork)], (215, 215, 220), cv2.LINE_AA, 3)
        napkin = np.array([[330, 60, 0], [400, 80, 0], [380, 150, 0], [310, 130, 0]], float)
        cv2.fillPoly(img, [_poly(cam, napkin)], (240, 235, 235), cv2.LINE_AA, 3)
    return img


def draw_cup(img, cam, x, y, radius=RAD, rim=RIM, wall=2.0, fill=0.55, food=(40, 140, 235), body=(232, 232, 238)):
    outer = np.vstack([_ring(x, y, radius + wall, 0), _ring(x, y, radius + wall, rim)])
    hull = cv2.convexHull(_poly(cam, outer))
    cv2.fillPoly(img, [hull], body, cv2.LINE_AA, 3)
    shade = np.zeros(img.shape[:2], np.uint8)                          # the side away from the light is darker
    cv2.fillPoly(shade, [hull], 255, cv2.LINE_AA, 3)
    c = cam.project([[x, y, rim / 2]])[0]
    ramp = np.clip((np.arange(W) - c[0]) / (2.5 * radius), -0.5, 0.5)[None, :, None] * 40
    k = shade[..., None] / 255.0
    img[:] = np.clip(img - ramp * k, 0, 255).astype(np.uint8)
    cv2.fillPoly(img, [_poly(cam, _ring(x, y, radius + wall, rim))], (250, 250, 252), cv2.LINE_AA, 3)      # the lip
    opening = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(opening, [_poly(cam, _ring(x, y, radius, rim))], 255, cv2.LINE_AA, 3)
    inside = np.full_like(img, (150, 150, 162))                        # inner wall, in shadow
    surface = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(surface, [_poly(cam, _ring(x, y, radius, rim * fill))], 255, cv2.LINE_AA, 3)
    inside[surface > 127] = food
    k = (opening[..., None] / 255.0)
    img[:] = (img * (1 - k) + inside * k).astype(np.uint8)


def finish(img, seed=0, blur=1.1, noise=4.0):
    out = cv2.GaussianBlur(img, (0, 0), blur).astype(np.float32)
    out += np.random.default_rng(seed).normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def scene(cam, cups, kind="plain", seed=0, **cup_kw):
    """cups: [(x, y)] or [(x, y, radius)] in arm-frame mm, drawn far to near."""
    img = draw_table(cam, kind)
    for c in sorted(cups, key=lambda c: -np.linalg.norm(np.array([c[0], c[1], 0]) - cam.pos)):
        draw_cup(img, cam, c[0], c[1], *(c[2:3] or (RAD,)), **cup_kw)
    return finish(img, seed)


def rim_px(cam, x, y, rim=RIM):
    return cam.project([[x, y, rim]])[0]


def box(cam, x, y, radius=RAD, rim=RIM, wall=2.0):
    """A tight box around the whole cup, as the setup step hands over (x, y, w, h)."""
    p = cam.project(np.vstack([_ring(x, y, radius + wall, 0), _ring(x, y, radius + wall, rim)]))
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return (x0, y0, x1 - x0, y1 - y0)


def bar(img, cam, x, y, frac, at=0.1, radius=RAD):
    """Something (the arm) standing across `frac` of the opening's width."""
    p = cam.project(_ring(x, y, radius, RIM))
    x0, x1, y0, y1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
    mid, half = (x0 + x1) / 2 + at * (x1 - x0), frac * (x1 - x0) / 2
    out = img.copy()
    cv2.rectangle(out, (int(mid - half), int(y0 - 60)), (int(mid + half), int(y1 + 25)), (120, 105, 60), -1)
    return finish(out, seed=3, blur=0.8, noise=3.0)


CAM = Cam((150, -450, 330), (210, 0, 40))          # the phone on a stand at the side: about 33 deg above the table
LOW = Cam((150, -520, 140), (210, 0, 45))          # the phone lying low: about 10 deg above the rim
A, B = (165.0, -30.0), (270.0, 45.0)               # two cups, arm frame


def rimcfg(**kw):
    d = dict(view_max_age_s=2.0, min_confidence=0.55, moved_tol_mm=6.0, max_follow_mm=45.0, min_axis_ratio=0.25,
             reference_max_age_h=4.0)
    d.update(kw)
    return SimpleNamespace(**d)


class Feed:
    """A camera and a clock the test controls."""

    def __init__(self, img=None):
        self.img, self.age, self.now = img, 0.1, 1000.0

    def observer(self, **kw):
        return R.RimObserver(rimcfg(**kw), lambda: self.img, lambda: self.age, lambda: self.now)


def two_cups(cam=CAM, kind="plain", **kw):
    """An observer with both cups confirmed in one picture."""
    feed = Feed(scene(cam, [A, B], kind))
    obs = feed.observer(**kw)
    for name, c in (("a", A), ("b", B)):
        obs.set_reference(name, feed.img, box(cam, *c), c, RAD)
    return feed, obs


# ---------------------------------------------------------------- find_rim
@pytest.mark.parametrize("kind", ["plain", "cluttered"])
@pytest.mark.parametrize("xy", [(210.0, 0.0), (160.0, -90.0), (270.0, 110.0)])
def test_centred_cup_is_found_on_its_rim(kind, xy):
    img = scene(CAM, [xy], kind)
    s = R.find_rim(img, box(CAM, *xy))
    want = rim_px(CAM, *xy)
    assert math.hypot(s.cx - want[0], s.cy - want[1]) < 3.0          # the rim centre, not the middle of the cup
    assert abs(s.cy - want[1]) < abs(s.cy - (box(CAM, *xy)[1] + box(CAM, *xy)[3] / 2))
    scale = F / np.linalg.norm(np.array([*xy, RIM]) - CAM.pos)
    assert s.a == pytest.approx((RAD + 1) * scale, rel=0.08)          # the ring found is the lip, not the base
    assert 0.3 < s.b / s.a < 0.8 and abs(s.angle_deg) < 12
    assert s.confidence > 0.8 and s.coverage > 0.85


@pytest.mark.parametrize("spoil", ["jpeg", "dark", "blurred", "grainy", "pale cup"])
def test_poor_pictures_are_found_or_refused_but_never_misplaced(spoil):
    xy = (170.0, -60.0)
    for kind in ("plain", "cluttered"):
        img = scene(CAM, [xy], kind)
        if spoil == "jpeg":                                           # the phone sends JPEG frames
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], cv2.IMREAD_COLOR)
        elif spoil == "dark":
            img = (img * 0.35).astype(np.uint8)
        elif spoil == "blurred":
            img = cv2.GaussianBlur(img, (0, 0), 2.5)
        elif spoil == "grainy":
            img = finish(img, seed=9, blur=0.1, noise=12.0)
        else:                                                         # a cup nearly the colour of the plain table
            img = scene(CAM, [xy], kind, body=(192, 198, 202), food=(168, 188, 203))
        try:
            s = R.find_rim(img, box(CAM, *xy))
        except R.RimNotFound as e:
            assert e.reason in R.REASONS
            continue
        assert math.dist((s.cx, s.cy), rim_px(CAM, *xy)) < 5.0


def test_no_cup_no_answer():
    for kind in ("plain", "cluttered"):
        with pytest.raises(R.RimNotFound) as e:
            R.find_rim(scene(CAM, [], kind), (400, 200, 160, 140))
        assert e.value.reason in ("no_ellipse", "low_contrast", "partly_hidden")
    with pytest.raises(R.RimNotFound) as e:
        R.find_rim(None)
    assert e.value.reason == "no_frame"
    with pytest.raises(R.RimNotFound) as e:
        R.find_rim(np.full((H, W, 3), 128, np.uint8), (400, 200, 160, 140))
    assert e.value.reason == "low_contrast"


@pytest.mark.parametrize("kind", ["plain", "cluttered"])
def test_partly_hidden_opening(kind):
    """Something (the arm) stands across the opening of a cup that is known and has not moved."""
    xy = (210.0, 0.0)
    clear = scene(CAM, [xy], kind)
    full = R.find_rim(clear, box(CAM, *xy))
    want = rim_px(CAM, *xy)
    for at in (-0.3, 0.1):
        s = R.find_rim(bar(clear, CAM, *xy, frac=0.3, at=at), box(CAM, *xy), full, reach_px=80)
        assert math.hypot(s.cx - want[0], s.cy - want[1]) < 4.0       # 30% hidden: still the same ellipse
        assert s.confidence < full.confidence - 0.1 and 0.55 <= s.coverage < 0.85
        with pytest.raises(R.RimNotFound) as e:                       # 60% hidden: refused, not guessed
            R.find_rim(bar(clear, CAM, *xy, frac=0.6, at=at), box(CAM, *xy), full, reach_px=80)
        assert e.value.reason in ("partly_hidden", "no_ellipse")


@pytest.mark.parametrize("kind", ["plain", "cluttered"])
@pytest.mark.parametrize("frac", [0.3, 0.6])
def test_partly_hidden_opening_is_refused_without_a_known_place(kind, frac):
    """A first sighting, or a cup that may have moved: a half-seen ellipse could be the cup's body or base."""
    xy = (210.0, 0.0)
    clear = scene(CAM, [xy], kind)
    full = R.find_rim(clear, box(CAM, *xy))
    hidden = bar(clear, CAM, *xy, frac=frac)
    elsewhere = R.RimSighting(full.cx + 40, full.cy - 6, full.a, full.b, full.angle_deg, 1, 0, 1)
    for kw in ({}, {"expect": full}, {"expect": elsewhere, "reach_px": 80}):
        with pytest.raises(R.RimNotFound) as e:
            R.find_rim(hidden, box(CAM, *xy), **kw)
        assert e.value.reason in ("partly_hidden", "no_ellipse")


def test_two_similar_cups_in_the_roi_is_ambiguous():
    cups = [(210.0, -45.0), (215.0, 50.0)]
    img = scene(CAM, cups)
    b0, b1 = box(CAM, *cups[0]), box(CAM, *cups[1])
    x0, y0 = min(b0[0], b1[0]), min(b0[1], b1[1])
    both = (x0, y0, max(b0[0] + b0[2], b1[0] + b1[2]) - x0, max(b0[1] + b0[3], b1[1] + b1[3]) - y0)
    with pytest.raises(R.RimNotFound) as e:
        R.find_rim(img, both)
    assert e.value.reason == "ambiguous"
    one = R.find_rim(img, b0)
    with pytest.raises(R.RimNotFound) as e:                           # knowing what one cup looks like doesn't help
        R.find_rim(img, both, expect=one)
    assert e.value.reason == "ambiguous"
    other = R.find_rim(img, both, expect=one, ignore=[one])           # unless that one is known to be another bowl
    assert math.dist((other.cx, other.cy), rim_px(CAM, *cups[1])) < 3.0


def test_camera_at_table_height_is_too_flat():
    cam = Cam((150, -520, 85), (210, 0, 45))                          # 4 deg above the rim
    with pytest.raises(R.RimNotFound) as e:
        R.find_rim(scene(cam, [(210.0, 0.0)]), box(cam, 210.0, 0.0))
    assert e.value.reason in ("too_flat", "no_ellipse", "partly_hidden")


# ---------------------------------------------------------------- pixels -> mm
def test_displacement_uses_the_ellipse_as_the_ruler():
    ref = R.RimSighting(400, 300, 70, 35, 0, 1, 0, 1)
    lat, dep, ok = R.displacement_mm(ref, R.RimSighting(440, 290, 70, 35, 0, 1, 0, 1), RAD, 0.25)
    assert (lat, dep, ok) == (pytest.approx(20.0), pytest.approx(10.0), True)   # right; up the picture = further
    flat = R.RimSighting(400, 300, 70, 10, 0, 1, 0, 1)
    assert R.displacement_mm(flat, flat, RAD, 0.25)[2] is False
    tilted = R.RimSighting(400, 300, 70, 35, 90, 1, 0, 1)             # camera on its side: major axis is vertical
    lat, dep, _ = R.displacement_mm(tilted, R.RimSighting(400, 340, 70, 35, 90, 1, 0, 1), RAD, 0.25)
    assert abs(lat) == pytest.approx(20.0) and dep == pytest.approx(0.0, abs=1e-6)


def test_view_calib_directions():
    c = R.ViewCalib(0.0, False, "t", 0, 0)                            # looking along +x: right of the picture is -y
    assert c.to_table(0, 10) == pytest.approx((10, 0))
    assert c.to_table(10, 0) == pytest.approx((0, -10))
    assert R.ViewCalib(90.0, False, "t", 0, 0).to_table(10, 0) == pytest.approx((10, 0))
    assert R.ViewCalib(0.0, True, "t", 0, 0).to_table(10, 0) == pytest.approx((0, 10))


def test_calib_from_pair_finds_the_view_direction():
    img = scene(CAM, [A, B])
    sa, sb = R.find_rim(img, box(CAM, *A)), R.find_rim(img, box(CAM, *B))
    calib = R.calib_from_pair(sa, A, RAD, sb, B, RAD)
    assert calib is not None and calib.source == "pair" and not calib.mirrored
    assert abs((calib.azimuth_deg - CAM.azimuth_deg + 180) % 360 - 180) < 10
    assert calib.residual_mm < 0.15 * math.dist(A, B)


def test_calib_from_pair_refuses_when_the_separations_disagree():
    img = scene(CAM, [A, B])
    sa, sb = R.find_rim(img, box(CAM, *A)), R.find_rim(img, box(CAM, *B))
    far_b = (B[0] + 90.0, B[1] + 60.0)                                # the arm thinks cup b is somewhere else
    assert R.calib_from_pair(sa, A, RAD, sb, far_b, RAD) is None
    assert R.calib_from_pair(sa, A, RAD * 1.5, sb, B, RAD * 1.5) is None    # or the cups are not the size it was told
    assert R.calib_from_pair(sa, A, RAD, sb, (A[0] + 10, A[1]), RAD) is None      # no baseline to speak of
    low = scene(LOW, [A, B])
    la, lb = R.find_rim(low, box(LOW, *A)), R.find_rim(low, box(LOW, *B))
    assert R.calib_from_pair(la, A, RAD, lb, B, RAD) is None          # too flat a view to measure depth


def test_calib_from_jacobian():
    def jac(cam, yaw_deg, mirror=False):
        """What AimCheck measures: px per mm of motion further / left / up, at a point above the table."""
        yaw, p = math.radians(yaw_deg), np.array([210.0, 0.0, 60.0])
        rows = []
        for d in ([math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]):
            j = (cam.project([p + np.array(d)])[0] - cam.project([p])[0])
            rows.append([-j[0], j[1]] if mirror else j)
        return np.array(rows)
    for yaw in (0.0, 25.0, -40.0):
        c = R.calib_from_jacobian(jac(CAM, yaw), yaw_deg=yaw)
        assert c is not None and not c.mirrored and c.source == "jacobian"
        assert abs((c.azimuth_deg - CAM.azimuth_deg + 180) % 360 - 180) < 6
    m = R.calib_from_jacobian(jac(CAM, 0.0, mirror=True))
    assert m is not None and m.mirrored and abs((m.azimuth_deg - CAM.azimuth_deg + 180) % 360 - 180) < 6
    assert R.calib_from_jacobian(jac(Cam((150, -520, 60), (210, 0, 45)), 0.0)) is None    # depth barely visible
    assert R.calib_from_jacobian(np.zeros((3, 2))) is None
    assert R.calib_from_jacobian([[1.0, 0.0], [0.6, 0.8], [0, 0]]) is None                # not a view of a table


# ---------------------------------------------------------------- the observer
def test_unmoved_cup_is_ok_and_no_reference_is_lost():
    feed, obs = two_cups()
    assert obs.has_reference("a") and not obs.has_reference("c")
    assert obs.observe("c").reason == "no_reference"
    feed.img = scene(CAM, [A, B], seed=7)                             # a new frame of the same table
    o = obs.observe("a")
    assert (o.status, o.bowl) == ("ok", "a") and o.confidence > 0.8 and o.sighting is not None
    assert math.hypot(o.dx, o.dy) < 3.0
    obs.forget("a")
    assert obs.observe("a").reason == "no_reference" and obs.observe("b").status == "ok"
    assert obs.calib is not None                                      # the camera did not move: the view stays known
    obs.forget()
    assert not obs.has_reference("b") and obs.calib is None


def test_camera_dropout_and_stale_frame_are_lost():
    feed, obs = two_cups()
    feed.img = None
    o = obs.observe("a")
    assert (o.status, o.reason) == ("lost", "no_frame")
    feed.img, feed.age = scene(CAM, [A, B]), 5.0
    assert obs.observe("a").reason == "stale_frame"
    feed.age = 0.2
    assert obs.observe("a").status == "ok"


def test_frame_that_stops_changing_is_stale_without_an_age_function():
    feed = Feed(scene(CAM, [A]))
    obs = R.RimObserver(rimcfg(), lambda: feed.img, None, lambda: feed.now)
    obs.set_reference("a", feed.img, box(CAM, *A), A, RAD)
    assert obs.observe("a").status == "ok"
    feed.now += 1.0
    assert obs.observe("a").status == "ok"
    feed.now += 3.0                                                   # the very same picture, 4 s on
    assert obs.observe("a").reason == "stale_frame"
    feed.img = feed.img.copy()                                        # a new frame arrives
    assert obs.observe("a").status == "ok"


def test_old_reference_is_stale():
    feed, obs = two_cups()
    feed.now += 4.1 * 3600
    o = obs.observe("a")
    assert (o.status, o.reason) == ("lost", "reference_stale")


def test_set_reference_refuses_what_it_cannot_see():
    feed = Feed(scene(CAM, [A]))
    obs = feed.observer()
    obs.set_reference("a", feed.img, box(CAM, *A), A, RAD)
    with pytest.raises(R.RimNotFound):
        obs.set_reference("a", bar(feed.img, CAM, *A, frac=0.7), box(CAM, *A), A, RAD)
    assert not obs.has_reference("a")                                 # the old reference is not kept as if confirmed


# Stated tolerance of a followed move in these pictures: 6 mm for moves up to 35 mm, cup about 0.5 m from the camera.
# The error grows with the move (about a tenth to a sixth of it: one azimuth serves both cups, see rim.py), so this
# is not a claim about longer moves, other camera placements, or a real table.
MOVE_TOL_MM = 6.0


@pytest.mark.parametrize("kind", ["plain", "cluttered"])
@pytest.mark.parametrize("move", [(0, 30), (0, -30), (30, 0), (-30, 0), (20, 20), (-25, 15), (10, -30)])
def test_moved_cup_is_measured_in_table_mm(kind, move):
    """Cup a is moved by a known amount: right/left of the picture (arm x here) and nearer/further (arm y)."""
    feed, obs = two_cups(kind=kind)
    assert obs.calib is not None and obs.calib.source == "pair"
    feed.img = scene(CAM, [(A[0] + move[0], A[1] + move[1]), B], kind, seed=11)
    o = obs.observe("a")
    assert o.status == "moved", o.reason
    assert math.hypot(o.dx - move[0], o.dy - move[1]) < MOVE_TOL_MM, (o.dx, o.dy)
    assert obs.observe("b").status == "ok"                            # the other cup did not move


def test_moved_cup_without_calibration_is_not_guessed():
    feed = Feed(scene(CAM, [A]))
    obs = feed.observer()
    obs.set_reference("a", feed.img, box(CAM, *A), A, RAD)
    assert obs.calib is None                                          # one cup teaches nothing about the view
    feed.img = scene(CAM, [(A[0] + 25, A[1] + 10)], seed=2)
    o = obs.observe("a")
    assert (o.status, o.reason) == ("lost", "moved_unmeasurable") and (o.dx, o.dy) == (0, 0)
    assert o.sighting is not None                                     # it was seen; it just can't be turned into mm
    obs.set_calib(R.ViewCalib(CAM.azimuth_deg, False, "test", 0, 0))
    o = obs.observe("a")
    assert o.status == "moved" and math.hypot(o.dx - 25, o.dy - 10) < MOVE_TOL_MM


def test_pair_calibration_needs_both_cups_where_they_were_confirmed():
    feed = Feed(scene(CAM, [A, B]))
    obs = feed.observer()
    obs.set_reference("a", feed.img, box(CAM, *A), A, RAD)
    moved_a = (A[0] + 30, A[1] - 20)                                  # cup a is pushed before cup b is confirmed
    feed.img = scene(CAM, [moved_a, B])
    obs.set_reference("b", feed.img, box(CAM, *B), B, RAD)
    assert obs.calib is None


def test_recalibrate_and_per_bowl_calibration():
    feed, obs = two_cups()
    learned = obs.calib
    assert obs.recalibrate() is not None and abs(obs.calib.azimuth_deg - learned.azimuth_deg) < 1e-6
    wrong = {"a": (*A, RAD), "b": (B[0] + 90, B[1] + 60, RAD)}        # the arm's idea of the cups doesn't fit
    assert obs.recalibrate(wrong) is None and obs.calib is None
    handed = R.ViewCalib(CAM.azimuth_deg, False, "jacobian", 0, 0)
    obs.set_calib(handed)
    assert obs.recalibrate(wrong) is None and obs.calib is handed     # a handed-over one is not thrown away by that
    obs.set_calib(R.ViewCalib(CAM.azimuth_deg + 90, False, "jacobian", 0, 0), bowl="a")   # (wrong on purpose)
    feed.img = scene(CAM, [A, (B[0] + 25, B[1])])
    ob = obs.observe("b")
    feed.img = scene(CAM, [(A[0] + 25, A[1]), B])
    oa = obs.observe("a")
    assert ob.status == "moved" and math.hypot(ob.dx - 25, ob.dy) < MOVE_TOL_MM
    assert oa.status == "moved" and math.hypot(oa.dx, oa.dy - 25) < MOVE_TOL_MM    # bowl a used its own calibration


def test_neighbour_moving_does_not_disturb_a_cup_that_stayed():
    feed, obs = two_cups()
    feed.img = scene(CAM, [A, (B[0] - 30, B[1] - 15)])                # b is pushed toward a
    assert obs.observe("a").status == "ok"
    assert obs.observe("b").status == "moved"


def test_moved_too_far_is_lost():
    feed, obs = two_cups()
    feed.img = scene(CAM, [(A[0] - 55, A[1] - 10), B])
    o = obs.observe("a")
    assert (o.status, o.reason) == ("lost", "moved_too_far")
    feed.img = scene(CAM, [B])                                        # taken off the table altogether
    o = obs.observe("a")
    assert o.status == "lost" and o.reason in ("no_ellipse", "partly_hidden", "low_contrast")


def test_other_size_of_cup_in_its_place_is_lost():
    feed, obs = two_cups()
    feed.img = scene(CAM, [(A[0], A[1], RAD * 1.4), B])
    o = obs.observe("a")
    assert o.status == "lost" and o.reason in ("size_changed", "no_ellipse", "partly_hidden")


def test_all_references_shifting_together_is_a_moved_camera():
    feed, obs = two_cups()
    bumped = Cam(CAM.pos + [25, 0, 0], (245, 5, 40))                  # the phone was nudged
    feed.img = scene(bumped, [A, B])
    for name in ("a", "b"):
        o = obs.observe(name)
        assert (o.status, o.reason) == ("lost", "camera_moved")


def test_moved_camera_is_noticed_with_one_cup_on_a_textured_table():
    feed = Feed(scene(CAM, [A], "cluttered"))
    obs = feed.observer()
    obs.set_reference("a", feed.img, box(CAM, *A), A, RAD)
    obs.set_calib(R.ViewCalib(CAM.azimuth_deg, False, "test", 0, 0))
    feed.img = scene(Cam(CAM.pos, (238, 6, 40)), [A], "cluttered")    # the phone turned a little
    o = obs.observe("a")
    assert (o.status, o.reason) == ("lost", "camera_moved")
    feed.img = scene(CAM, [(A[0] + 25, A[1])], "cluttered")           # the cup moved, the table did not
    assert obs.observe("a").status == "moved"


def test_accept_move_makes_the_new_place_the_reference():
    feed, obs = two_cups()
    there = (A[0] + 20, A[1] + 25)
    feed.img = scene(CAM, [there, B])
    o = obs.observe("a")
    assert o.status == "moved"
    obs.accept_move("a", o)
    again = obs.observe("a")
    assert again.status == "ok" and math.hypot(again.dx, again.dy) < 3.0
    feed.img = scene(CAM, [A, B])                                     # put back where it first was: that is a move now
    back = obs.observe("a")
    assert back.status == "moved" and math.hypot(back.dx + 20, back.dy + 25) < MOVE_TOL_MM
    feed.now += 4.1 * 3600                                            # following did not renew the confirmation
    assert obs.observe("a").reason == "reference_stale"


def test_accept_move_ignores_anything_but_a_measured_move():
    feed, obs = two_cups()
    before = obs.refs["a"].sighting
    obs.accept_move("a", obs.observe("a"))                            # "ok"
    obs.accept_move("a", R.Observation("lost", "a", reason="no_frame"))
    obs.accept_move("a", None)
    assert obs.refs["a"].sighting is before


def test_low_camera_measures_sideways_but_refuses_depth():
    """About 10 deg above the rim: the opening is found but too flat for depth."""
    img = scene(LOW, [A])
    ref = R.find_rim(img, box(LOW, *A))
    assert ref.b / ref.a < 0.25
    feed = Feed(img)
    obs = feed.observer()
    obs.set_reference("a", img, box(LOW, *A), A, RAD)
    obs.set_calib(R.ViewCalib(LOW.azimuth_deg, False, "test", 0, 0))
    side = scene(LOW, [(A[0] + 30, A[1])], seed=4)                    # across this camera's view
    lat, dep, ok = R.displacement_mm(ref, R.find_rim(side, None, ref), RAD, 0.25)
    assert ok is False and abs(lat) == pytest.approx(30, abs=5)
    feed.img = side
    o = obs.observe("a")
    assert o.status == "moved" and abs(o.dx - 30) < MOVE_TOL_MM and abs(o.dy) < MOVE_TOL_MM
    feed.img = scene(LOW, [(A[0], A[1] + 35)], seed=4)                # straight away from the camera
    o = obs.observe("a")
    assert (o.status, o.reason) == ("lost", "moved_unmeasurable")


def test_observe_never_raises():
    feed, obs = two_cups()
    feed.img = "not a picture"
    o = obs.observe("a")
    assert o.status == "lost" and o.reason


def test_draw_overlays_without_touching_the_frame():
    feed, obs = two_cups()
    obs.observe("a")
    keep = feed.img.copy()
    out = obs.draw(feed.img)
    assert out.shape == feed.img.shape and (feed.img == keep).all() and (out != keep).any()
    assert obs.draw(feed.img, "b").shape == feed.img.shape


def test_accepts_the_app_config(cfg):
    obs = R.RimObserver(cfg, lambda: None)
    assert obs.rc is cfg.plate.rim
    assert obs.observe("bowl1").reason == "no_reference"
