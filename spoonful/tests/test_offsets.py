import numpy as np
import pytest

from spoonful.perception.face import FaceState
from spoonful.perception.offsets import OffsetMap, OffsetTracker, fit

JOINTS = ["base", "shoulder", "elbow", "wrist_pitch"]


def samples():
    # ground truth: base moves -0.1 deg/px in x, shoulder +0.05 deg/px in y
    nom = {"base": 0, "shoulder": 10, "elbow": 50, "wrist_pitch": 30}
    out = [{"label": "nominal", "px": [640, 400], "pose": nom}]
    for lab, dx, dy in [("left", -90, 0), ("right", 90, 0), ("up", 0, -90), ("down", 0, 90)]:
        p = dict(nom)
        p["base"] += -0.1 * dx
        p["shoulder"] += 0.05 * dy
        out.append({"label": lab, "px": [640 + dx, 400 + dy], "pose": p})
    return out


def test_fit_recovers_linear_map(tmp_path):
    m = fit(samples(), JOINTS)
    assert m.px_per_mm == pytest.approx(3.0)
    d = m.joint_delta((10, 0))
    assert d["base"] == pytest.approx(-1.0, abs=1e-6)
    assert d["shoulder"] == pytest.approx(0.0, abs=1e-6)
    m.save(tmp_path / "m.json")
    assert OffsetMap.load(tmp_path / "m.json").joint_delta((0, 20))["shoulder"] == pytest.approx(1.0)


def test_fit_rejects_collinear():
    s = [x for x in samples() if x["label"] in ("nominal", "left", "right")]
    with pytest.raises(ValueError):
        fit(s, JOINTS)


def face_at(px, fw=220.0):
    return FaceState(True, 1.0, 1.0, 0, 0, 0.2, px, fw, 1280, 720)


def test_tracker_clamps_and_rate_limits(cfg):
    m = fit(samples(), JOINTS)
    tr = OffsetTracker(cfg, m)
    far = face_at((640 + 3 * 200, 400))          # 200 mm away -> clamp to max_correction_mm
    assert np.linalg.norm(tr.target_mm(far)) == pytest.approx(cfg.offsets.max_correction_mm)
    tr.update(far, 0.0)
    tr.update(far, 0.5)                           # 0.5 s at rate_limit_mm_s
    assert tr.offset_mm == pytest.approx(cfg.offsets.rate_limit_mm_s * 0.5, rel=1e-3)
    d = tr.joint_delta()
    assert d["base"] < 0                          # moved right in the image -> base turns the right way


def test_distance_scaling(cfg):
    m = fit(samples(), JOINTS)
    tr = OffsetTracker(cfg, m)
    near = tr.target_mm(face_at((640 + 30, 400), fw=440.0))   # face twice as big -> half the correction
    far = tr.target_mm(face_at((640 + 30, 400), fw=220.0))
    assert near[0] == pytest.approx(far[0] / 2)


def test_no_map_means_no_correction(cfg):
    tr = OffsetTracker(cfg, None)
    assert tr.update(face_at((900, 400)), 1.0) == {}
