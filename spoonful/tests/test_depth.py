"""Depth from one camera (iris scale -> per-person face width) and the depth part of adaptive delivery."""
import math
import random

import pytest
from conftest import wait_state

from spoonful.perception.face import FaceState, DepthEstimator, focal_px
from spoonful.perception.offsets import OffsetMap, OffsetTracker, fit

JOINTS = ["base", "shoulder", "elbow", "wrist_pitch"]
F = 914.0


def project(face_mm, z, yaw=0.0, iris_noise=0.0, rng=None):
    """Pinhole camera: (iris px, face width px) for a face at distance z."""
    iris = F * 11.7 / z + (rng.uniform(-iris_noise, iris_noise) if rng else 0.0)
    return iris, F * face_mm / z * math.cos(math.radians(yaw))


@pytest.fixture
def dcfg(cfg):
    cfg.offsets.focal_px = F
    return cfg


@pytest.mark.parametrize("face_mm", [130.0, 150.0])     # narrow and wide faces both get the right distance
def test_estimator_learns_face_width_from_iris(dcfg, face_mm):
    rng = random.Random(1)
    est = DepthEstimator(dcfg)
    t = 0.0
    for _ in range(120):                                  # 4 s at 30 fps with +-1 px iris jitter
        t += 1 / 30
        z, src = est.update(*project(face_mm, 500, iris_noise=1.0, rng=rng), 0.0, 1280, t)
    assert src == "face"
    assert est.face_width_mm == pytest.approx(face_mm, rel=0.03)
    # once learned, depth follows face width: smooth, and still right when they lean back
    iris, fw = project(face_mm, 600)
    z, _ = est.update(0.0, fw, 0.0, 1280, t + 0.03)      # even with the iris lost (blink)
    assert z == pytest.approx(600, rel=0.03)


def test_estimator_starts_on_iris_and_resets_for_a_new_person(dcfg):
    est = DepthEstimator(dcfg)
    z, src = est.update(*project(140, 450), 0.0, 1280, 0.0)
    assert src == "iris" and z == pytest.approx(450, rel=1e-6)
    for i in range(30):
        est.update(*project(140, 450), 0.0, 1280, i / 30)
    assert est.face_width_mm
    est.update(*project(120, 450), 0.0, 1280, 10.0)      # gone for > depth_person_reset_s: relearn
    assert len(est.samples) == 1


def test_estimator_ignores_turned_heads_for_learning_and_undoes_yaw(dcfg):
    est = DepthEstimator(dcfg)
    for i in range(40):
        est.update(*project(140, 450, yaw=25), 25.0, 1280, i / 30)
    assert not est.samples                                # only frames facing the camera teach face width
    for i in range(40):
        est.update(*project(140, 450), 0.0, 1280, 2 + i / 30)
    z, _ = est.update(*project(140, 450, yaw=25), 25.0, 1280, 4.0)
    assert z == pytest.approx(450, rel=0.01)              # cos(yaw) foreshortening removed


def test_estimator_scales_with_frame_width(dcfg):
    est = DepthEstimator(dcfg)
    iris, fw = project(140, 450)
    z, _ = est.update(iris / 2, fw / 2, 0.0, 640, 0.0)   # fallback resolution: half the pixels, same distance
    assert z == pytest.approx(450, rel=1e-6)


def test_focal_from_fov(cfg):
    cfg.offsets.focal_px = 0
    cfg.camera.hfov_deg = 90
    assert focal_px(cfg) == pytest.approx(640)


# ------------------------------------------------------------------ map with depth
def samples(depth=True, away_px_drift=0):
    nom = {"base": 0, "shoulder": 10, "elbow": 50, "wrist_pitch": 30}
    out = [{"label": "nominal", "px": [640, 400], "pose": nom}]
    for lab, dx, dy in [("left", -90, 0), ("right", 90, 0), ("up", 0, -90), ("down", 0, 90)]:
        p = dict(nom)
        p["base"] += -0.1 * dx
        p["shoulder"] += 0.05 * dy
        out.append({"label": lab, "px": [640 + dx, 400 + dy], "pose": p})
    if depth:   # 30 mm away from the camera: shoulder +3, elbow -1.5
        p = dict(nom, shoulder=13.0, elbow=48.5)
        out.append({"label": "away", "px": [640 + away_px_drift, 402], "pose": p})
    return out


def test_fit_depth_column_and_geometry():
    m = fit(samples(), JOINTS, hfov_focal_px=F)
    assert m.has_depth and not m.warnings
    assert m.depth_joint_per_mm == pytest.approx([0, 0.1, -0.05, 0])
    assert m.nominal_depth_mm == pytest.approx(F / 3.0)     # Z = f / px_per_mm
    m2 = fit(samples(), JOINTS, hfov_focal_px=F, measured_depth_mm=400)
    assert m2.nominal_depth_mm == 400 and m2.focal_px == pytest.approx(3.0 * 400)


def test_fit_warns_when_away_sample_moved_sideways():
    m = fit(samples(away_px_drift=60), JOINTS, hfov_focal_px=F)
    assert m.warnings and "line of sight" in m.warnings[0]


def test_old_map_without_depth_loads(tmp_path):
    m = fit(samples(depth=False), JOINTS)
    d = {k: v for k, v in m.__dict__.items() if k not in ("focal_px", "nominal_depth_mm", "depth_joint_per_mm",
                                                           "warnings")}
    d["obsolete_field"] = 1
    import json
    (tmp_path / "m.json").write_text(json.dumps(d))
    old = OffsetMap.load(tmp_path / "m.json")
    assert not old.has_depth


def face(z_eyes, px=(640, 400), frame_w=1280):
    return FaceState(True, 1.0, 1.0, 0, 0, 0.2, px, 280.0, frame_w, 720, depth_mm=z_eyes, depth_src="face")


def test_tracker_depth_correction(dcfg):
    m = fit(samples(), JOINTS, hfov_focal_px=F)
    o = dcfg.offsets
    tr = OffsetTracker(dcfg, m)
    at_nominal = m.nominal_depth_mm + o.eyes_behind_lips_mm
    assert tr.target_mm(face(at_nominal))[2] == 0
    assert tr.target_mm(face(at_nominal + o.depth_deadband_mm - 1))[2] == 0          # deadband
    assert tr.target_mm(face(at_nominal + 30))[2] == pytest.approx(30 - o.depth_deadband_mm)
    assert tr.target_mm(face(at_nominal + 300))[2] == o.max_depth_correction_mm      # clamp
    assert tr.target_mm(face(at_nominal - 300))[2] == -o.max_depth_correction_mm
    tr.mm[:] = [0, 0, 20]
    d = tr.joint_delta()
    assert d["shoulder"] == pytest.approx(2.0) and d["elbow"] == pytest.approx(-1.0)
    o.depth = False                                                                   # live switch
    assert tr.target_mm(face(at_nominal + 30))[2] == 0 and tr.joint_delta()["shoulder"] == 0


def test_tracker_lateral_uses_depth(dcfg):
    """The same pixel offset is twice the millimetres when the face is twice as far."""
    m = fit(samples(), JOINTS, hfov_focal_px=F)
    dcfg.offsets.max_correction_mm = 1000
    dcfg.offsets.max_depth_correction_mm = 1000
    tr = OffsetTracker(dcfg, m)
    e = dcfg.offsets.eyes_behind_lips_mm
    a = tr.target_mm(face(300 + e, px=(660, 400)))[0]
    b = tr.target_mm(face(600 + e, px=(660, 400)))[0]
    assert b == pytest.approx(2 * a)
    assert a == pytest.approx(20 * 300 / F)
    # same physical position seen at the fallback resolution
    assert tr.target_mm(face(300 + e, px=(330, 200), frame_w=640))[0] == pytest.approx(a)


# ------------------------------------------------------------------ end to end on the sim
def test_sim_diner_leans_back_and_the_spoon_follows(sim_system):
    s = sim_system
    w = s.cfg.face.delivery_window_px
    nominal_px = [(w.x[0] + w.x[1]) / 2, (w.y[0] + w.y[1]) / 2]   # sim mouth sits at the window centre
    smp = samples()
    for x in smp:
        x["px"] = [x["px"][0] - 640 + nominal_px[0], x["px"][1] - 400 + nominal_px[1]]
    m = fit(smp, JOINTS, hfov_focal_px=focal_px(s.cfg))
    s.offsets.map = s.offset_map = m
    s._use_map_geometry(m)
    s.cfg.offsets.rate_limit_mm_s = 1000
    wait_state(s, "IDLE")
    s.fsm.post("session_start")
    wait_state(s, "LISTENING")
    s.face.sim.update(present=True, mouth_open=False, dz=25.0)    # 25 mm further back than nominal
    s.fsm.post("bite", source="test")
    wait_state(s, "PRE_DELIVERY_HOLD")
    s.face.sim["mouth_open"] = True
    wait_state(s, "MOUTH_HOLD")
    assert s.face.latest().depth_src in ("iris", "face")
    assert s.fsm.status()["depth_corr_mm"] == pytest.approx(25 - s.cfg.offsets.depth_deadband_mm, abs=0.5)
    s.face.sim["mouth_open"] = False
    wait_state(s, "LISTENING")
    s.log.flush()
    row = s.log.query("SELECT depth_mm FROM bites ORDER BY id DESC LIMIT 1")[0]
    assert row["depth_mm"] == pytest.approx(17, abs=0.5)
