import json
from pathlib import Path

import pytest

from spoonful.perception.face import FaceState
from spoonful.perception.gate import REASONS, Gate

FIX = Path(__file__).parent / "fixtures"


def face(ts, present=True, ratio=0.15, yaw=0.0, pitch=0.0, mouth=(640, 400)):
    return FaceState(present, ts, 5.0, yaw, pitch, ratio, mouth if present else None, 220.0, 1280, 720)


def run(gate, seq, state="DELIVERING"):
    """seq: list of FaceState; evaluated at their own timestamps (+10 ms)."""
    return [gate.permit(f, state, f.ts + 0.01) for f in seq]


def steady(n=40, dt=1 / 30, t0=100.0, **kw):
    return [face(t0 + i * dt, **kw) for i in range(n)]


def test_permit_after_presence_and_open_hold(cfg):
    g = Gate(cfg.face)
    v = run(g, steady())
    assert not v[0].ok and v[0].reason == "face_too_brief"
    first_ok = next(i for i, x in enumerate(v) if x.ok)
    assert first_ok * (1 / 30) >= cfg.face.presence_hold_s - 1 / 30
    assert all(x.ok for x in v[first_ok:])


@pytest.mark.parametrize("kw,reason", [
    ({"present": False}, "no_face"),
    ({"yaw": 30.0}, "looking_away"),
    ({"pitch": -40.0}, "looking_away"),
    ({"ratio": 0.02}, "mouth_closed"),
    ({"mouth": (50, 50)}, "mouth_outside_window"),
])
def test_deny_reasons(cfg, kw, reason):
    g = Gate(cfg.face)
    v = run(g, steady(**kw))
    assert not v[-1].ok and v[-1].reason == reason


def test_stale_frame(cfg):
    g = Gate(cfg.face)
    run(g, steady())
    f = face(200.0)
    v = g.permit(f, "DELIVERING", 200.0 + cfg.face.frame_max_age_ms / 1000 + 0.05)
    assert v.reason == "frame_stale"


def test_paused_and_stopped_override_everything(cfg):
    g = Gate(cfg.face)
    run(g, steady())
    assert g.permit(face(102), "PAUSED", 102.01).reason == "paused"
    assert g.permit(face(102.1), "STOPPED", 102.11).reason == "stop_latched"


def test_hysteresis_holds_then_denies(cfg):
    th, hf = cfg.face.mouth_open_ratio_threshold, cfg.face.hysteresis_fraction
    g = Gate(cfg.face)
    v = run(g, steady())
    assert v[-1].ok
    t = 102.0
    assert g.permit(face(t, ratio=th * (hf + 0.05)), "DELIVERING", t + 0.01).ok      # sags: still OK
    assert not g.permit(face(t + 0.04, ratio=th * (hf - 0.1)), "DELIVERING", t + 0.05).ok  # below: denied
    # re-opening requires the hold again (not flutter back instantly)
    assert not g.permit(face(t + 0.08, ratio=th * 2), "DELIVERING", t + 0.09).ok


def test_turn_away_denies_on_next_frame(cfg):
    """Acceptance (Phase 2): closes within 200 ms of turning away. The gate itself denies on the first frame."""
    g = Gate(cfg.face)
    run(g, steady())
    v = g.permit(face(103.0, yaw=40), "DELIVERING", 103.01)
    assert v.reason == "looking_away"


def test_recorded_fixture_sequences(cfg):
    """Fixtures: JSON lists of FaceState dicts with the verdict reason we expect at the end."""
    for p in sorted(FIX.glob("gate_*.json")):
        d = json.loads(p.read_text())
        g = Gate(cfg.face)
        v = None
        for row in d["frames"]:
            row = dict(row)
            if row.get("mouth_center_px"):
                row["mouth_center_px"] = tuple(row["mouth_center_px"])
            f = FaceState(**row)
            v = g.permit(f, d.get("state", "DELIVERING"), f.ts + 0.01)
        assert v.reason == d["expect_reason"], p.name
        assert v.ok == (d["expect_reason"] == "")


def test_reason_strings_are_the_documented_set():
    assert set(REASONS) == {"stop_latched", "paused", "frame_stale", "no_face", "face_too_brief", "looking_away",
                            "mouth_outside_window", "mouth_closed"}
