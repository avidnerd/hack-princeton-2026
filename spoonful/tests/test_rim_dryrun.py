"""brain/rim_dryrun.py: the picture and the verdict of the planned scoop. No arm, no camera; files only under
tmp_path. The pictures are checked for size and for the colours that carry meaning, not pixel by pixel."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spoonful import config as C
from spoonful.arm import cup as U
from spoonful.arm.scoop_style import lower_tip
from spoonful.arm.trajectory import TrajFile, make_waypoints
from spoonful.brain import rim_dryrun as D

ROOT = Path(__file__).resolve().parent.parent


def has(img, bgr) -> bool:
    return bool((img.reshape(-1, 3) == np.array(bgr, np.uint8)).all(axis=1).any())


@pytest.fixture
def dcfg(fast_cfg, tmp_path, monkeypatch):
    """Sim kinematics, an empty trajectory folder (no old path, no rest pose), placeholder spoon, and a config that
    cannot be saved."""
    monkeypatch.setattr(C.Config, "save", lambda *a, **k: pytest.fail("the dry run must not save config"))
    fast_cfg.arm.trajectories_dir = str(tmp_path / "traj")
    fast_cfg.spoon = C.SpoonGeomCfg()
    fast_cfg.plate.bowls[0].label = "applesauce"
    return fast_cfg


def set_cup(cfg, **kw):
    d = dict(x_mm=200.0, y_mm=60.0, radius_mm=35.0, rim_mm=45.0, uncertainty_mm=3.0, confirmed=True,
             source="confirmed")
    d.update(kw)
    cfg.plate.bowls[0].cup = C.CupCfg(**d)
    return cfg


def direct(cfg, name="bowl1"):
    """plan_entry called the way the dry run documents it (no start pose: the folder has no rest)."""
    bowl = next(b for b in cfg.plate.bowls if b.name == name)
    rim = cfg.plate.rim
    return U.plan_entry(cfg.kinematics, cfg.arm.soft_limits_deg, cfg.arm.home_pose,
                        U.Cup.from_cfg(bowl.cup, rim.default_rim_mm), U.Spoon.from_cfg(cfg.spoon),
                        clearance_mm=rim.clearance_mm, hover_above_rim_mm=rim.hover_above_rim_mm,
                        want_angle_deg=float(bowl.attack.get("entry_angle_deg", 45.0)),
                        want_drag_mm=float(bowl.attack.get("drag_mm", 20.0)), headroom_mm=rim.sag_headroom_mm,
                        want_depth_mm=None)


def test_feasible_confirmed_cup(dcfg):
    set_cup(dcfg)
    v, p = D.verdict(dcfg, "bowl1"), direct(dcfg)
    assert v["feasible"] and v["reason"] == "" and not v["refusals"]
    assert v["entry_angle_deg"] == p.entry_angle_deg and v["drag_mm"] == p.drag_mm and v["dip_z_mm"] == p.dip_z
    assert v["min_clearance_mm"] == p.min_clearance_mm >= v["margin_mm"] == 6.0
    assert "safe entry found" in v["text"] and "applesauce" in v["text"]
    assert v["spoon_measured"] is False and v["cup"]["confirmed"] and v["cup"]["rim_measured"]
    img = D.render(dcfg, "bowl1")
    assert img.dtype == np.uint8 and img.shape[1] == D.W and img.shape[0] >= 700 and img.shape[2] == 3
    for name in ("hover", "dip (lowest)"):
        assert has(img, D.POSE_COL[name])                 # the spoon is drawn at those waypoints
    assert has(img, D.WALL) and has(img, D.BAND) and has(img, D.SAFE)
    assert not has(img, D.BROWN)                          # no old path in an empty folder
    # nothing the plan does is predicted to touch the cup, and the picture agrees: no red dot inside the views
    views = img[40:, :D.LEFT - 20]
    assert not has(views[D.TOP_H + 100:], D.RED)


def test_too_narrow_cup(dcfg):
    set_cup(dcfg, radius_mm=15.0)
    v = D.verdict(dcfg, "bowl1")
    with pytest.raises(U.NoSafeEntry) as e:
        direct(dcfg)
    assert e.value.reason == "too_narrow"
    assert not v["feasible"] and v["reason"] == "too_narrow" and v["detail"] == e.value.detail
    assert "NO SAFE ENTRY (too_narrow)" in v["text"] and v["rejected"] == [e.value.detail]
    assert v["entry_angle_deg"] is None and v["min_clearance_mm"] is None
    img = D.render(dcfg, "bowl1")
    assert img.shape[1] == D.W and img.shape[0] >= 700
    assert has(img[40:, :D.LEFT - 20], D.RED)             # where a straight entry would hit is shown in red
    assert not has(img, D.POSE_COL["hover"])              # and nothing is drawn as if it were a plan


def test_photo_only_cup_is_refused_twice(dcfg):
    set_cup(dcfg, uncertainty_mm=25.0, confirmed=False, source="photo")
    v = D.verdict(dcfg, "bowl1")
    assert not v["feasible"] and v["reason"] == "too_narrow" and v["margin_mm"] == 28.0
    assert any("not confirmed" in r for r in v["refusals"]) and "a real scoop would refuse" in v["text"]


def test_bowl_without_a_cup(dcfg):
    assert dcfg.plate.bowls[0].cup is None
    v = D.verdict(dcfg, "bowl1")
    assert not v["feasible"] and not v["has_cup"] and v["reason"] == "no_cup"
    assert v["text"].endswith("no cup geometry: map the table first")
    img = D.render(dcfg, "bowl1")
    assert img.ndim == 3 and img.shape[0] >= 150 and img.shape[1] >= 500 and has(img, D.RED)
    v = D.verdict(dcfg, "nope")
    assert v["reason"] == "no_bowl" and D.render(dcfg, "nope").shape[0] >= 150


@pytest.mark.parametrize("radius", [35.0, 45.0])
def test_unmeasured_rim(dcfg, radius):
    set_cup(dcfg, rim_mm=None, radius_mm=radius)
    v = D.verdict(dcfg, "bowl1")
    assert v["cup"]["rim_measured"] is False and v["cup"]["rim_mm"] == dcfg.plate.rim.default_rim_mm
    try:
        p = direct(dcfg)
        assert v["feasible"] and v["dip_z_mm"] == p.dip_z
    except U.NoSafeEntry as e:
        assert not v["feasible"] and v["reason"] == e.reason
    img = D.render(dcfg, "bowl1")
    assert img.shape[1] == D.W and img.shape[0] >= 700 and has(img, D.WALL)
    assert any("ASSUMED" in t for t, _, _ in D._lines(dcfg, D._case(dcfg, "bowl1"), v, None))


def test_no_rim_and_no_default_says_unmeasured(dcfg):
    set_cup(dcfg, rim_mm=None)
    dcfg.plate.rim.default_rim_mm = 0.0
    v = D.verdict(dcfg, "bowl1")
    assert not v["feasible"] and v["reason"] == "unmeasured"
    assert D.render(dcfg, "bowl1").shape[1] == D.W        # a picture that says so, not an exception


def test_observation_and_sighting_inset(dcfg):
    set_cup(dcfg)
    plain = D.render(dcfg, "bowl1")
    shot = np.zeros((360, 640, 3), np.uint8)
    shot[:] = (17, 99, 201)
    img = D.render(dcfg, "bowl1", SimpleNamespace(status="ok", reason="", confidence=0.8, dx=0.0, dy=0.0), shot)
    assert has(img, (17, 99, 201)) and not has(plain, (17, 99, 201)) and img.shape[0] > plain.shape[0]
    assert (shot == (17, 99, 201)).all()                  # the picture handed in is not drawn on
    lost = D.verdict(dcfg, "bowl1", SimpleNamespace(status="lost", reason="partly_hidden", confidence=0.2))
    assert lost["feasible"] and any("partly_hidden" in r for r in lost["refusals"])
    moved = D.verdict(dcfg, "bowl1", SimpleNamespace(status="moved", reason="", confidence=0.7, dx=10.0, dy=0.0))
    assert moved["cup"]["x_mm"] == 210.0 and moved["cup"]["uncertainty_mm"] == pytest.approx(3 + 1.5 + 2)
    assert moved["margin_mm"] == pytest.approx(9.5)


def test_old_path_is_drawn_and_its_collisions_counted(dcfg, tmp_path):
    set_cup(dcfg)
    p = direct(dcfg)
    low = lower_tip(dcfg.kinematics, p.dip, p.dip_z + 4.0)           # the old scoop pushes through the bottom
    assert low is not None
    full = {**dcfg.arm.home_pose, **p.hover}
    poses = [{**full, **q} for q in (p.hover, p.dip, low)]
    TrajFile("scoop_bowl1", "path", make_waypoints(dcfg, poses)).save(tmp_path / "traj")
    case = D._case(dcfg, "bowl1")
    assert case.lib.has("scoop_bowl1"), case.lib.errors
    before = sorted((f.name, f.stat().st_mtime_ns) for f in (tmp_path / "traj").iterdir())
    img = D.render(dcfg, "bowl1")
    assert has(img, D.BROWN) and has(img[D.TOP_H + 140:, :D.LEFT - 20], D.RED)
    assert sorted((f.name, f.stat().st_mtime_ns) for f in (tmp_path / "traj").iterdir()) == before   # read-only
    assert D.verdict(dcfg, "bowl1")["feasible"]                      # the old path does not change the verdict


def test_cli_writes_only_where_told(dcfg, tmp_path, monkeypatch, capsys):
    from spoonful import main as M
    set_cup(dcfg)
    monkeypatch.setattr(C, "load", lambda path=None: dcfg)
    monkeypatch.setattr(M, "_system", lambda *a, **k: pytest.fail("dryrun must not build a System"))
    data = sorted(f.name for f in (ROOT / "data").iterdir())
    out = tmp_path / "pics" / "plan.png"
    M.main(["dryrun", "--bowl", "bowl1", "--out", str(out)])
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    line = capsys.readouterr().out.strip().splitlines()
    assert len(line) == 1 and line[0].startswith("bowl1 (applesauce): safe entry found") and str(out) in line[0]
    M.main(["dryrun", "--out", str(out)])                            # every bowl: the name goes into the file name
    assert (tmp_path / "pics" / "plan_bowl1.png").exists() and (tmp_path / "pics" / "plan_bowl2.png").exists()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and "no cup geometry: map the table first" in lines[1]
    assert sorted(f.name for f in (ROOT / "data").iterdir()) == data


def test_import_does_not_load_the_arm():
    code = ("import sys; import spoonful.brain.rim_dryrun, spoonful.main; "
            "bad = [m for m in sys.modules if m in ('spoonful.arm.arm', 'spoonful.arm.backend', 'spoonful.system') "
            "or m.startswith('spoonful.perception')]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
