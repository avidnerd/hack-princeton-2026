"""Preflight: every red item blocks the demo and prints its fix. Used by `python -m spoonful demo` and
scripts/preflight.py."""
from __future__ import annotations

import time
from pathlib import Path

from .arm import kinematics as K
from .arm.backend import make_backend
from .arm.trajectory import Library
from .config import JOINTS, Config

REQUIRED_PATHS = ("scoop", "lift")
REQUIRED_POSES = ("rest", "pre_delivery", "mouth_nominal")
MIN_VOLTS, MAX_VOLTS = 6.5, 8.5


def _age_h(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 3600 if p.exists() else 1e9


def run_checks(cfg: Config, backend_kind: str, camera: bool = True) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok_all = True

    def item(ok: bool, label: str, fix: str = "", warn: bool = False) -> None:
        nonlocal ok_all
        tag = "GREEN" if ok else ("AMBER" if warn else "RED  ")
        lines.append(f"[{tag}] {label}" + ("" if ok or not fix else f"\n         fix: {fix}"))
        if not ok and not warn:
            ok_all = False

    # arm + power
    b = make_backend(backend_kind, [cfg.arm.servo_map[j] for j in JOINTS])
    try:
        b.connect()
        v = b.voltage()
        item(v is not None and MIN_VOLTS <= v <= MAX_VOLTS, f"arm power: {v} V",
             "flip the rocker switch on; check the 7.5 V 6 A adapter")
        r = b.read_all([cfg.arm.servo_map[j] for j in JOINTS])
        dead = [j for j in JOINTS if r[cfg.arm.servo_map[j]] is None]
        item(not dead, f"servo reads: {'all 6' if not dead else 'missing ' + ','.join(dead)}",
             "reseat the servo bus cable; check servo_map in config.yaml")
        if not dead:
            pose = {j: (r[cfg.arm.servo_map[j]] - cfg.arm.center_units) / cfg.arm.units_per_deg for j in JOINTS}
            out = [j for j in JOINTS if not cfg.arm.soft_limits_deg[j][0] <= pose[j] <= cfg.arm.soft_limits_deg[j][1]]
            item(not out, "arm inside soft limits" if not out else f"outside soft limits: {out}",
                 "move the arm by hand (power off) back into range")
        b.close()
    except Exception as e:
        item(False, f"arm connect ({backend_kind}): {e}", "USB cable in? power on? try --backend raw")

    # trajectories
    d = cfg.path(cfg.arm.sim_trajectories_dir if backend_kind == "sim" else cfg.arm.trajectories_dir)
    lib = Library(cfg, d)
    miss = lib.missing(REQUIRED_PATHS + REQUIRED_POSES)
    item(not miss, f"trajectories in {d.name}/" + ("" if not miss else f": missing {miss}"),
         "python -m spoonful teach record <name> [--pose]")
    item(not lib.errors, "trajectories inside soft limits" if not lib.errors else f"invalid: {lib.errors}",
         "re-teach those paths inside the limits")
    if lib.has("present") or lib.has("mouth_nominal"):
        mouth = lib.pose("mouth_nominal") if lib.has("mouth_nominal") else lib.path("present")[-1]
        item(K.in_face_zone(cfg.kinematics, mouth), "mouth_nominal inside the face zone",
             "check kinematics link lengths / signs, or widen kinematics.face_zone")
    if lib.has("pre_delivery"):
        item(not K.in_face_zone(cfg.kinematics, lib.pose("pre_delivery")), "pre_delivery outside the face zone",
             "re-teach pre_delivery ~10 cm further back, or shrink kinematics.face_zone")
    for n in ("rest",) + REQUIRED_PATHS:
        if lib.has(n):
            item(not K.in_face_zone(cfg.kinematics, lib.path(n)[-1] if lib.items[n].kind == "path" else lib.pose(n)),
                 f"{n} ends outside the face zone", "re-teach it away from the diner")
    for n in REQUIRED_PATHS + REQUIRED_POSES:
        if lib.has(n):
            age = _age_h(d / f"{n}.json")
            item(age < cfg.calibration.max_age_h or backend_kind == "sim", f"{n} taught {age:.1f} h ago",
                 "re-teach today (arm or table may have moved)", warn=True)

    # calibration
    om = cfg.path(cfg.offsets.joint_map)
    item(om.exists(), "offset map present" if om.exists() else "no offset map (adaptive delivery off)",
         "python -m spoonful calibrate", warn=True)
    if om.exists():
        item(_age_h(om) < cfg.calibration.max_age_h, f"offset map {_age_h(om):.1f} h old",
             "re-run the offset calibration", warn=True)
        try:
            from .perception.offsets import OffsetMap
            m = OffsetMap.load(om)
            item(m.has_depth or not cfg.offsets.depth, "depth correction calibrated" if m.has_depth
                 else "offset map has no depth samples (leaning in/back not followed)",
                 "add the 'away' sample in Calibrate and re-fit", warn=True)
        except Exception as e:
            item(False, f"offset map unreadable: {e!r}", "re-run the offset calibration")

    # perception
    mp = cfg.path(cfg.face.model_path)
    item(mp.exists(), f"face model {mp.name}", "curl the face_landmarker.task into models/ (README)")
    if camera:
        try:
            import cv2
            from .perception.camera import open_capture
            cap = open_capture(cfg.camera.index, cfg.camera.api)
            ok, f = cap.read()
            cap.release()
            item(ok, f"camera {cfg.camera.index}" + (f" {f.shape[1]}x{f.shape[0]}" if ok else " no frame"),
                 "System Settings > Privacy > Camera: allow the terminal; or change camera.index")
        except Exception as e:
            item(False, f"camera: {e}", "pip install opencv-python")

    # optional peripherals
    from .io.esp32 import find_port
    port = find_port()
    item(port is not None, f"ESP32 {'on ' + port if port else 'not found (spacebar is STOP)'}", warn=True)
    return ok_all, lines
