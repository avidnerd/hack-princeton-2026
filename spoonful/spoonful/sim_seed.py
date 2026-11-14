"""Placeholder trajectories for the SIM backend only (data/trajectories_sim). Real trajectories are taught by hand
on the arm (data/trajectories). Poses come from a small planar IK so the FK geometry is self-consistent:
rest/home outside the face zone, pre_delivery just outside it, mouth_nominal inside it.
"""
from __future__ import annotations

import math

from .arm import kinematics as K
from .arm.trajectory import TrajFile, make_waypoints, now_iso
from .config import Config


def ik(cfg: Config, r: float, z: float, tool_pitch_deg: float, base_deg: float = 0.0) -> dict:
    """Tip at (r, z) with the tool at tool_pitch from vertical (90 = horizontal). Elbow-forward solution."""
    k = cfg.kinematics
    p3 = math.radians(tool_pitch_deg)
    wr = r - k.tool_mm * math.sin(p3) - k.tool_offset_mm * math.sin(p3 - math.pi / 2)
    wz = z - k.tool_mm * math.cos(p3) - k.tool_offset_mm * math.cos(p3 - math.pi / 2) - k.base_height_mm
    L1, L2 = k.upper_arm_mm, k.forearm_mm
    d2 = wr * wr + wz * wz
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1:
        raise ValueError(f"unreachable tip r={r} z={z} pitch={tool_pitch_deg}")
    delta = math.acos(c)
    t1 = math.atan2(wr, wz) - math.atan2(L2 * math.sin(delta), L1 + L2 * math.cos(delta))
    t2 = t1 + delta
    q = {"base": base_deg, "shoulder": math.degrees(t1), "elbow": math.degrees(delta),
         "wrist_pitch": tool_pitch_deg - math.degrees(t2), "wrist_roll": 0.0}
    # undo sign / zero conventions
    return {j: (v / k.sign.get(j, 1)) + k.zero_deg.get(j, 0) if j in k.sign else v for j, v in q.items()}


def seed(cfg: Config) -> list[str]:
    d = cfg.path(cfg.arm.sim_trajectories_dir)
    out = []
    P = lambda r, z, p, b=0.0: ik(cfg, r, z, p, b)  # noqa: E731
    rest = dict(cfg.arm.home_pose)
    above_bowl = P(190, 175, 110, -40)
    poses = {
        "rest": [rest],
        "pre_delivery": [P(215, 250, 90)],
        "mouth_nominal": [P(300, 250, 90)],
    }
    scoop_end = P(210, 185, 112, -40)
    paths = {
        "scoop": [rest, above_bowl, P(195, 135, 125, -40), P(220, 140, 120, -40), scoop_end],
        "scoop_small": [rest, above_bowl, P(205, 150, 120, -40), scoop_end],
        "scoop_full": [rest, above_bowl, P(195, 130, 128, -40), P(222, 138, 122, -40), scoop_end],
        "lift": [scoop_end, P(205, 225, 100, -30), P(210, 245, 92, -10), poses["pre_delivery"][0]],
        "present": [poses["pre_delivery"][0], P(260, 250, 90), poses["mouth_nominal"][0]],
        "retreat": [poses["pre_delivery"][0], P(205, 255, 92), rest],
    }
    for name, ps in {**poses, **paths}.items():
        kind = "pose" if name in poses else "path"
        speed = "near_face" if name in ("present", "pre_delivery", "mouth_nominal") else "normal"
        tf = TrajFile(name, kind, make_waypoints(cfg, ps, speed), {"recorded_at": now_iso(), "backend": "sim",
                                                                    "note": "IK placeholder, not taught"})
        tf.validate(cfg)
        tf.save(d)
        tip = K.fk(cfg.kinematics, ps[-1])
        out.append(f"{name:14s} {kind:4s} {len(ps):2d} wp  end tip=({tip[0]:.0f},{tip[1]:.0f},{tip[2]:.0f}) mm  "
                   f"face_zone={K.in_face_zone(cfg.kinematics, ps[-1])}")
    return out
