"""Forward kinematics for the xArm 1S as base yaw + a planar 3-link chain (shoulder, elbow, wrist pitch).

Only used for safety geometry (is the spoon tip in the face zone? is a move approaching the mouth?) and for
mm <-> joint conversions in the offset map. Wrist roll is ignored (spoon stays on the roll axis plane).

Convention: phi_k = sign_k * (q_k - zero_k), measured from vertical, accumulated along the chain.
x forward, y left, z up, origin on the table under the base axis.
"""
from __future__ import annotations

import math
from typing import Mapping

from ..config import KinCfg

Pose = Mapping[str, float]


def fk(kin: KinCfg, q: Pose) -> tuple[float, float, float]:
    s, z0 = kin.sign, kin.zero_deg

    def ang(j: str) -> float:
        return math.radians(s[j] * (q.get(j, 0.0) - z0[j]))

    yaw = ang("base")
    p1 = ang("shoulder")
    p2 = p1 + ang("elbow")
    p3 = p2 + ang("wrist_pitch")
    r = kin.upper_arm_mm * math.sin(p1) + kin.forearm_mm * math.sin(p2) + kin.tool_mm * math.sin(p3)
    z = kin.base_height_mm + kin.upper_arm_mm * math.cos(p1) + kin.forearm_mm * math.cos(p2) \
        + kin.tool_mm * math.cos(p3)
    # spoon sits tool_offset_mm "above" the tool axis (perpendicular, in the arm plane)
    r += kin.tool_offset_mm * math.sin(p3 - math.pi / 2)
    z += kin.tool_offset_mm * math.cos(p3 - math.pi / 2)
    return (r * math.cos(yaw), r * math.sin(yaw), z)


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)


def in_face_zone(kin: KinCfg, q: Pose) -> bool:
    x, y, z = fk(kin, q)
    b = kin.face_zone
    return b.x[0] <= x <= b.x[1] and b.y[0] <= y <= b.y[1] and b.z[0] <= z <= b.z[1]


def tip_distance(kin: KinCfg, a: Pose, b: Pose) -> float:
    return dist(fk(kin, a), fk(kin, b))
